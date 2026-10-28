"""Per-patient streaming inference + escalation, driven one minute at a time.

    bundle = Bundle.load("path/to/export_bundle", int8=True)
    mon = PatientMonitor(bundle)
    out = mon.step({"hr": 92, "rr": None, "spo2": 95, "sbp": None, "temp": None}, on_o2=0)

Feature construction, normalisation and NEWS2 scoring reuse ``edews.core`` -- the same code
that built the training data -- so offline and on-device inputs are identical.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edews.core.channels import Normalizer
from edews.core.features import FeatureSpec, OnlineFeaturizer
from edews.core.news2 import news2

LEVELS = {"NONE": 0, "NOTIFY_DOCTOR": 1, "CALL_MET": 2}


class Bundle:
    def __init__(self, path: Path, cfg: dict, norm: Normalizer, runner, kind: str):
        self.path, self.cfg, self.norm, self.runner, self.kind = path, cfg, norm, runner, kind
        fs = cfg["feature_spec"]
        self.spec = FeatureSpec(tuple(cfg["vitals"]), tuple(fs["ffill_limit"]), tuple(fs["median_filter"]),
                                int(fs["median_window"]), int(fs["tsince_cap"]))
        self.window = int(cfg["window_min"])

    @classmethod
    def load(cls, path: str | Path, int8: bool = False, threads: int = 1) -> Bundle:
        path = Path(path)
        cfg = json.loads((path / "edge_config.json").read_text())
        norm = Normalizer.load(path / "normalizer.json")
        fmt = cfg["model_info"]["format"]
        if fmt == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            f = "model.int8.onnx" if int8 and (path / "model.int8.onnx").exists() else "model.onnx"
            sess = ort.InferenceSession(str(path / f), so, providers=["CPUExecutionProvider"])
            runner = lambda w: sess.run(None, {"x": w["x"]})
            kind = f
        elif fmt == "xgboost":
            import xgboost as xgb

            from edews.core.tabular_features import window_features
            m = {}
            for head in ("current", "future"):
                m[head] = xgb.XGBClassifier()
                m[head].load_model(path / f"xgb_{head}.json")

            def runner(w):
                X = window_features(w["values"], w["mask"], w["tsince"], w["score"])
                return m["current"].predict_proba(X), m["future"].predict_proba(X)
            kind = "xgboost"
        else:
            raise ValueError(f"bundle format {fmt} has no edge runner (baselines need no model file)")
        return cls(path, cfg, norm, runner, kind)


@dataclass
class StepResult:
    t: int
    ready: bool
    news2: float | None
    current_class: int
    p_fut: list[float] | None
    decision: str
    reason: str
    infer_ms: float


class PatientMonitor:
    def __init__(self, bundle: Bundle):
        self.b = bundle
        W = bundle.window
        self.feat = OnlineFeaturizer(bundle.spec)
        self.values, self.mask, self.tsince = deque(maxlen=W), deque(maxlen=W), deque(maxlen=W)
        self.score = deque(maxlen=W)
        self.cls_hist = deque(maxlen=int(bundle.cfg["labels"]["current_lookback_min"]))
        self.last_alert = {"NOTIFY_DOCTOR": -10 ** 9, "CALL_MET": -10 ** 9}
        self.t = -1

    def step(self, obs: dict, on_o2: int = 0) -> StepResult:
        b, cfg = self.b, self.b.cfg
        self.t += 1
        row = np.array([np.nan if obs.get(v) is None else float(obs[v]) for v in b.spec.vitals])
        val, msk, ts = self.feat.step(row)
        self.values.append(val)
        self.mask.append(msk)
        self.tsince.append(ts)

        lab = cfg["labels"]
        vals = {v: np.array([val[i]], dtype=float) for i, v in enumerate(b.spec.vitals)}
        s, c, _ = news2(vals, np.array([bool(on_o2)]) if lab["use_o2_item"] else None, list(lab["required_vitals"]))
        self.score.append(float(s[0]))
        self.cls_hist.append(int(c[0]))
        cur = max(self.cls_hist) if c[0] >= 0 else -1

        if len(self.values) < b.window:
            return StepResult(self.t, False, _f(s[0]), cur, None, "NONE", "warming up", 0.0)

        v = np.asarray(self.values, np.float32)[None]
        m = np.asarray(self.mask, np.float32)[None]
        tsn = np.asarray(self.tsince, np.float32)[None]
        w = {"x": b.norm.assemble(v, m, tsn), "values": v, "mask": m, "tsince": tsn,
             "score": np.asarray(self.score, np.float32)[None]}
        t0 = time.perf_counter()
        _, p_fut = b.runner(w)
        ms = (time.perf_counter() - t0) * 1000
        p = np.asarray(p_fut)[0]
        decision, reason = self._escalate(cur, p)
        return StepResult(self.t, True, _f(s[0]), cur, [float(x) for x in p], decision, reason, ms)

    def _escalate(self, cur: int, p: np.ndarray) -> tuple[str, str]:
        """Rule layer on top of the model. Replace with an RL / learned policy later."""
        cfg = self.b.cfg
        esc = cfg["escalation"]
        ac = int(cfg["alert_class"])
        notify_thr = cfg.get("threshold_from_val") or esc["notify_threshold"]
        p_det, p_high = float(p[ac:].sum()), float(p[3])
        if cur == 3:
            want, why = "CALL_MET", "current NEWS2 class high"
        elif p_high >= esc["met_threshold"]:
            want, why = "CALL_MET", f"P(high in horizon)={p_high:.2f}"
        elif p_det >= notify_thr:
            want, why = "NOTIFY_DOCTOR", f"P(>=medium in horizon)={p_det:.2f}"
        else:
            return "NONE", ""
        if self.t - self.last_alert[want] < int(esc["cooldown_min"]):
            return "NONE", f"suppressed (cooldown) {why}"
        self.last_alert[want] = self.t
        if want == "CALL_MET":
            self.last_alert["NOTIFY_DOCTOR"] = self.t
        return want, why


def _f(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)
