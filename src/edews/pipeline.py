"""Pipeline orchestration with per-stage caching.

    edews run --stages all                          # everything, skipping cached stages
    edews run --stages train,evaluate model=gru     # new model; data stages are reused
    edews run --stages all labels.horizon_end=30    # relabels; ingest/featurize/split reused
    edews run --stages featurize --force featurize  # recompute one stage
    edews compare                                   # table of all evaluated runs
    edews synth --out data/raw_synth                # synthetic MC-MED-like data for dev/tests

Stage graph:
    ingest -> featurize -> label -> index -> train -> evaluate -> export
    ingest -> split ------------------^
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from edews.config import load_config, section
from edews.utils.cache import StageDir, stage_dir
from edews.utils.misc import get_logger, git_commit, set_seed

log = get_logger("edews.pipeline")

ORDER = ["ingest", "featurize", "label", "split", "index", "train", "evaluate", "export"]
DEPS = {
    "ingest": [], "featurize": ["ingest"], "label": ["featurize"], "split": ["ingest"],
    "index": ["featurize", "label", "split"], "train": ["index"], "evaluate": ["train"], "export": ["evaluate"],
}
# Bump a stage's version when you change its code in a way that changes its output.
VERSIONS = {"ingest": 1, "featurize": 2, "label": 1, "split": 1, "index": 1, "train": 1, "evaluate": 2, "export": 2}


def _params(cfg, stage: str) -> dict:
    c = cfg
    if stage == "ingest":
        return {"raw_dir": c.paths.raw_dir, **section(c, "data"), "ranges": section(c.clean, "ranges")}
    if stage == "featurize":
        return {**section(c, "resample"), "median": [list(c.clean.median_filter_vitals), c.clean.median_filter_minutes]}
    if stage == "label":
        return {**section(c, "labels"), "vitals": list(c.resample.vitals)}
    if stage == "split":
        return {**section(c, "split"), "seed": c.seed, "raw_dir": c.paths.raw_dir}
    if stage == "index":
        return {**section(c, "windows"), "required": list(c.labels.required_vitals), "seed": c.seed}
    if stage == "train":
        tr = {k: v for k, v in section(c, "train")["train"].items() if k not in ("device", "tracker")}
        return {**section(c, "model"), "train": tr, "seed": c.seed}
    if stage == "evaluate":
        return section(c, "eval")
    if stage == "export":
        return section(c, "export")
    raise KeyError(stage)


class Pipeline:
    def __init__(self, cfg, force: set[str] = frozenset()):
        self.cfg = cfg
        self.force = set(force)
        self._dirs: dict[str, StageDir] = {}
        self._ensured: set[str] = set()

    def dir(self, stage: str) -> StageDir:
        if stage not in self._dirs:
            up = [self.dir(d).key for d in DEPS[stage]]
            if stage in ("train", "evaluate", "export"):
                train_key = stage_dir("", "train", VERSIONS["train"], _params(self.cfg, "train"),
                                      [self.dir("index").key]).key
                base = Path(self.cfg.paths.runs_dir) / f"{self.cfg.model.name}_{train_key}"
                if stage == "train":
                    sd = StageDir("train", train_key, base)
                else:
                    sd = stage_dir(base, stage, VERSIONS[stage], _params(self.cfg, stage), up)
            else:
                sd = stage_dir(self.cfg.paths.work_dir, stage, VERSIONS[stage], _params(self.cfg, stage), up)
            self._dirs[stage] = sd
        return self._dirs[stage]

    def ensure(self, stage: str) -> StageDir:
        if stage in self._ensured:
            return self.dir(stage)
        for d in DEPS[stage]:
            self.ensure(d)
        sd = self.dir(stage)
        self._ensured.add(stage)
        if sd.done and stage not in self.force:
            log.info("[%s] cached -> %s", stage, sd.path)
            return sd
        log.info("[%s] running -> %s", stage, sd.path)
        t0 = time.time()
        if stage == "train" and sd.path.exists() and stage not in self.force:
            pass  # keep last.pt so an interrupted (e.g. Kaggle time-out) training run resumes
        else:
            sd.begin()
        meta = getattr(self, f"_run_{stage}")(sd)
        meta = {"stage": stage, "key": sd.key, "seconds": round(time.time() - t0, 1), "git": git_commit(), **(meta or {})}
        sd.finish(meta)
        self.force.discard(stage)
        return sd

    # ------------------------------------------------------------------ stages
    def _run_ingest(self, sd):
        from edews.data import ingest
        return ingest.run(self.cfg, sd.path)

    def _run_featurize(self, sd):
        from edews.data import featurize
        return featurize.run(self.cfg, self.dir("ingest").path, sd.path)

    def _run_label(self, sd):
        from edews.data import label
        return label.run(self.cfg, self.dir("featurize").path, sd.path)

    def _run_split(self, sd):
        from edews.data import split
        return split.run(self.cfg, self.dir("ingest").path, sd.path)

    def _run_index(self, sd):
        from edews.data import windows
        return windows.run(self.cfg, self.dir("featurize").path, self.dir("label").path, self.dir("split").path, sd.path)

    def data(self, split: int):
        from edews.data.windows import WindowData
        return WindowData(self.dir("featurize").path, self.dir("label").path, self.dir("index").path, split,
                          int(self.cfg.windows.length_min))

    def _model(self):
        from edews.models import build_model
        return build_model(self.cfg)

    def _run_train(self, sd):
        from edews.utils.tracking import Tracker
        sd.path.mkdir(parents=True, exist_ok=True)
        set_seed(int(self.cfg.seed))
        OmegaConf.save(self.cfg, sd.path / "config.yaml")
        model = self._model()
        tracker = Tracker(sd.path, sd.path.name, OmegaConf.to_container(self.cfg, resolve=True),
                          str(self.cfg.train.tracker))
        model.fit(self.data(0), self.data(1), sd.path, tracker)
        model.save(sd.path)
        tracker.close()
        return {"model": self.cfg.model.name, "n_params": model.n_params()}

    def _run_evaluate(self, sd):
        from edews.evaluate import run as ev
        model = self._model()
        model.load(self.dir("train").path)
        res = ev.run(self.cfg, model, self.data(1), self.data(2), sd.path)
        return {"test_transition_auprc": res["test"]["future_transition_auprc"]}

    def _run_export(self, sd):
        return export_bundle(self, sd.path)


def export_bundle(p: Pipeline, out: Path) -> dict:
    """Everything the Raspberry Pi needs, in one folder: model, normaliser, edge config, replay stream."""
    import polars as pl

    from edews.core.features import FeatureSpec
    from edews.data.featurize import load_arrays

    cfg = p.cfg
    model = p._model()
    model.load(p.dir("train").path)
    test = p.data(2)
    sel = np.arange(min(len(test), 64))
    info = model.export(p.dir("train").path, out, test.get(sel))
    shutil.copy(p.dir("index").path / "normalizer.json", out / "normalizer.json")

    metrics_file = p.dir("evaluate").path / "metrics.json"
    threshold = json.loads(metrics_file.read_text())["threshold_from_val"] if metrics_file.exists() else None
    spec = FeatureSpec.from_cfg(cfg)
    edge_cfg = {
        "model": cfg.model.name, "model_info": info, "vitals": list(spec.vitals),
        "feature_spec": {"ffill_limit": list(spec.ffill_limit), "median_filter": list(spec.median_filter),
                         "median_window": spec.median_window, "tsince_cap": spec.tsince_cap},
        "window_min": int(cfg.windows.length_min),
        "labels": OmegaConf.to_container(cfg.labels, resolve=True),
        "alert_class": int(cfg.eval.alert_class),
        "threshold_from_val": threshold,
        "escalation": OmegaConf.to_container(cfg.edge.escalation, resolve=True),
    }
    (out / "edge_config.json").write_text(json.dumps(edge_cfg, indent=2))

    # Replay stream: a few test visits, minute by minute, with the offline model's expected output
    arr = load_arrays(p.dir("featurize").path)
    offsets = pl.read_parquet(p.dir("featurize").path / "offsets.parquet")
    idx = test.index.with_row_index("i")
    events = idx.group_by("CSN").agg(pl.col("y_cur").max().alias("mx")).sort(["mx", "CSN"], descending=[True, False])
    chosen = events["CSN"].head(int(cfg.export.stream_visits)).to_list()
    n_lines = 0
    with gzip.open(out / "stream_sample.jsonl.gz", "wt") as f:
        for c in chosen:
            s, n = offsets.filter(pl.col("CSN") == c).select("start", "length").row(0)
            vi = idx.filter(pl.col("CSN") == c)
            exp = {}
            if vi.height:
                b = test.get(vi["i"].to_numpy())
                _, pf = model.predict(b)
                exp = {int(t): [float(x) for x in row] for t, row in zip(vi["t"].to_list(), pf)}
            raw = np.asarray(arr["raw_obs"][s:s + n])
            o2 = np.asarray(arr["on_o2"][s:s + n])
            for t in range(n):
                rec = {"bed": f"visit-{c}", "t": t,
                       "obs": {v: (None if np.isnan(raw[t, j]) else float(raw[t, j]))
                               for j, v in enumerate(spec.vitals)},
                       "on_o2": int(o2[t])}
                if t in exp:
                    rec["expected_p_fut"] = exp[t]
                f.write(json.dumps(rec) + "\n")
                n_lines += 1
    log.info("export bundle -> %s (%s)", out, info)
    return {"bundle": str(out), "stream_visits": len(chosen), "stream_minutes": n_lines, **info}


def compare(cfg) -> None:
    import polars as pl
    rows = []
    latest: dict[Path, Path] = {}   # newest evaluation per run (older eval versions are ignored)
    for mf in Path(cfg.paths.runs_dir).glob("*/evaluate/*/metrics.json"):
        run = mf.parents[2]
        if run not in latest or mf.stat().st_mtime > latest[run].stat().st_mtime:
            latest[run] = mf
    for run, mf in sorted(latest.items()):
        m = json.loads(mf.read_text())
        meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
        t, e, ci = m["test"], m["test_events"], m.get("test_ci95", {})
        rows.append({
            "run": run.name, "model": m["model"], "params": meta.get("n_params"),
            "trans_AUPRC": t["future_transition_auprc"], "AUPRC_CI": ci.get("future_transition_auprc"),
            "trans_AUROC": t["future_transition_auroc"], "prevalence": t["transition_prevalence"],
            "fut_macroF1": t["future_macro_f1"], "cur_macroF1": t["current_macro_f1"],
            "event_sens": e["event_sensitivity"], "lead_min": e["lead_time_median_min"],
            "false_alerts_ph": e["false_alerts_per_patient_hour"], "alert_time": e.get("alert_time_frac"),
        })
    if not rows:
        print("no evaluated runs yet")
        return
    df = pl.DataFrame(rows).sort("trans_AUPRC", descending=True, nulls_last=True)
    with pl.Config(tbl_cols=-1, tbl_width_chars=220, float_precision=3):
        print(df)
    df.with_columns(pl.col("AUPRC_CI").cast(pl.List(pl.Float64)).list.eval(pl.element().round(3)).cast(pl.List(pl.Utf8))
                    .list.join("-")).write_csv(Path(cfg.paths.runs_dir) / "comparison.csv")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="edews")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run pipeline stages (cached)")
    r.add_argument("--config", default="configs/config.yaml")
    r.add_argument("--stages", default="all", help="comma list or 'all'")
    r.add_argument("--force", default="", help="comma list of stages to recompute")
    r.add_argument("overrides", nargs="*", help="key=value config overrides")
    c = sub.add_parser("compare", help="compare evaluated runs")
    c.add_argument("--config", default="configs/config.yaml")
    c.add_argument("overrides", nargs="*")
    s = sub.add_parser("synth", help="write a synthetic MC-MED-like dataset")
    s.add_argument("--out", default="data/raw_synth")
    s.add_argument("--n-visits", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd == "synth":
        from edews.data.synthetic import generate
        print(generate(a.out, a.n_visits, a.seed))
        return
    cfg = load_config(a.config, a.overrides)
    if a.cmd == "compare":
        compare(cfg)
        return
    stages = ORDER if a.stages == "all" else [s.strip() for s in a.stages.split(",") if s.strip()]
    unknown = set(stages) - set(ORDER)
    if unknown:
        sys.exit(f"unknown stages: {unknown}")
    p = Pipeline(cfg, force={s for s in a.force.split(",") if s})
    for st in stages:
        p.ensure(st)


if __name__ == "__main__":
    main()
