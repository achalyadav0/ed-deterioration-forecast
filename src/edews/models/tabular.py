"""XGBoost on hand-crafted window features: the strong, cheap baseline.

Features (see ``edews.core.tabular_features``) per vital: last, mean, min, max, std, slope,
change over the last 15 min, fraction of minutes measured, minutes since last measurement;
plus the current NEWS2-derived score and its change over the window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from edews.core.tabular_features import window_features as _wf
from edews.data.windows import Batch
from edews.models.base import Model


def window_features(b: Batch) -> np.ndarray:
    return _wf(b.values, b.mask, b.tsince, b.score)


def _collect(data, batch_size: int):
    X, yc, yf = [], [], []
    for _, b in data.batches(batch_size):
        X.append(window_features(b))
        yc.append(b.y_cur)
        yf.append(b.y_fut)
    return np.concatenate(X), np.concatenate(yc), np.concatenate(yf)


class XGBModel(Model):
    name = "xgb"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.models = {}

    def _make(self):
        import xgboost as xgb
        m = self.cfg.model
        return xgb.XGBClassifier(
            n_estimators=int(m.n_estimators), max_depth=int(m.max_depth), learning_rate=float(m.learning_rate),
            subsample=float(m.subsample), colsample_bytree=float(m.colsample_bytree),
            objective="multi:softprob", num_class=4, tree_method="hist", eval_metric="mlogloss",
            early_stopping_rounds=int(m.early_stopping_rounds), random_state=int(self.cfg.seed),
        )

    def fit(self, train, val, run_dir: Path, tracker) -> None:
        bs = int(self.cfg.train.batch_size) * 4
        Xtr, yc_tr, yf_tr = _collect(train, bs)
        Xva, yc_va, yf_va = _collect(val, bs)
        for head, ytr, yva in (("current", yc_tr, yc_va), ("future", yf_tr, yf_va)):
            counts = np.bincount(ytr, minlength=4).astype(float)
            w = (1.0 / np.sqrt(np.maximum(counts, 1)))[ytr]
            # make sure all 4 classes exist so predict_proba always returns (B, 4)
            missing = [c for c in range(4) if counts[c] == 0]
            Xh, yh = Xtr, ytr
            if missing:
                Xh = np.concatenate([Xtr, np.repeat(Xtr[:1], len(missing), 0)])
                yh = np.concatenate([ytr, missing])
                w = np.concatenate([w, np.full(len(missing), 1e-6)])
            vm = np.isin(yva, np.unique(yh))
            model = self._make()
            model.fit(Xh, yh, sample_weight=w, eval_set=[(Xva[vm], yva[vm])], verbose=False)
            self.models[head] = model
            tracker.log({f"xgb_{head}_best_iteration": int(getattr(model, "best_iteration", -1) or -1)})

    def predict(self, batch: Batch):
        X = window_features(batch)
        return (self.models["current"].predict_proba(X).astype(np.float32),
                self.models["future"].predict_proba(X).astype(np.float32))

    def save(self, run_dir: Path) -> None:
        for head, m in self.models.items():
            m.save_model(run_dir / f"xgb_{head}.json")

    def load(self, run_dir: Path) -> None:
        import xgboost as xgb
        for head in ("current", "future"):
            m = xgb.XGBClassifier()
            m.load_model(run_dir / f"xgb_{head}.json")
            self.models[head] = m

    def export(self, run_dir: Path, out_dir: Path, example: Batch) -> dict:
        self.save(out_dir)
        return {"format": "xgboost", "files": ["xgb_current.json", "xgb_future.json"],
                "note": "edge runtime needs `pip install xgboost` for this model"}
