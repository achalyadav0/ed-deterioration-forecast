"""Rule baselines that every learned model has to beat.

``persistence``: the future class equals the current class.
``news2_rule``: escalate one class when the NEWS2 score rose quickly; otherwise persistence.

The current class at time t is computable from the inputs (NEWS2 of the latest vitals), so both
baselines are (near-)perfect on the *current* task. That is the point: the current-state head
is a sanity check, the future head is the research question.
"""

from __future__ import annotations

import numpy as np

from edews.data.windows import Batch
from edews.models.base import Model


def _onehot(c: np.ndarray, soft: np.ndarray | None = None) -> np.ndarray:
    p = np.zeros((len(c), 4), dtype=np.float32)
    p[np.arange(len(c)), np.clip(c, 0, 3)] = 1.0
    if soft is not None:   # a continuous risk score for ranking metrics (AUROC/AUPRC)
        p = 0.9 * p + 0.1 * soft
    return p


def _score_risk(score_t: np.ndarray) -> np.ndarray:
    """Soft class distribution from the NEWS2 score, used only to break ties in ranking."""
    s = np.nan_to_num(score_t, nan=0.0)
    hi = np.clip((s - 4) / 6, 0, 1)
    return np.stack([1 - hi, np.zeros_like(hi), hi * 0.5, hi * 0.5], axis=1).astype(np.float32)


class PersistenceModel(Model):
    name = "persistence"

    def fit(self, train, val, run_dir, tracker) -> None:
        pass

    def predict(self, batch: Batch):
        cur = batch.y_cur
        soft = _score_risk(batch.score[:, -1])
        return _onehot(cur), _onehot(cur, soft)


class NEWS2RuleModel(Model):
    name = "news2_rule"

    def fit(self, train, val, run_dir, tracker) -> None:
        pass

    def predict(self, batch: Batch):
        cur = batch.y_cur
        w = int(self.cfg.model.rise_window_min)
        s_now = batch.score[:, -1]
        s_then = batch.score[:, -1 - min(w, batch.score.shape[1] - 1)]
        rising = np.nan_to_num(s_now - s_then, nan=0.0) >= float(self.cfg.model.rise_points)
        fut = np.where(rising, np.minimum(cur + 1, 3), cur)
        soft = _score_risk(s_now)
        return _onehot(cur), _onehot(fut, soft)
