"""Common model interface. Training, evaluation, export and the Pi only ever talk to this.

Swapping TCN -> GRU -> XGBoost is a config change (``model=gru``), not a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from edews.data.windows import Batch, WindowData


class Model(ABC):
    name: str = "base"

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def fit(self, train: WindowData, val: WindowData, run_dir: Path, tracker) -> None: ...

    @abstractmethod
    def predict(self, batch: Batch) -> tuple[np.ndarray, np.ndarray]:
        """Return (p_current, p_future), each (B, 4) class probabilities."""

    def save(self, run_dir: Path) -> None:  # baselines have nothing to save
        pass

    def load(self, run_dir: Path) -> None:
        pass

    def export(self, run_dir: Path, out_dir: Path, example: Batch) -> dict:
        """Write an edge-deployable artefact into out_dir. Returns info for the bundle manifest."""
        return {"format": "builtin", "model": self.name}

    def n_params(self) -> int | None:
        return None


def build_model(cfg) -> Model:
    name = cfg.model.name
    if name in ("persistence", "news2_rule"):
        from edews.models.baselines import NEWS2RuleModel, PersistenceModel
        return {"persistence": PersistenceModel, "news2_rule": NEWS2RuleModel}[name](cfg)
    if name == "xgb":
        from edews.models.tabular import XGBModel
        return XGBModel(cfg)
    if name in ("tcn", "gru"):
        from edews.models.torch_models import TorchModel
        return TorchModel(cfg)
    raise ValueError(f"unknown model {name}")
