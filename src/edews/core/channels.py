"""Turn (value, mask, tsince) into the model input tensor. Shared by training and the Pi.

Channel layout (C = 3 * V): [normalised values | measured masks | scaled time-since].
Missing values become 0 after normalisation (= the training mean); the mask tells the model
the value is not real.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Normalizer:
    vitals: list[str]
    mean: np.ndarray
    std: np.ndarray
    tsince_cap: float

    @classmethod
    def fit(cls, vitals: list[str], values: np.ndarray, tsince_cap: float) -> Normalizer:
        """Fit on *training-split* values only (rows x V, NaN allowed)."""
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std[~np.isfinite(std) | (std < 1e-6)] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        return cls(list(vitals), mean.astype(np.float32), std.astype(np.float32), float(tsince_cap))

    def assemble(self, value: np.ndarray, mask: np.ndarray, tsince: np.ndarray) -> np.ndarray:
        """(..., V) x3 -> (..., 3V) float32."""
        z = (value - self.mean) / self.std
        z = np.where(np.isnan(z), 0.0, z)
        ts = np.log1p(tsince) / np.log1p(self.tsince_cap)
        return np.concatenate([z, mask, ts], axis=-1).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "vitals": self.vitals, "mean": self.mean.tolist(), "std": self.std.tolist(),
            "tsince_cap": self.tsince_cap}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Normalizer:
        d = json.loads(Path(path).read_text())
        return cls(d["vitals"], np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32),
                   float(d["tsince_cap"]))
