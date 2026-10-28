"""Hand-crafted window features for tree models. Pure numpy (runs on the Pi too)."""

from __future__ import annotations

import warnings

import numpy as np

STATS = ["last", "mean", "min", "max", "std", "slope", "delta15", "measured_frac", "tsince"]


def window_features(values: np.ndarray, mask: np.ndarray, tsince: np.ndarray, score: np.ndarray) -> np.ndarray:
    """values/mask/tsince: (B, W, V); score: (B, W). Returns (B, 9V + 2) float32, NaN allowed."""
    v = values.astype(np.float64)
    W = v.shape[1]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        last = v[:, -1]
        mean = np.nanmean(v, axis=1)
        vmin = np.nanmin(v, axis=1)
        vmax = np.nanmax(v, axis=1)
        std = np.nanstd(v, axis=1)
        x = np.arange(W, dtype=float)[None, :, None]
        ok = ~np.isnan(v)
        n = np.maximum(ok.sum(axis=1), 1)
        xm = np.where(ok, x, 0).sum(axis=1) / n
        ym = np.where(ok, v, 0).sum(axis=1) / n
        dx = np.where(ok, x - xm[:, None], 0)
        dy = np.where(ok, v - ym[:, None], 0)
        sxx = (dx ** 2).sum(axis=1)
        slope = (dx * dy).sum(axis=1) / np.where(sxx > 0, sxx, np.nan)
        k = min(15, W - 1)
        delta15 = v[:, -1] - v[:, -1 - k]
        measured = mask.mean(axis=1)
        ts = tsince[:, -1]
        score_now = score[:, -1:]
        score_delta = (score[:, -1] - score[:, 0])[:, None]
    return np.concatenate([last, mean, vmin, vmax, std, slope, delta15, measured, ts, score_now, score_delta],
                          axis=1).astype(np.float32)


def feature_names(vitals: list[str]) -> list[str]:
    return [f"{s}_{v}" for s in STATS for v in vitals] + ["news2_now", "news2_delta"]
