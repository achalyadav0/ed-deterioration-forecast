"""Minute-grid feature construction, identical offline (whole visit) and online (one minute at a time).

Input per visit: ``obs`` of shape (T, V), one row per minute, NaN where a vital was not measured
in that minute (values already range-checked). Output per minute and vital:

* ``value``  : spike-filtered value carried forward up to ``ffill_limit`` minutes, else NaN
* ``mask``   : 1 if actually measured this minute, else 0
* ``tsince`` : minutes since the last real measurement, capped at ``cap``

Only *past* information is ever used: forward fill only, causal median filter. No backward
fill, no interpolation (both would leak the future into the input window).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class FeatureSpec:
    vitals: tuple[str, ...]
    ffill_limit: tuple[int, ...]          # minutes, per vital
    median_filter: tuple[bool, ...]       # per vital
    median_window: int                    # minutes
    tsince_cap: int                       # minutes

    @classmethod
    def from_cfg(cls, cfg) -> FeatureSpec:
        vitals = tuple(cfg.resample.vitals)
        return cls(
            vitals=vitals,
            ffill_limit=tuple(int(cfg.resample.ffill_limit_min[v]) for v in vitals),
            median_filter=tuple(v in set(cfg.clean.median_filter_vitals) for v in vitals),
            median_window=int(cfg.clean.median_filter_minutes),
            tsince_cap=int(cfg.resample.time_since_cap_min),
        )


def causal_median(obs: np.ndarray, window: int) -> np.ndarray:
    """Median of the observations in the last ``window`` minutes, evaluated only where observed."""
    if window <= 1:
        return obs.copy()
    padded = np.concatenate([np.full(window - 1, np.nan), obs])
    win = sliding_window_view(padded, window)          # (T, window), row t = minutes t-w+1..t
    out = np.full_like(obs, np.nan)
    observed = ~np.isnan(obs)
    if observed.any():
        out[observed] = np.nanmedian(win[observed], axis=1)
    return out


def ffill_with_limit(x: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Forward fill a 1-D series for at most ``limit`` minutes. Returns (filled, minutes_since_obs)."""
    t = np.arange(len(x))
    observed = ~np.isnan(x)
    last = np.where(observed, t, -1)
    last = np.maximum.accumulate(last)
    gap = np.where(last >= 0, t - last, np.iinfo(np.int64).max).astype(np.int64)
    filled = np.full_like(x, np.nan)
    ok = (last >= 0) & (gap <= limit)
    filled[ok] = x[last[ok]]
    return filled, gap


def transform_visit(obs: np.ndarray, spec: FeatureSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offline: whole visit at once. Returns (value, mask, tsince), each (T, V) float32."""
    T, V = obs.shape
    value = np.full((T, V), np.nan, dtype=np.float32)
    mask = (~np.isnan(obs)).astype(np.float32)
    tsince = np.empty((T, V), dtype=np.float32)
    for j in range(V):
        x = obs[:, j].astype(np.float64)
        if spec.median_filter[j]:
            x = causal_median(x, spec.median_window)
        filled, gap = ffill_with_limit(x, spec.ffill_limit[j])
        value[:, j] = filled
        tsince[:, j] = np.minimum(gap, spec.tsince_cap)
    return value, mask, tsince


class OnlineFeaturizer:
    """Online: one minute at a time (Raspberry Pi). Must equal ``transform_visit`` row by row."""

    def __init__(self, spec: FeatureSpec):
        self.spec = spec
        V = len(spec.vitals)
        self._raw = [deque([np.nan] * max(spec.median_window - 1, 0), maxlen=max(spec.median_window - 1, 0))
                     for _ in range(V)]
        self._last_val = np.full(V, np.nan)
        self._last_t = np.full(V, -1, dtype=np.int64)
        self.t = -1

    def step(self, obs_row: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.t += 1
        spec, t = self.spec, self.t
        V = len(spec.vitals)
        value = np.full(V, np.nan, dtype=np.float32)
        mask = np.zeros(V, dtype=np.float32)
        tsince = np.empty(V, dtype=np.float32)
        for j in range(V):
            x = float(obs_row[j])
            observed = not np.isnan(x)
            if observed:
                mask[j] = 1.0
                v = x
                if spec.median_filter[j] and spec.median_window > 1:
                    v = float(np.nanmedian(list(self._raw[j]) + [x]))
                self._last_val[j] = v
                self._last_t[j] = t
            if self._raw[j].maxlen:
                self._raw[j].append(x)
            gap = t - self._last_t[j] if self._last_t[j] >= 0 else np.iinfo(np.int64).max
            if self._last_t[j] >= 0 and gap <= spec.ffill_limit[j]:
                value[j] = self._last_val[j]
            tsince[j] = min(gap, spec.tsince_cap)
        return value, mask, tsince
