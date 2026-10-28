"""Current / future labels from a per-minute NEWS2 class series.

current[t] = worst valid class in minutes [t-L+1, t]; requires class[t] itself to be valid
future[t]  = worst valid class in minutes (t+a, t+b]; requires enough valid minutes in that span

The future label only ever looks at minutes strictly after t, and model inputs only ever use
minutes up to and including t. ``tests/test_labels.py`` checks this.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def current_label(cls: np.ndarray, lookback: int) -> np.ndarray:
    T = len(cls)
    if lookback <= 1:
        return cls.copy()
    padded = np.concatenate([np.full(lookback - 1, -1, dtype=cls.dtype), cls])
    out = sliding_window_view(padded, lookback).max(axis=1)[:T]
    out[cls < 0] = -1
    return out


def future_label(cls: np.ndarray, start: int, end: int, min_coverage: float) -> np.ndarray:
    """Worst class in (t+start, t+end]. -1 where the horizon is not observed well enough."""
    if not 0 <= start < end:
        raise ValueError("need 0 <= horizon_start < horizon_end")
    T = len(cls)
    n = end - start
    padded = np.concatenate([cls, np.full(end + 1, -1, dtype=cls.dtype)])
    win = sliding_window_view(padded, n)[start + 1: start + 1 + T]   # row t = minutes t+start+1 .. t+end
    worst = win.max(axis=1)
    coverage = (win >= 0).sum(axis=1) / n
    worst[(coverage < min_coverage) | (worst < 0)] = -1
    return worst
