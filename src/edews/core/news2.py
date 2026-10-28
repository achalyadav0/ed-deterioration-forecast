"""NEWS2 scoring (Royal College of Physicians, 2017), SpO2 scale 1.

Consciousness (ACVPU) is not available in MC-MED numerics, so this is a NEWS2-derived score
without the consciousness item. Report it as such in the thesis.

Response bands used for the 4 classes:
    0 low          aggregate 0-4 and no single parameter scoring 3
    1 low-medium   aggregate 0-4 but at least one parameter scores 3
    2 medium       aggregate 5-6
    3 high         aggregate >= 7
Invalid / unscorable minutes get class -1.
"""

from __future__ import annotations

import numpy as np

# Upper bounds (inclusive) of each band and the points awarded. A value above the last bound
# gets the last points entry.
_TABLES: dict[str, tuple[list[float], list[int]]] = {
    "rr":   ([8, 11, 20, 24], [3, 1, 0, 2, 3]),
    "spo2": ([91, 93, 95], [3, 2, 1, 0]),
    "sbp":  ([90, 100, 110, 219], [3, 2, 1, 0, 3]),
    "hr":   ([40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3]),
    "temp": ([35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2]),
}
O2_POINTS = 2
CLASS_NAMES = ["low", "low-medium", "medium", "high"]
N_CLASSES = 4


def subscore(vital: str, x: np.ndarray) -> np.ndarray:
    """Points for one vital. NaN in -> NaN out."""
    edges, pts = _TABLES[vital]
    x = np.asarray(x, dtype=float)
    idx = np.searchsorted(np.asarray(edges, dtype=float), np.nan_to_num(x, nan=0.0), side="left")
    out = np.asarray(pts, dtype=float)[idx]
    out[np.isnan(x)] = np.nan
    return out


def news2(values: dict[str, np.ndarray], on_oxygen: np.ndarray | None,
          required: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate NEWS2-derived score and class per time step.

    Args:
        values: vital name -> array (latest available value per minute, NaN if none).
        on_oxygen: boolean array, True when on supplemental O2 (None = item not used).
        required: vitals that must be present for the minute to be scorable.

    Returns:
        total score (float, NaN if not scorable), class (int, -1 if not scorable),
        any_three (bool).
    """
    n = len(next(iter(values.values())))
    subs = {v: subscore(v, values[v]) for v in values if v in _TABLES}
    stacked = np.vstack([subs[v] for v in subs]) if subs else np.zeros((0, n))
    total = np.nansum(stacked, axis=0)
    if on_oxygen is not None:
        total = total + O2_POINTS * np.asarray(on_oxygen, dtype=float)
    any_three = np.any(stacked == 3, axis=0)

    valid = np.ones(n, dtype=bool)
    for v in required:
        valid &= ~np.isnan(subs[v])

    cls = np.zeros(n, dtype=np.int64)
    cls[any_three] = 1
    cls[total >= 5] = 2
    cls[total >= 7] = 3
    cls[~valid] = -1
    total = np.where(valid, total, np.nan)
    return total, cls, any_three
