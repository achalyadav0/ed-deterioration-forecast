"""Labels: the future label uses only minutes strictly after t; the current label only <= t."""

import numpy as np

from edews.core.labels import current_label, future_label


def test_future_label_window():
    cls = np.zeros(100, dtype=np.int64)
    cls[50] = 3
    fut = future_label(cls, start=30, end=60, min_coverage=0.5)
    # minute 50 lies in (t+30, t+60] for t in [-10, 19]
    assert (fut[:20] == 3).all()
    assert fut[20] == 0
    # near the end there is not enough horizon left -> invalid
    assert (fut[-35:] == -1).all()


def test_future_label_ignores_past_and_present():
    rng = np.random.default_rng(0)
    cls = rng.integers(0, 4, 200)
    fut = future_label(cls, 30, 60, 0.5)
    for t in (40, 90, 120):
        changed = cls.copy()
        changed[: t + 31] = rng.integers(0, 4, t + 31)   # rewrite everything up to t+30
        assert future_label(changed, 30, 60, 0.5)[t] == fut[t]


def test_current_label_ignores_future():
    rng = np.random.default_rng(1)
    cls = rng.integers(0, 4, 200)
    cur = current_label(cls, 5)
    for t in (10, 100, 150):
        changed = cls.copy()
        changed[t + 1:] = 3
        assert current_label(changed, 5)[t] == cur[t]
    assert cur[10] == cls[6:11].max()


def test_coverage_requirement():
    cls = np.full(100, -1)
    cls[::4] = 0                       # only 25% of minutes scorable
    assert (future_label(cls, 30, 60, 0.5) == -1).all()
    assert (future_label(cls, 30, 60, 0.2)[:30] == 0).all()
