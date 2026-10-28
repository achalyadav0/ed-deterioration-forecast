"""Feature construction: forward-fill limits, no backward fill, causality, online == offline."""

import numpy as np
import pytest

from edews.core.features import (
    FeatureSpec,
    OnlineFeaturizer,
    causal_median,
    ffill_with_limit,
    transform_visit,
)

SPEC = FeatureSpec(vitals=("hr", "rr", "spo2", "sbp", "temp"), ffill_limit=(5, 5, 5, 60, 360),
                   median_filter=(True, True, True, False, False), median_window=3, tsince_cap=360)


def random_obs(T=300, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal([80, 16, 97, 120, 37], [10, 3, 2, 15, 0.5], size=(T, 5))
    obs[rng.random((T, 5)) < np.array([0.05, 0.05, 0.1, 0.95, 0.995])] = np.nan
    obs[:7, 3] = np.nan    # leading gap: must stay missing (no backward fill)
    return obs.astype(np.float32).astype(np.float64)


def test_ffill_limit_and_no_backfill():
    x = np.array([np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, 2.0, np.nan])
    filled, gap = ffill_with_limit(x, 2)
    assert np.isnan(filled[:2]).all()                       # nothing before the first observation
    assert filled[2:5].tolist() == [1.0, 1.0, 1.0]
    assert np.isnan(filled[5])                              # beyond the limit
    assert filled[6:].tolist() == [2.0, 2.0]
    assert gap[2:7].tolist() == [0, 1, 2, 3, 0]


def test_causal_median_ignores_future():
    x = np.array([10.0, 10.0, 500.0, 10.0])
    out = causal_median(x, 3)
    assert out[2] == 10.0                                  # spike suppressed using past only
    y = x.copy()
    y[3] = 9999.0
    assert causal_median(y, 3)[2] == out[2]                 # changing the future doesn't change t=2


def test_transform_is_causal():
    """Features at minute t must not change when anything after t changes."""
    obs = random_obs()
    full = transform_visit(obs, SPEC)
    for t in (10, 77, 150, 299):
        part = transform_visit(obs[: t + 1], SPEC)
        for a, b in zip(full, part):
            np.testing.assert_array_equal(a[: t + 1], b)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_online_equals_offline(seed):
    obs = random_obs(seed=seed)
    v, m, ts = transform_visit(obs, SPEC)
    f = OnlineFeaturizer(SPEC)
    for t in range(len(obs)):
        ov, om, ots = f.step(obs[t])
        np.testing.assert_array_equal(ov, v[t])
        np.testing.assert_array_equal(om, m[t])
        np.testing.assert_array_equal(ots, ts[t])
