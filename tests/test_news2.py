import numpy as np

from edews.core.news2 import news2, subscore


def test_band_edges():
    assert subscore("rr", np.array([8, 9, 11, 12, 20, 21, 24, 25])).tolist() == [3, 1, 1, 0, 0, 2, 2, 3]
    assert subscore("spo2", np.array([91, 92, 93, 94, 95, 96])).tolist() == [3, 2, 2, 1, 1, 0]
    assert subscore("sbp", np.array([90, 91, 100, 101, 110, 111, 219, 220])).tolist() == [3, 2, 2, 1, 1, 0, 0, 3]
    assert subscore("hr", np.array([40, 41, 50, 51, 90, 91, 110, 111, 130, 131])).tolist() == \
        [3, 1, 1, 0, 0, 1, 1, 2, 2, 3]
    assert subscore("temp", np.array([35.0, 35.1, 36.0, 36.1, 38.0, 38.1, 39.0, 39.1])).tolist() == \
        [3, 1, 1, 0, 0, 1, 1, 2]


def test_nan_stays_nan():
    assert np.isnan(subscore("hr", np.array([np.nan]))[0])


def _one(**kw):
    vals = {k: np.array([v], dtype=float) for k, v in kw.items() if k != "o2"}
    o2 = np.array([kw.get("o2", False)])
    return news2(vals, o2, ["hr", "rr", "spo2"])


def test_classes():
    # healthy -> low
    s, c, _ = _one(hr=75, rr=16, spo2=98, sbp=125, temp=37.0)
    assert s[0] == 0 and c[0] == 0
    # single parameter scoring 3 with low total -> low-medium
    s, c, _ = _one(hr=75, rr=26, spo2=98, sbp=125, temp=37.0)
    assert s[0] == 3 and c[0] == 1
    # total 5-6 -> medium:  hr 115 (2) + rr 22 (2) + spo2 95 (1) = 5
    s, c, _ = _one(hr=115, rr=22, spo2=95, sbp=125, temp=37.0)
    assert s[0] == 5 and c[0] == 2
    # supplemental oxygen adds 2 -> high:  5 + 2 = 7
    s, c, _ = _one(hr=115, rr=22, spo2=95, sbp=125, temp=37.0, o2=True)
    assert s[0] == 7 and c[0] == 3


def test_missing_required_is_unscorable_but_optional_is_zero():
    s, c, _ = _one(hr=75, rr=np.nan, spo2=98, sbp=125, temp=37.0)
    assert c[0] == -1 and np.isnan(s[0])
    s, c, _ = _one(hr=75, rr=16, spo2=98, sbp=np.nan, temp=np.nan)
    assert c[0] == 0 and s[0] == 0
