import math

import numpy as np
import pandas as pd
import pytest

from addon_analytics.lift import profile_lift, two_proportion_z


def test_two_proportion_z_matches_hand_calculation():
    # p1 = 30/100, p0 = 100/1000, pooled p = 130/1100
    p = 130 / 1100
    expected = (0.3 - 0.1) / math.sqrt(p * (1 - p) * (1 / 100 + 1 / 1000))
    assert two_proportion_z(30, 100, 100, 1000) == pytest.approx(expected)
    assert expected == pytest.approx(5.907, abs=1e-3)
    assert two_proportion_z(10, 100, 100, 1000) == pytest.approx(0.0)
    assert two_proportion_z(5, 100, 200, 1000) < 0


def test_z_matches_statsmodels_if_available():
    sm = pytest.importorskip("statsmodels.stats.proportion")
    z, _ = sm.proportions_ztest([30, 100], [100, 1000])
    assert two_proportion_z(30, 100, 100, 1000) == pytest.approx(z)


def test_profile_lift_and_filters():
    # seed: 100 merchants, 40 chains; base: 1000 merchants, 100 chains -> lift 4
    feat = pd.DataFrame({"merchant_id": range(1100),
                         "branches": ["2-3"] * 40 + ["1"] * 60 + ["2-3"] * 100 + ["1"] * 900,
                         "tier": ["x"] * 3 + ["y"] * 97 + ["x"] * 2 + ["y"] * 998})
    seed = np.arange(1100) < 100
    out = profile_lift(feat, seed, ~seed, feature_cols=["branches", "tier"]).set_index(["feature", "value"])
    assert out.loc[("branches", "2-3"), "lift"] == pytest.approx(4.0)
    assert out.loc[("branches", "2-3"), "keep"]
    # tier=x: lift 15 and |z| > 2, but only 3 seed merchants -> dropped by the n >= 5 rule
    assert out.loc[("tier", "x"), "lift"] == pytest.approx(15.0)
    assert abs(out.loc[("tier", "x"), "z"]) > 2
    assert not out.loc[("tier", "x"), "keep"]
