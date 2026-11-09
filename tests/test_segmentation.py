import pandas as pd

from addon_analytics.segmentation import segment_merchants

AS_OF = pd.Timestamp("2025-01-01")


def _rows(mid, dates, n):
    return [{"merchant_id": mid, "date": pd.Timestamp(d), "use_case": "transactional", "n_attempted": n}
            for d in dates]


def _weekly(mid, n_weeks, per_day, days_per_week=2):
    """`days_per_week` send days in each of the last `n_weeks` 7-day blocks before AS_OF."""
    dates = [AS_OF - pd.Timedelta(days=7 * w + 1 + k) for w in range(n_weeks) for k in range(days_per_week)]
    return _rows(mid, dates, per_day)


def _seg(rows, ids):
    out = segment_merchants(pd.DataFrame(rows), AS_OF, merchant_ids=ids)
    return out.set_index("merchant_id")


def test_segments_on_hand_made_data():
    rows = []
    rows += _weekly(1, 14, 30)                                            # regular, ~780 msgs -> S1
    rows += _rows(2, ["2024-12-10"], 600)                                 # one burst -> S2
    rows += _rows(3, ["2024-10-05", "2024-11-05", "2024-12-05"], 1)       # monthly, tiny -> S3
    rows += _rows(4, ["2024-12-20"], 5)                                   # tiny, one month -> S4
    rows += _rows(5, ["2024-01-15"], 50)                                  # outside window -> S0
    seg = _seg(rows, [1, 2, 3, 4, 5])
    assert seg["segment"].to_dict() == {1: "S1", 2: "S2", 3: "S3", 4: "S4", 5: "S0"}
    assert seg.loc[1, "regular_weeks"] == 14


def test_weekly_regularity_threshold():
    # >= 500 messages with 11 regular weeks -> S1; with 10 regular weeks -> S2
    a = _weekly(1, 11, 25)                     # 11 weeks x 2 days x 25 = 550
    b = _weekly(2, 10, 30)                     # 10 weeks x 2 days x 30 = 600
    seg = _seg(a + b, [1, 2])
    assert seg.loc[1, "regular_weeks"] == 11 and seg.loc[1, "segment"] == "S1"
    assert seg.loc[2, "regular_weeks"] == 10 and seg.loc[2, "segment"] == "S2"


def test_volume_threshold_is_inclusive():
    seg = _seg(_weekly(1, 11, 25) + _rows(2, ["2024-12-01"], 499) + _rows(3, ["2024-12-01"], 500), [1, 2, 3])
    assert seg.loc[2, "segment"] == "S4"
    assert seg.loc[3, "segment"] == "S2"


def test_week_needs_two_send_days():
    one_day_weeks = _weekly(1, 14, 60, days_per_week=1)    # 840 msgs but 1 send day / week -> S2
    low_regular = _weekly(2, 14, 17)                        # < 500 msgs in 3 months, every month -> S3
    seg = _seg(one_day_weeks + low_regular, [1, 2])
    assert seg.loc[1, "regular_weeks"] == 0 and seg.loc[1, "segment"] == "S2"
    assert seg.loc[2, "msgs_window"] < 500
    assert seg.loc[2, "segment"] == "S3"
