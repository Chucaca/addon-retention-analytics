import numpy as np
import pandas as pd

from addon_analytics.churn import REASON_ORDER, attribute_reasons, label_churn


def _monthly(series: dict, start="2024-01"):
    rows = []
    for mid, msgs in series.items():
        for k, n in enumerate(msgs):
            rows.append({"merchant_id": mid, "month": pd.Period(start, "M") + k, "msgs": n})
    return pd.DataFrame(rows)


def test_churn_definition_edge_cases():
    m = _monthly({
        1: [5, 0, 0, 7, 0, 0],   # Jan churns (the later comeback does not matter); Apr churns
        2: [5, 0, 3, 0, 0, 0],   # a one-month gap is not churn; Mar churns
        3: [5, 5, 5, 5, 5, 5],   # never churns; May/Jun are not observable
    })
    lab = label_churn(m, "2024-06")
    got = {(r.merchant_id, str(r.month)): v for r, v in zip(m.itertuples(), lab)}
    assert got[(1, "2024-01")] == 1
    assert np.isnan(got[(1, "2024-02")])          # inactive months are not labelled
    assert got[(1, "2024-04")] == 1               # M+2 == last month is observable
    assert got[(2, "2024-01")] == 0
    assert got[(2, "2024-03")] == 1
    assert got[(3, "2024-04")] == 0
    assert np.isnan(got[(3, "2024-05")]) and np.isnan(got[(3, "2024-06")])


def test_churn_label_does_not_look_into_other_merchants():
    # merchant 1 is active only in its last row; the next rows belong to merchant 2 (always active)
    m = _monthly({1: [0, 0, 0, 5, 0, 0], 2: [5, 5, 5, 5, 5, 5]})
    m = m[~((m.merchant_id == 1) & (m.month > pd.Period("2024-04", "M")))].reset_index(drop=True)
    lab = label_churn(m, "2024-06")
    assert lab.iloc[3] == 1


NEUTRAL = dict(pos_invoices=100, pos_invoices_next=100, pos_prior3=100, core_sub_active_next=True,
               lifetime_msgs=500, est_balance=300, runway_months=3.0, config_err_rate=0.0,
               config_err_rate_prev=0.0, trend_ratio=1.0)
FLAGS = {
    "store_stopped_selling": dict(pos_invoices_next=0),
    "core_subscription_expired": dict(core_sub_active_next=False),
    "trial_only": dict(lifetime_msgs=8),
    "balance_depleted": dict(est_balance=0, runway_months=0.0),
    "config_errors": dict(config_err_rate=0.4),
    "usage_decay": dict(trend_ratio=0.3),
}


def _row(*reasons):
    r = dict(NEUTRAL)
    for k in reasons:
        r.update(FLAGS[k])
    return r


def test_each_reason_alone():
    rows = pd.DataFrame([_row(k) for k in FLAGS] + [_row()])
    assert list(attribute_reasons(rows)) == list(FLAGS) + ["unexplained"]


def test_attribution_priority_order():
    cases = [
        (("balance_depleted", "config_errors"), "balance_depleted"),
        (("store_stopped_selling", "balance_depleted"), "store_stopped_selling"),
        (("core_subscription_expired", "balance_depleted", "usage_decay"), "core_subscription_expired"),
        (("trial_only", "balance_depleted"), "trial_only"),
        (("config_errors", "usage_decay"), "config_errors"),
        (tuple(FLAGS), "store_stopped_selling"),
    ]
    rows = pd.DataFrame([_row(*c) for c, _ in cases])
    assert list(attribute_reasons(rows)) == [want for _, want in cases]


def test_pairwise_priority_is_consistent_with_reason_order():
    names = list(FLAGS)
    assert REASON_ORDER[:-1] == names
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            got = attribute_reasons(pd.DataFrame([_row(a, b)])).iloc[0]
            assert got == a, (a, b, got)


def test_low_runway_alone_counts_as_balance_and_previous_month_config_errors_count():
    r1 = _row()
    r1["runway_months"] = 0.1          # balance still > 5 credits but < 1 week of typical spend
    r2 = _row()
    r2["config_err_rate_prev"] = 0.5   # blocked last month, quiet this month
    assert list(attribute_reasons(pd.DataFrame([r1, r2]))) == ["balance_depleted", "config_errors"]
