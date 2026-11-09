import pandas as pd

from addon_analytics.lookalike import apply_rules
from addon_analytics.retention import decompose


def P(s):
    return pd.Period(s, "M")


def test_decompose_counts():
    act = pd.DataFrame([
        (1, P("2024-01"), 5, 5), (1, P("2024-02"), 5, 5), (1, P("2024-04"), 5, 5),
        (2, P("2024-02"), 1, 1), (2, P("2024-03"), 9, 4),
    ], columns=["merchant_id", "month", "msgs", "send_days"])
    d = decompose(act, "any").set_index("month")
    assert d.loc[P("2024-01"), ["new", "retained", "churned"]].tolist() == [1, 0, 0]
    assert d.loc[P("2024-02"), ["new", "retained", "churned"]].tolist() == [1, 1, 0]
    assert d.loc[P("2024-03"), ["retained", "churned"]].tolist() == [1, 1]
    assert d.loc[P("2024-04"), ["reactivated", "churned"]].tolist() == [1, 1]
    real = decompose(act, "real").set_index("month")
    # merchant 2 is only "really" active in March (>= 3 send days)
    assert real.loc[P("2024-03"), "new"] == 1 and real.loc[P("2024-02"), "active"] == 1


def test_rules_priority():
    m = pd.DataFrame({
        "n_branches": [3, 1, 1, 1], "n_users": [1, 4, 1, 1], "uses_einvoice": [False, True, False, False],
        "invoices_30d": [10, 150, 50, 50], "industry": ["beauty_spa", "beauty_spa", "beauty_spa", "grocery_minimart"],
        "n_employees": [5, 5, 5, 5],
    })
    rules = {"min_users_B": 3, "min_invoices_B": 100, "min_employees_C": 3, "min_invoices_C": 30,
             "service_industries": ["beauty_spa"]}
    assert apply_rules(m, rules).tolist() == ["A", "B", "C", ""]
