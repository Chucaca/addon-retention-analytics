"""Month-over-month retention decomposition and cohort curves (questions 2 and 3).

Two activity definitions are compared:

* ``any``  - at least 1 message in the month
* ``real`` - at least 3 distinct send days in the month ("real use")

The gap between them shows how much of the "active base" is merchants that only
touch the add-on occasionally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFINITIONS = {"any": "≥1 message", "real": "≥3 send days"}


def monthly_activity(messages: pd.DataFrame) -> pd.DataFrame:
    """merchant x month: msgs, success, errors, send_days."""
    m = messages.assign(month=messages["date"].dt.to_period("M"))
    agg = m.groupby(["merchant_id", "month"]).agg(
        msgs=("n_attempted", "sum"), success=("n_success", "sum"),
        config_err=("n_config_err", "sum"), user_err=("n_user_err", "sum"))
    days = m[m["n_attempted"] > 0].groupby(["merchant_id", "month"])["date"].nunique().rename("send_days")
    return agg.join(days).fillna({"send_days": 0}).reset_index()


def active_flags(activity: pd.DataFrame, definition: str = "any") -> pd.DataFrame:
    if definition == "any":
        a = activity[activity["msgs"] >= 1]
    elif definition == "real":
        a = activity[activity["send_days"] >= 3]
    else:
        raise ValueError(definition)
    return a[["merchant_id", "month"]].drop_duplicates()


def decompose(activity: pd.DataFrame, definition: str = "any", months=None) -> pd.DataFrame:
    """Per month: new / retained / reactivated / churned (churned = active last month, not this month)."""
    act = active_flags(activity, definition)
    if months is None:
        months = pd.period_range(activity["month"].min(), activity["month"].max(), freq="M")
    sets = {m: set(act.loc[act["month"] == m, "merchant_id"]) for m in months}
    seen: set = set()
    prev: set = set()
    rows = []
    for m in months:
        cur = sets[m]
        new = cur - seen
        retained = cur & prev
        reactivated = (cur - prev) & seen
        churned = prev - cur
        rows.append({"month": m, "active": len(cur), "new": len(new), "retained": len(retained),
                     "reactivated": len(reactivated), "churned": len(churned),
                     "churn_rate": len(churned) / len(prev) if prev else np.nan})
        seen |= cur
        prev = cur
    return pd.DataFrame(rows)


def cohort_retention(activity: pd.DataFrame, definition: str = "any", max_age: int = 12,
                     min_cohort: int = 1) -> pd.DataFrame:
    """Share of each first-active-month cohort that is active k months later (NaN = not observable)."""
    act = active_flags(activity, definition)
    first = act.groupby("merchant_id")["month"].min().rename("cohort")
    act = act.join(first, on="merchant_id")
    act["age"] = (act["month"] - act["cohort"]).apply(lambda x: x.n)
    size = first.value_counts().sort_index()
    tab = act[act["age"] <= max_age].pivot_table(index="cohort", columns="age", values="merchant_id",
                                                 aggfunc="nunique").reindex(size.index).fillna(0)
    tab = tab.div(size, axis=0)
    last = activity["month"].max()
    for c in tab.index:
        tab.loc[c, [a for a in tab.columns if (c + a) > last]] = np.nan
    tab.insert(0, "cohort_size", size)
    return tab[tab["cohort_size"] >= min_cohort]


def use_case_retention(messages: pd.DataFrame, activity: pd.DataFrame, horizons=(3, 6),
                       first_days: int = 60, definition: str = "real") -> pd.DataFrame:
    """Retention by the merchant's dominant use case in its first ``first_days`` days.

    Retention at k = share of the merchant's adoption cohort still active (``definition``)
    exactly k months after the adoption month; only merchants with month k observable count.
    Merchants with <= 10 messages in the first window are reported as 'trial (<=10 msgs)'.
    """
    first_date = messages.groupby("merchant_id")["date"].min().rename("first_date")
    m = messages.join(first_date, on="merchant_id")
    early = m[m["date"] < m["first_date"] + pd.Timedelta(days=first_days)]
    mix = early.pivot_table(index="merchant_id", columns="use_case", values="n_attempted", aggfunc="sum").fillna(0)
    tot = mix.sum(1)
    shares = mix.div(tot, axis=0)
    dominant = shares.drop(columns=["test"], errors="ignore").idxmax(1)
    n_cases = (shares.drop(columns=["test"], errors="ignore") >= 0.15).sum(1)
    label = dominant.where(tot > 10, "trial (<=10 msgs)")
    # multi-use-case flag only for non-trial merchants (trial merchants would bias the comparison)
    multi = pd.Series(np.where(n_cases >= 2, "True", "False"), index=n_cases.index).where(tot > 10, "trial")
    df = pd.DataFrame({"use_case": label, "multi_use_case": multi})
    df["adopt_month"] = first_date.dt.to_period("M").reindex(df.index)

    act = active_flags(activity, definition)
    act_set = set(zip(act["merchant_id"], act["month"]))
    last = activity["month"].max()
    rows = []
    for col in ("use_case", "multi_use_case"):
        for g, sub in df.groupby(col):
            if col == "multi_use_case" and g == "trial":
                continue
            row = {"group": f"{col}={g}" if col != "use_case" else str(g), "merchants": len(sub)}
            for k in horizons:
                ok = sub[sub["adopt_month"] + k <= last]
                row[f"n_obs_m{k}"] = len(ok)
                row[f"retained_m{k}"] = (np.mean([(i, a + k) in act_set for i, a in zip(ok.index, ok["adopt_month"])])
                                         if len(ok) else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)
