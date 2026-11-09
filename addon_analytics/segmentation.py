"""Rule-based usage-habit segmentation (question 1).

Window: the last ``months`` calendar months before ``as_of`` for volume and
monthly regularity, and the last ``weeks`` 7-day blocks before ``as_of`` for
weekly regularity.

* S1 regular high volume: >= 500 messages in the window AND >= 11 of 14 weeks
  with >= 2 send days
* S2 irregular high volume: >= 500 messages, fewer regular weeks
* S3 low but monthly-regular: < 500 messages, >= 1 send day in each of the months
* S4 low and irregular: everything else with >= 1 message
* S0 inactive: adopters with no message in the window
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT_LABELS = {
    "S1": "S1 regular high-volume",
    "S2": "S2 irregular high-volume",
    "S3": "S3 low, monthly-regular",
    "S4": "S4 low, irregular",
    "S0": "S0 inactive in window",
}


def segment_merchants(messages: pd.DataFrame, as_of, merchant_ids=None, months: int = 3, weeks: int = 14,
                      high_volume: int = 500, min_regular_weeks: int = 11,
                      min_days_per_week: int = 2) -> pd.DataFrame:
    """Assign each merchant to S0..S4. ``messages`` needs merchant_id, date, n_attempted."""
    as_of = pd.Timestamp(as_of)
    vol_start = as_of - pd.DateOffset(months=months)
    wk_start = as_of - pd.Timedelta(weeks=weeks)
    daily = messages.groupby(["merchant_id", "date"], as_index=False)["n_attempted"].sum()
    daily = daily[(daily["date"] < as_of) & (daily["n_attempted"] > 0)]

    vol = daily[daily["date"] >= vol_start]
    msgs = vol.groupby("merchant_id")["n_attempted"].sum().rename("msgs_window")
    month_idx = vol["date"].dt.to_period("M")
    active_months = vol.assign(m=month_idx).groupby("merchant_id")["m"].nunique().rename("active_months")

    wk = daily[daily["date"] >= wk_start].copy()
    wk["week"] = (as_of - wk["date"]).dt.days // 7
    per_week = wk.groupby(["merchant_id", "week"]).size()
    regular_weeks = (per_week >= min_days_per_week).groupby("merchant_id").sum().rename("regular_weeks")
    send_days = wk.groupby("merchant_id").size().rename("send_days_window")

    out = pd.concat([msgs, active_months, regular_weeks, send_days], axis=1)
    if merchant_ids is not None:
        out = out.reindex(pd.Index(merchant_ids, name="merchant_id"))
    out = out.fillna(0).astype(int)
    out.index.name = "merchant_id"

    high = out["msgs_window"] >= high_volume
    regular = out["regular_weeks"] >= min_regular_weeks
    monthly = out["active_months"] >= months
    out["segment"] = np.select(
        [out["msgs_window"] == 0, high & regular, high, monthly],
        ["S0", "S1", "S2", "S3"], default="S4")
    return out.reset_index()


def profile_segments(seg: pd.DataFrame, messages: pd.DataFrame, topups: pd.DataFrame, as_of,
                     months: int = 3) -> pd.DataFrame:
    """Usage and top-up profile per segment (S1..S4)."""
    as_of = pd.Timestamp(as_of)
    start = as_of - pd.DateOffset(months=months)
    win = messages[(messages["date"] >= start) & (messages["date"] < as_of)].merge(
        seg[["merchant_id", "segment"]], on="merchant_id")
    win = win[win["segment"] != "S0"]
    tu = topups[(topups["kind"] == "topup") & (topups["date"] >= start) & (topups["date"] < as_of)]
    tu = tu.merge(seg[["merchant_id", "segment"]], on="merchant_id")

    s = seg[seg["segment"] != "S0"]
    prof = s.groupby("segment").agg(
        merchants=("merchant_id", "size"),
        median_msgs_per_month=("msgs_window", lambda x: float(np.median(x)) / months),
        median_send_days=("send_days_window", "median"),
    )
    prof["topup_rate"] = tu.groupby("segment")["merchant_id"].nunique() / prof["merchants"]
    prof["median_topup_amount"] = tu.groupby("segment")["amount"].median()
    prof["topups_per_topping_merchant"] = (tu.groupby("segment").size()
                                           / tu.groupby("segment")["merchant_id"].nunique())
    w = win["n_attempted"]
    prof["mean_send_hour"] = (win["hour"] * w).groupby(win["segment"]).sum() / w.groupby(win["segment"]).sum()
    wkend = win["date"].dt.dayofweek >= 5
    prof["weekend_share"] = (w * wkend).groupby(win["segment"]).sum() / w.groupby(win["segment"]).sum()
    mix = win.pivot_table(index="segment", columns="use_case", values="n_attempted", aggfunc="sum").fillna(0)
    mix = mix.div(mix.sum(1), axis=0).add_prefix("share_")
    prof = prof.join(mix).fillna({"topup_rate": 0})
    return prof.reset_index()


def hour_profile(seg: pd.DataFrame, messages: pd.DataFrame, as_of, months: int = 3) -> pd.DataFrame:
    """Share of messages by hour of day per segment (rows: hour, cols: segment)."""
    as_of = pd.Timestamp(as_of)
    win = messages[(messages["date"] >= as_of - pd.DateOffset(months=months)) & (messages["date"] < as_of)]
    win = win.merge(seg[["merchant_id", "segment"]], on="merchant_id")
    tab = win.pivot_table(index="hour", columns="segment", values="n_attempted", aggfunc="sum").fillna(0)
    return tab / tab.sum()
