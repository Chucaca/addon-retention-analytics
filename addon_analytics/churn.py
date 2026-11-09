"""Churn panel, rule-based reason attribution and an ML cross-check (question 3).

Churn definition: a merchant churns at month M if it is active in M (>= 1 message)
and sends nothing in M+1 and M+2. Months M whose M+2 lies beyond the data are
not labelled (NaN). A merchant that comes back later still counts as churned at M
(the reactivation shows up in the retention decomposition).

Reason attribution is a PRIORITY-ORDERED heuristic: the first rule that fires wins.
Exogenous stops (store closed, core subscription lapsed) are checked first so that
a merchant whose store closed is not blamed on its balance; then onboarding
failures, then the product-addressable causes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# order matters: first match wins
REASONS = [
    # name, addressable_by_addon_team, lever
    ("store_stopped_selling", "no", "none (exogenous)"),
    ("core_subscription_expired", "no", "core-product renewal team"),
    ("trial_only", "partly", "onboarding: first-campaign wizard"),
    ("balance_depleted", "yes", "low-balance top-up nudges / auto top-up"),
    ("config_errors", "yes", "config-error alerting + guided template fix"),
    ("usage_decay", "partly", "re-engagement / use-case expansion"),
    ("unexplained", "unknown", "survey / qualitative follow-up"),
]
REASON_ORDER = [r[0] for r in REASONS]
ADDRESSABLE = {r[0]: r[1] for r in REASONS}
LEVER = {r[0]: r[2] for r in REASONS}

DEFAULT_THRESHOLDS = {
    "trial_max_lifetime_msgs": 10,
    "low_balance_credits": 5,
    "low_runway_months": 0.25,
    "config_err_rate": 0.10,
    "decay_ratio": 0.5,
    "pos_drop_ratio": 0.35,
}

ML_FEATURES = [
    "log_balance", "runway_months", "trend_ratio", "config_err_rate", "config_err_rate_prev",
    "user_err_rate", "days_since_topup", "log_lifetime_msgs", "log_msgs", "send_days",
    "pos_trend", "n_paid_topups", "is_chain", "tenure_months",
]


def label_churn(monthly: pd.DataFrame, last_month) -> pd.Series:
    """Churn label for a dense merchant x month frame (msgs column, sorted, no gaps).

    Returns 1/0 for active months whose M+2 is observable, NaN otherwise
    (inactive months and the last two months).
    """
    last_month = pd.Period(last_month, freq="M")
    g = monthly.groupby("merchant_id")["msgs"]
    n1 = g.shift(-1).fillna(0)
    n2 = g.shift(-2).fillna(0)
    observable = monthly["month"] + 2 <= last_month
    active = monthly["msgs"] > 0
    lab = ((n1 == 0) & (n2 == 0)).astype(float)
    return lab.where(active & observable)


def build_panel(activity: pd.DataFrame, topups: pd.DataFrame, status: pd.DataFrame,
                merchants: pd.DataFrame, last_month=None) -> pd.DataFrame:
    """Dense merchant x month panel with engineered features; returns active months only."""
    last_month = pd.Period(last_month or activity["month"].max(), freq="M")
    first = activity[activity["msgs"] > 0].groupby("merchant_id")["month"].min()
    months = pd.period_range(first.min(), last_month, freq="M")
    grid = pd.MultiIndex.from_product([first.index, months], names=["merchant_id", "month"]).to_frame(index=False)
    grid = grid[grid["month"] >= grid["merchant_id"].map(first)]
    p = grid.merge(activity, on=["merchant_id", "month"], how="left")
    for c in ["msgs", "success", "config_err", "user_err", "send_days"]:
        p[c] = p[c].fillna(0)

    t = topups.assign(month=topups["date"].dt.to_period("M"))
    credit = t.groupby(["merchant_id", "month"]).agg(credit=("amount", "sum"), last_topup=("date", "max"))
    paid = t[t["kind"] == "topup"].groupby(["merchant_id", "month"]).size().rename("paid_topups")
    p = p.merge(credit, on=["merchant_id", "month"], how="left").merge(paid, on=["merchant_id", "month"], how="left")
    p["credit"] = p["credit"].fillna(0)
    p["paid_topups"] = p["paid_topups"].fillna(0)
    p = p.sort_values(["merchant_id", "month"]).reset_index(drop=True)
    # credits granted before the first active month (rare) are folded into the first month
    pre = t[t["month"] < t["merchant_id"].map(first)].groupby("merchant_id")["amount"].sum()
    is_first = p["month"] == p["merchant_id"].map(first)
    p.loc[is_first, "credit"] += p.loc[is_first, "merchant_id"].map(pre).fillna(0)

    g = p.groupby("merchant_id")
    p["est_balance"] = g["credit"].cumsum() - g["success"].cumsum()
    spend3 = g["success"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    p["runway_months"] = (p["est_balance"].clip(lower=0) / spend3.clip(lower=1)).clip(upper=24)
    prior3 = g["msgs"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    p["trend_ratio"] = (p["msgs"] / prior3).where(prior3 > 0, 1.0).clip(upper=5)
    p["config_err_rate"] = (p["config_err"] / p["msgs"]).where(p["msgs"] > 0, 0.0)
    p["config_err_rate_prev"] = g["config_err_rate"].shift(1).fillna(0)
    p["user_err_rate"] = (p["user_err"] / p["msgs"]).where(p["msgs"] > 0, 0.0)
    p["lifetime_msgs"] = g["msgs"].cumsum()
    p["n_paid_topups"] = g["paid_topups"].cumsum()
    last_tu = g["last_topup"].ffill()
    month_end = p["month"].dt.end_time.dt.normalize()
    p["days_since_topup"] = (month_end - last_tu).dt.days.clip(lower=0)

    st = status[["merchant_id", "month", "pos_invoices", "core_sub_active"]]
    p = p.merge(st, on=["merchant_id", "month"], how="left")
    g = p.groupby("merchant_id")
    p["pos_invoices_next"] = g["pos_invoices"].shift(-1)
    p["core_sub_active_next"] = g["core_sub_active"].shift(-1)
    pos_prior = g["pos_invoices"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    p["pos_prior3"] = pos_prior
    p["pos_trend"] = (p["pos_invoices"] / pos_prior).where(pos_prior > 0, 1.0).clip(upper=3)

    p["churn"] = label_churn(p, last_month)
    prof = merchants.set_index("merchant_id")
    p["is_chain"] = p["merchant_id"].map(prof["n_branches"] >= 2).astype(int)
    p["tenure_months"] = p["merchant_id"].map(prof["tenure_months"])
    p["log_balance"] = np.log1p(p["est_balance"].clip(lower=0))
    p["log_lifetime_msgs"] = np.log1p(p["lifetime_msgs"])
    p["log_msgs"] = np.log1p(p["msgs"])
    return p[p["msgs"] > 0].reset_index(drop=True)


def attribute_reasons(rows: pd.DataFrame, thresholds: dict | None = None) -> pd.Series:
    """Priority-ordered churn-reason attribution for churned panel rows."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    pos_next = rows["pos_invoices_next"].fillna(rows["pos_invoices"])
    pos_ref = rows["pos_prior3"].fillna(rows["pos_invoices"]).clip(lower=1)
    sub_next = rows["core_sub_active_next"].astype("boolean").fillna(True).astype(bool)
    conds = [
        pos_next < th["pos_drop_ratio"] * pos_ref,
        ~sub_next,
        rows["lifetime_msgs"] <= th["trial_max_lifetime_msgs"],
        (rows["est_balance"] < th["low_balance_credits"]) | (rows["runway_months"] < th["low_runway_months"]),
        np.maximum(rows["config_err_rate"], rows["config_err_rate_prev"]) >= th["config_err_rate"],
        rows["trend_ratio"] < th["decay_ratio"],
    ]
    return pd.Series(np.select(conds, REASON_ORDER[:-1], default="unexplained"), index=rows.index, name="reason")


def reason_summary(churned: pd.DataFrame) -> pd.DataFrame:
    s = churned["reason"].value_counts().reindex(REASON_ORDER).fillna(0).astype(int).rename("churn_events")
    out = s.to_frame()
    out["share"] = out["churn_events"] / out["churn_events"].sum()
    out["addressable"] = out.index.map(ADDRESSABLE)
    out["lever"] = out.index.map(LEVER)
    out["avg_monthly_spend_before"] = churned.groupby("reason")["success"].mean().reindex(out.index)
    out.index.name = "reason"
    return out.reset_index()


def match_ground_truth(churned: pd.DataFrame, stop_events: pd.DataFrame) -> pd.DataFrame:
    """Attach the simulator's true stop reason (first stop event within M .. M+2)."""
    ev = stop_events.sort_values("date")
    out = []
    for r in churned[["merchant_id", "month", "reason"]].itertuples(index=False):
        lo, hi = r.month.start_time, (r.month + 2).end_time
        e = ev[(ev["merchant_id"] == r.merchant_id) & (ev["date"] >= lo) & (ev["date"] <= hi)]
        out.append(e["true_reason"].iloc[0] if len(e) else "no_stop_event")
    return churned.assign(true_reason=out)


TRUE_TO_RULE = {"store_closed": "store_stopped_selling", "subscription_expired": "core_subscription_expired"}


def ml_crosscheck(panel: pd.DataFrame, test_months: int = 6, seed: int = 0) -> dict:
    """Time-split churn model (logistic + gradient boosting) with permutation importance."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = panel.dropna(subset=["churn"]).copy()
    d[ML_FEATURES] = d[ML_FEATURES].astype(float).fillna(0)
    months = sorted(d["month"].unique())
    cut = months[-test_months]
    train, test = d[d["month"] < cut], d[d["month"] >= cut]
    Xtr, ytr, Xte, yte = train[ML_FEATURES], train["churn"], test[ML_FEATURES], test["churn"]

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(Xtr, ytr)
    gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250,
                                         random_state=seed).fit(Xtr, ytr)
    auc_logit = roc_auc_score(yte, logit.predict_proba(Xte)[:, 1])
    auc_gbm = roc_auc_score(yte, gbm.predict_proba(Xte)[:, 1])
    pi = permutation_importance(gbm, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=seed)
    imp = pd.DataFrame({"feature": ML_FEATURES, "importance": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("importance", ascending=False)
    coefs = pd.Series(logit[-1].coef_[0], index=ML_FEATURES).sort_values(key=abs, ascending=False)
    return {
        "train_rows": len(train), "test_rows": len(test), "test_from": str(cut),
        "train_churn_rate": float(ytr.mean()), "test_churn_rate": float(yte.mean()),
        "auc_logit": float(auc_logit), "auc_gbm": float(auc_gbm),
        "importance": imp.reset_index(drop=True), "logit_coefs": coefs,
    }


FEATURE_TO_RULE = {
    "log_balance": "balance_depleted", "runway_months": "balance_depleted", "days_since_topup": "balance_depleted",
    "n_paid_topups": "balance_depleted", "config_err_rate": "config_errors", "config_err_rate_prev": "config_errors",
    "trend_ratio": "usage_decay", "log_msgs": "usage_decay", "send_days": "usage_decay",
    "log_lifetime_msgs": "trial_only", "pos_trend": "store_stopped_selling",
    "user_err_rate": "(none)", "is_chain": "(profile)", "tenure_months": "(profile)",
}
