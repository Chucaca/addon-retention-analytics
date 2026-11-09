"""Interpretable lookalike targeting rules for the rollout (question 4, step 2).

Thresholds are read off the lift table (descriptive over-representation of good
adopters vs the non-adopter base), so every rule can be explained in one line:

* A - chains: >= 2 branches
* B - organized single stores: 1 branch, >= U users, e-invoice in use, >= I1 invoices/30d
* C - service businesses with staff: service industry with significant lift,
      >= E employees and >= I2 invoices/30d

Rules are evaluated by recall of current good adopters (S1/S2/S3) versus the share
of the non-adopter base they select. Recall / selection is a *concentration*
ratio, not a forecast of conversion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lift import BANDS, band_lower_bound
from .synthetic import SERVICE_INDUSTRIES

DEFAULT_RULES = {"min_users_B": 3, "min_invoices_B": 100, "min_employees_C": 3, "min_invoices_C": 30,
                 "service_industries": sorted(SERVICE_INDUSTRIES)}
RULE_NAMES = {"A": "A chains", "B": "B organized single stores", "C": "C service businesses with staff"}


def _min_threshold(lift: pd.DataFrame, feature: str, default, min_lift: float = 1.0):
    """Lowest band from which every higher band is over-represented (lift >= min_lift)
    and the band itself is significant (|z| >= 2, n >= 5)."""
    labels = BANDS[feature][2]
    t = lift[lift["feature"] == feature].set_index("value").reindex(labels)
    best = None
    for i in range(len(labels) - 1, -1, -1):
        row = t.iloc[i]
        if pd.isna(row["lift"]) or row["lift"] < min_lift:
            break
        if bool(row["keep"]) and row["z"] > 0:
            best = labels[i]
    return band_lower_bound(feature, best) if best is not None else default


def derive_rules(lift: pd.DataFrame, lift_single: pd.DataFrame | None = None) -> dict:
    """Derive rule thresholds from lift tables (``lift_single`` = lift among single stores)."""
    ls = lift_single if lift_single is not None else lift
    rules = dict(DEFAULT_RULES)
    rules["min_users_B"] = int(_min_threshold(ls, "users", DEFAULT_RULES["min_users_B"]))
    rules["min_invoices_B"] = int(_min_threshold(ls, "invoices_30d", DEFAULT_RULES["min_invoices_B"], min_lift=1.5))
    rules["min_employees_C"] = int(max(1, _min_threshold(ls, "employees", DEFAULT_RULES["min_employees_C"])))
    sig = pd.concat([lift, ls])
    ind = sig[(sig["feature"] == "industry") & sig["keep"] & (sig["z"] > 0)]
    svc = sorted(set(ind["value"]) & SERVICE_INDUSTRIES)
    if svc:
        rules["service_industries"] = svc
    return rules


def apply_rules(merchants: pd.DataFrame, rules: dict) -> pd.Series:
    """First matching rule (A > B > C) per merchant, '' if none."""
    m = merchants
    single = m["n_branches"] == 1
    a = m["n_branches"] >= 2
    b = single & (m["n_users"] >= rules["min_users_B"]) & m["uses_einvoice"] & (m["invoices_30d"] >= rules["min_invoices_B"])
    c = (single & m["industry"].isin(rules["service_industries"]) & (m["n_employees"] >= rules["min_employees_C"])
         & (m["invoices_30d"] >= rules["min_invoices_C"]))
    return pd.Series(np.select([a, b, c], ["A", "B", "C"], default=""), index=m.index, name="rule")


def evaluate_coverage(merchants: pd.DataFrame, segments: pd.DataFrame, rules: dict,
                      seed_segments=("S1", "S2", "S3")) -> pd.DataFrame:
    """Recall of seed adopters by segment vs share of the non-adopter base selected, cumulative A, A+B, A+B+C."""
    m = merchants.assign(rule=apply_rules(merchants, rules))
    seg = segments.set_index("merchant_id")["segment"]
    m["segment"] = m["merchant_id"].map(seg)
    base = m[~m["is_adopter"]]
    seeds = m[m["segment"].isin(seed_segments)]
    rows = []
    for k, label in [("A", "A"), ("AB", "A+B"), ("ABC", "A+B+C")]:
        sel_b = base["rule"].isin(list(k)) & (base["rule"] != "")
        sel_s = seeds["rule"].isin(list(k)) & (seeds["rule"] != "")
        row = {"rules": label, "base_selected": int(sel_b.sum()), "base_share": float(sel_b.mean()),
               "seed_recall": float(sel_s.mean())}
        for s in seed_segments:
            ss = seeds["segment"] == s
            row[f"recall_{s}"] = float(sel_s[ss].mean()) if ss.any() else np.nan
        row["concentration"] = row["seed_recall"] / row["base_share"] if row["base_share"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def suggested_use_cases(merchants: pd.DataFrame, segments: pd.DataFrame, messages: pd.DataFrame,
                        rules: dict, seed_segments=("S1", "S2", "S3")) -> dict:
    """Suggested use case per rule: the (non-test) use case whose share of message volume among the
    seed adopters matched by the rule is most over-indexed vs all seed adopters."""
    m = merchants.assign(rule=apply_rules(merchants, rules))
    seeds = segments.loc[segments["segment"].isin(seed_segments), "merchant_id"]
    r = m.set_index("merchant_id").loc[seeds, "rule"]
    msg = messages[messages["use_case"] != "test"].merge(r.rename("rule").reset_index(), on="merchant_id")
    mix = msg.groupby(["rule", "use_case"])["n_attempted"].sum().unstack(fill_value=0)
    share = mix.div(mix.sum(1), axis=0)
    overall = mix.sum() / mix.values.sum()
    return {k: (share.loc[k] / overall).idxmax() for k in "ABC" if k in share.index}


def target_list(merchants: pd.DataFrame, features: pd.DataFrame, lift: pd.DataFrame, rules: dict,
                use_cases: dict, top_n: int | None = None) -> pd.DataFrame:
    """Prioritised non-adopter targets: rule priority (A > B > C), then a lift score
    (sum of log-lift over the merchant's significant feature values)."""
    m = merchants.assign(rule=apply_rules(merchants, rules))
    m = m[(~m["is_adopter"]) & (m["rule"] != "")]
    f = features.set_index("merchant_id").loc[m["merchant_id"]]
    sig = lift[lift["keep"]].set_index(["feature", "value"])["lift"]
    score = np.zeros(len(f))
    for feat in f.columns:
        lk = sig.xs(feat, level="feature") if feat in sig.index.get_level_values(0) else pd.Series(dtype=float)
        score += np.log(f[feat].map(lk).fillna(1.0).clip(lower=1e-3)).to_numpy()
    out = pd.DataFrame({
        "merchant_id": m["merchant_id"].to_numpy(), "rule": m["rule"].to_numpy(),
        "rule_name": m["rule"].map(RULE_NAMES).to_numpy(), "industry": m["industry"].to_numpy(),
        "n_branches": m["n_branches"].to_numpy(), "invoices_30d": m["invoices_30d"].to_numpy(),
        "lift_score": np.round(score, 3), "suggested_use_case": m["rule"].map(use_cases).to_numpy(),
    })
    out = out.sort_values(["rule", "lift_score"], ascending=[True, False]).reset_index(drop=True)
    out.insert(0, "priority_rank", np.arange(1, len(out) + 1))
    return out.head(top_n) if top_n else out
