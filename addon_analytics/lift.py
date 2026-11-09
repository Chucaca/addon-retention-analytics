"""Profile lift of a seed segment versus the non-adopter base (question 4, step 1).

For every categorical feature value v:

    lift(v) = share of seed merchants with v / share of base merchants with v

tested with a pooled two-proportion z-test. Values are kept when |z| >= 2 and the
seed count is >= 5.

IMPORTANT: this is DESCRIPTIVE over-representation. A lift of 3x for "chains"
means chains are three times as common among good adopters as in the base. It
does not mean that being a chain causes adoption, nor that targeting chains will
triple conversion (no uplift / treatment-effect estimate is made here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (feature, column, bin edges (right-open), labels)
BANDS = {
    "branches": ("n_branches", [1, 2, 4, np.inf], ["1", "2-3", "4+"]),
    "users": ("n_users", [1, 2, 3, 6, np.inf], ["1", "2", "3-5", "6+"]),
    "employees": ("n_employees", [0, 1, 3, 6, 11, np.inf], ["0", "1-2", "3-5", "6-10", "11+"]),
    "customers": ("n_customers", [0, 100, 500, 2000, 10000, np.inf], ["<100", "100-499", "500-1999", "2000-9999", "10000+"]),
    "products": ("n_products", [0, 50, 200, 1000, np.inf], ["<50", "50-199", "200-999", "1000+"]),
    "invoices_30d": ("invoices_30d", [0, 30, 100, 300, 1000, np.inf], ["<30", "30-99", "100-299", "300-999", "1000+"]),
    "tenure": ("tenure_months", [0, 6, 12, 24, 48, np.inf], ["<6m", "6-11m", "1-2y", "2-4y", "4y+"]),
}
CATEGORICAL = ["industry", "province_tier", "uses_einvoice"]
FEATURE_ORDER = list(BANDS) + CATEGORICAL


def band_lower_bound(feature: str, label: str) -> float:
    """Numeric lower bound of a band label, e.g. ('invoices_30d', '100-299') -> 100."""
    _, edges, labels = BANDS[feature]
    return edges[labels.index(label)]


def add_bands(merchants: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"merchant_id": merchants["merchant_id"].to_numpy()})
    for feat, (col, edges, labels) in BANDS.items():
        out[feat] = pd.cut(merchants[col].to_numpy(), bins=edges, labels=labels, right=False).astype(str)
    out["industry"] = merchants["industry"].to_numpy()
    out["province_tier"] = merchants["province_tier"].to_numpy()
    out["uses_einvoice"] = np.where(merchants["uses_einvoice"].to_numpy(), "yes", "no")
    return out


def two_proportion_z(x1: float, n1: float, x0: float, n0: float) -> float:
    """Pooled two-proportion z statistic for p1 = x1/n1 vs p0 = x0/n0."""
    p1, p0 = x1 / n1, x0 / n0
    p = (x1 + x0) / (n1 + n0)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
    return float((p1 - p0) / se) if se > 0 else 0.0


def profile_lift(features: pd.DataFrame, seed_mask, base_mask, min_n: int = 5, z_threshold: float = 2.0,
                 feature_cols=None) -> pd.DataFrame:
    """Lift table for every feature value. ``keep`` marks |z| >= z_threshold and n_seed >= min_n."""
    feature_cols = feature_cols or [c for c in FEATURE_ORDER if c in features.columns]
    seed = features[np.asarray(seed_mask)]
    base = features[np.asarray(base_mask)]
    n1, n0 = len(seed), len(base)
    rows = []
    for f in feature_cols:
        c1 = seed[f].value_counts()
        c0 = base[f].value_counts()
        for v in c0.index.union(c1.index):
            x1, x0 = int(c1.get(v, 0)), int(c0.get(v, 0))
            s1, s0 = x1 / n1, x0 / n0
            rows.append({
                "feature": f, "value": v, "n_seed": x1, "share_seed": s1, "n_base": x0, "share_base": s0,
                "lift": s1 / s0 if s0 > 0 else np.inf, "z": two_proportion_z(x1, n1, x0, n0),
            })
    out = pd.DataFrame(rows)
    out["keep"] = (out["z"].abs() >= z_threshold) & (out["n_seed"] >= min_n)
    return out.sort_values(["feature", "lift"], ascending=[True, False]).reset_index(drop=True)
