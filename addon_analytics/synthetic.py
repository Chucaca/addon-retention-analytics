"""Seeded synthetic data generator.

Setting: a prepaid customer-messaging add-on sold inside a POS SaaS. Merchants
top up credit and send transactional / marketing notifications to their own
customers. Everything here is synthetic; the mechanisms are hand-designed so the
analysis has something realistic to find:

* the merchant base (~180k) has a POS profile; ~600 merchants adopt the add-on,
  with adoption propensity higher for chains, e-invoice users, busier stores and
  service businesses;
* adopters follow one of five latent usage archetypes (regular high-volume,
  campaign-driven, low but regular, low and irregular, trial-only);
* prepaid balance: every successful message costs 1 credit. When the balance hits
  zero a merchant either tops up (probability = its "diligence") or silently stops;
* template configuration errors (failed, uncharged messages) push merchants to quit;
* some merchants' usage decays, some stores close, some core POS subscriptions
  lapse, and a small hazard of unexplained stopping remains;
* some churned merchants come back (reactivation).

The true stop reason of every stop event is kept in ``stop_events`` so the
heuristic churn attribution can be validated against ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

START = pd.Timestamp("2023-01-01")
N_DAYS = 731  # 2023-01-01 .. 2024-12-31
USE_CASES = ["transactional", "customer_care", "marketing", "test"]
STARTER_CREDIT = 50
TOPUP_TIERS = np.array([100, 200, 300, 500, 1_000, 2_000, 5_000, 10_000])

INDUSTRY_WEIGHTS = {
    "fashion_apparel": 0.16,
    "grocery_minimart": 0.18,
    "food_beverage": 0.15,
    "pharmacy": 0.05,
    "beauty_spa": 0.06,
    "clinic_dental": 0.02,
    "mother_baby": 0.05,
    "electronics_phone": 0.06,
    "building_materials": 0.05,
    "books_stationery": 0.03,
    "pet_services": 0.02,
    "education_center": 0.02,
    "auto_moto_services": 0.03,
    "other_retail": 0.12,
}
SERVICE_INDUSTRIES = {
    "beauty_spa", "clinic_dental", "pet_services", "education_center", "auto_moto_services",
}
MARKETING_INDUSTRIES = {"fashion_apparel", "mother_baby", "electronics_phone"}
INDUSTRY_ADOPTION_EFFECT = {
    "fashion_apparel": 0.3, "grocery_minimart": -0.5, "food_beverage": -0.3,
    "pharmacy": 0.5, "beauty_spa": 1.0, "clinic_dental": 1.1, "mother_baby": 0.6,
    "electronics_phone": 0.2, "building_materials": -0.6, "books_stationery": -0.2,
    "pet_services": 0.8, "education_center": 0.8, "auto_moto_services": 0.6,
    "other_retail": -0.3,
}
CHAIN_PROB = {"pharmacy": 0.18, "food_beverage": 0.14, "fashion_apparel": 0.12, "mother_baby": 0.12}

ARCHETYPES = ["regular_high", "campaign_high", "low_regular", "low_irregular", "trial_only"]
# use-case mix (transactional, customer_care, marketing, test)
ARCHETYPE_MIX = {
    "regular_high": [0.66, 0.20, 0.11, 0.03],
    "campaign_high": [0.22, 0.12, 0.63, 0.03],
    "low_regular": [0.22, 0.65, 0.08, 0.05],
    "low_irregular": [0.35, 0.25, 0.32, 0.08],
    "trial_only": [0.20, 0.05, 0.05, 0.70],
}


@dataclass
class SyntheticData:
    merchants: pd.DataFrame    # whole POS base, one row per merchant, is_adopter flag
    messages: pd.DataFrame     # merchant x day x use case send log
    topups: pd.DataFrame       # credit events (starter credit + paid top-ups)
    status: pd.DataFrame       # adopter x month POS status (invoices, core subscription)
    stop_events: pd.DataFrame  # ground-truth stop reasons (validation only)
    start: pd.Timestamp = START
    end: pd.Timestamp = START + pd.Timedelta(days=N_DAYS - 1)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_merchants(n_merchants: int, n_adopters: int, rng: np.random.Generator) -> pd.DataFrame:
    names = np.array(list(INDUSTRY_WEIGHTS))
    w = np.array(list(INDUSTRY_WEIGHTS.values()))
    industry = rng.choice(names, size=n_merchants, p=w / w.sum())
    service = np.isin(industry, list(SERVICE_INDUSTRIES))

    tier = rng.choice(["tier1_metro", "tier2_city", "tier3_rural"], size=n_merchants, p=[0.45, 0.30, 0.25])
    chain_p = np.array([CHAIN_PROB.get(i, 0.08) for i in industry]) + 0.03 * (tier == "tier1_metro")
    is_chain = rng.random(n_merchants) < chain_p
    n_branches = np.where(is_chain, 1 + rng.geometric(0.45, n_merchants), 1).clip(1, 40)
    lb = np.log(n_branches)

    size = rng.normal(0, 1, n_merchants) + 0.25 * (tier == "tier1_metro") - 0.25 * (tier == "tier3_rural")
    n_users = 1 + rng.poisson(np.exp(-0.6 + 0.6 * size + 0.8 * lb))
    n_employees = rng.poisson(np.exp(0.0 + 0.7 * size + 0.9 * lb + 0.6 * service))
    n_customers = np.round(np.exp(5.2 + 1.0 * size + 0.5 * lb + rng.normal(0, 0.7, n_merchants))).astype(int)
    n_products = np.round(np.exp(4.6 + 0.8 * size - 1.2 * service + rng.normal(0, 0.9, n_merchants))).astype(int)
    invoices_30d = np.round(np.exp(3.9 + 1.0 * size + 0.8 * lb + rng.normal(0, 0.7, n_merchants))).astype(int)
    dormant = rng.random(n_merchants) < 0.08
    invoices_30d = np.where(dormant, rng.integers(0, 5, n_merchants), invoices_30d)
    tenure_months = (1 + rng.exponential(22, n_merchants)).astype(int).clip(1, 120)
    uses_einvoice = rng.random(n_merchants) < _sigmoid(-1.4 + 0.6 * size + 0.7 * is_chain)

    df = pd.DataFrame({
        "merchant_id": np.arange(1, n_merchants + 1),
        "industry": industry,
        "is_service": service,
        "province_tier": tier,
        "n_branches": n_branches,
        "n_users": n_users,
        "n_employees": n_employees,
        "n_customers": n_customers,
        "n_products": n_products,
        "invoices_30d": invoices_30d,
        "tenure_months": tenure_months,
        "uses_einvoice": uses_einvoice,
    })

    ind_eff = np.array([INDUSTRY_ADOPTION_EFFECT[i] for i in industry])
    logit = (1.0 * is_chain + 0.55 * np.log1p(invoices_30d) + 0.35 * np.log1p(n_users)
             + 0.25 * np.log1p(n_employees) + 0.6 * uses_einvoice + 0.3 * (tier == "tier1_metro")
             + 0.15 * (tenure_months > 12) + ind_eff - 2.0 * dormant)
    p = np.exp(logit - logit.max())
    adopters = rng.choice(n_merchants, size=n_adopters, replace=False, p=p / p.sum())
    df["is_adopter"] = False
    df.loc[adopters, "is_adopter"] = True
    return df


def _assign_archetype(m: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    chain = (m["n_branches"] >= 2).to_numpy()
    service = m["is_service"].to_numpy()
    mkt = m["industry"].isin(MARKETING_INDUSTRIES).to_numpy()
    linv = np.log1p(m["invoices_30d"].to_numpy())
    lcust = np.log1p(m["n_customers"].to_numpy())
    organized = (m["n_users"].to_numpy() >= 3) & m["uses_einvoice"].to_numpy()
    scores = np.column_stack([
        -0.6 + 2.2 * chain + 0.7 * (linv - 4.5) + 0.8 * organized,
        0.3 + 1.5 * mkt + 0.5 * (lcust - 6.0) + 0.4 * organized,
        -0.2 + 1.8 * service + 0.3 * (m["n_employees"].to_numpy() >= 3),
        0.3 + 0 * linv,
        -0.3 - 0.8 * chain - 0.3 * organized,
    ])
    probs = np.exp(scores)
    probs /= probs.sum(1, keepdims=True)
    cum = probs.cumsum(1)
    u = rng.random(len(m))[:, None]
    return np.array(ARCHETYPES)[(u > cum).sum(1)]


def _topup_amount(expected_monthly: float, rng: np.random.Generator) -> int:
    target = expected_monthly * rng.uniform(0.8, 3.0)
    idx = np.abs(TOPUP_TIERS - target).argmin()
    return int(TOPUP_TIERS[idx])


def _simulate_adopter(mid, prof, arche, rng, out_days, out_topups, out_stops, out_status_info):
    chain = prof["n_branches"] >= 2
    service = prof["is_service"]
    start = int(rng.integers(0, 560))

    # --- usage parameters by archetype
    if arche == "regular_high":
        p_send, vol, dil = 0.82, np.exp(rng.normal(2.6, 0.45)), rng.beta(8, 2)
    elif arche == "campaign_high":
        p_send, vol, dil = 0.13, np.exp(rng.normal(3.9, 0.45)), rng.beta(5, 3)
    elif arche == "low_regular":
        p_send, vol, dil = 0.22, np.exp(rng.normal(1.0, 0.4)), rng.beta(6, 3)
    elif arche == "low_irregular":
        p_send, vol, dil = 0.035, np.exp(rng.normal(1.3, 0.5)), rng.beta(2.5, 4)
    else:  # trial_only
        p_send, vol, dil = 0.6, 3.0, 0.1
    dil = min(0.98, dil + 0.08 * chain)
    weekend_factor = 1.25 if service else 0.55
    expected_monthly = p_send * vol * 30
    mix = rng.dirichlet(np.array(ARCHETYPE_MIX[arche]) * 40)
    peak_hour = {"regular_high": 14, "campaign_high": rng.choice([10, 19]),
                 "low_regular": 9, "low_irregular": 15, "trial_only": 11}[arche]
    peak_hour = float(peak_hour) + rng.normal(0, 1.2)

    # --- churn mechanisms (days relative to the calendar)
    horizon = N_DAYS + 400
    random_stop = start + int(rng.exponential(2600))
    close_day = start + int(rng.integers(30, 700)) if rng.random() < 0.09 else horizon
    sub_end = start + int(rng.integers(30, 700)) if rng.random() < 0.06 + 0.04 * (not chain) else horizon
    p_cfg = 0.22 - 0.08 * chain
    cfg_onset = start + int(rng.integers(5, 500)) if rng.random() < p_cfg else horizon
    cfg_rate = rng.uniform(0.35, 0.85)
    if rng.random() < 0.65:
        cfg_end, cfg_quit = horizon, cfg_onset + int(rng.integers(12, 40))
    else:
        cfg_end, cfg_quit = cfg_onset + int(rng.integers(7, 30)), horizon
    p_decay = {"campaign_high": 0.25, "low_irregular": 0.22, "low_regular": 0.12, "regular_high": 0.08}.get(arche, 0)
    decay_onset = start + int(rng.integers(20, 500)) if rng.random() < p_decay else horizon
    tau = rng.uniform(15, 40)
    trial_n = int(rng.integers(1, 11))
    user_err_rate = rng.uniform(0.01, 0.06)

    out_status_info.append((mid, close_day, sub_end))

    u_send = rng.random(N_DAYS)
    bal = float(STARTER_CREDIT)
    out_topups.append((mid, start, STARTER_CREDIT, "starter_credit"))
    if arche != "trial_only" and rng.random() < 0.75:
        amt = _topup_amount(expected_monthly, rng)
        bal += amt
        out_topups.append((mid, start, amt, "topup"))

    active, paused_until, react_day = True, -1, -1
    proactive_done = False
    lifetime = 0
    d = start
    while d < N_DAYS:
        if not active:
            if d == react_day:
                active = True
                amt = _topup_amount(expected_monthly, rng)
                bal += amt
                out_topups.append((mid, d, amt, "topup"))
                decay_onset, cfg_onset, cfg_quit = horizon, horizon, horizon
                random_stop = d + int(rng.exponential(2600))
            else:
                if react_day < 0 or d > react_day:
                    break
                d += 1
                continue
        if d < paused_until:
            d += 1
            continue

        reason = None
        if d >= close_day:
            reason = "store_closed"
        elif d >= sub_end:
            reason = "subscription_expired"
        elif d >= cfg_quit:
            reason = "config_errors"
        elif d >= random_stop:
            reason = "unexplained"
        mult = 1.0
        if d >= decay_onset:
            mult = np.exp(-(d - decay_onset) / tau)
            if mult < 0.08 and reason is None:
                reason = "usage_decay"
        if arche == "trial_only" and (lifetime >= trial_n or d > start + 6) and reason is None:
            reason = "trial_only"
        if reason is not None:
            out_stops.append((mid, d, reason))
            active = False
            if reason in {"unexplained", "usage_decay", "config_errors", "balance_depleted"} and rng.random() < 0.18:
                react_day = d + int(rng.integers(45, 220))
            else:
                break
            d += 1
            continue

        dow = (START.dayofweek + d) % 7
        p = p_send * (weekend_factor if dow >= 5 else 1.0)
        if u_send[d] < p:
            att = 1 + rng.poisson(max(vol * mult - 1, 0.05))
            if arche == "trial_only":
                att = min(att, trial_n - lifetime)
            in_cfg = cfg_onset <= d < cfg_end
            cfg = rng.binomial(att, cfg_rate if in_cfg else 0.004)
            ue = rng.binomial(att - cfg, user_err_rate)
            succ = att - cfg - ue
            if succ > bal:
                cut = succ - int(bal)
                succ, att = succ - cut, att - cut
            if att > 0:
                bal -= succ
                lifetime += att
                out_days.append((mid, d, att, succ, cfg, ue, mix, peak_hour))

        if bal < 1 and arche != "trial_only":
            if rng.random() < dil:
                delay = int(rng.geometric(0.25))
                amt = _topup_amount(expected_monthly, rng)
                bal += amt
                paused_until = d + delay
                out_topups.append((mid, d + delay, amt, "topup"))
                proactive_done = False
            else:
                out_stops.append((mid, d, "balance_depleted"))
                active = False
                if rng.random() < 0.18:
                    react_day = d + int(rng.integers(45, 220))
                else:
                    break
        elif not proactive_done and bal < expected_monthly / 4 and arche != "trial_only":
            proactive_done = True
            if rng.random() < dil * 0.6:
                amt = _topup_amount(expected_monthly, rng)
                bal += amt
                out_topups.append((mid, d, amt, "topup"))
                proactive_done = False
        d += 1


def generate(n_merchants: int = 180_000, n_adopters: int = 600, seed: int = 42) -> SyntheticData:
    rng = np.random.default_rng(seed)
    merchants = generate_merchants(n_merchants, n_adopters, rng)
    adopters = merchants[merchants["is_adopter"]].reset_index(drop=True)
    arche = _assign_archetype(adopters, rng)

    days, topups, stops, status_info = [], [], [], []
    for (_, prof), a in zip(adopters.iterrows(), arche):
        _simulate_adopter(int(prof["merchant_id"]), prof, a, rng, days, topups, stops, status_info)

    # ---- expand daily sends into use-case rows
    mids = np.array([r[0] for r in days])
    dnum = np.array([r[1] for r in days])
    counts = np.array([[r[2], r[3], r[4], r[5]] for r in days])
    mixes = np.vstack([r[6] for r in days])
    peaks = np.array([r[7] for r in days])
    split = [rng.multinomial(counts[:, j], mixes) for j in range(4)]  # each: n x 4 use cases
    rows = []
    for k, uc in enumerate(USE_CASES):
        att = split[0][:, k]
        keep = att > 0
        # errors/successes are split independently; reconcile so parts sum to attempts
        cfg = np.minimum(split[2][:, k], att)
        ue = np.minimum(split[3][:, k], att - cfg)
        succ = att - cfg - ue
        hour = np.clip(np.round(peaks + rng.normal(0, 2.2, len(peaks)) + (2 if uc == "marketing" else 0)), 6, 22)
        rows.append(pd.DataFrame({
            "merchant_id": mids[keep], "day": dnum[keep], "use_case": uc, "hour": hour[keep].astype(int),
            "n_attempted": att[keep], "n_success": succ[keep], "n_config_err": cfg[keep], "n_user_err": ue[keep],
        }))
    messages = pd.concat(rows, ignore_index=True)
    messages["date"] = START + pd.to_timedelta(messages.pop("day"), unit="D")
    messages = messages.sort_values(["merchant_id", "date", "use_case"]).reset_index(drop=True)
    messages = messages[["merchant_id", "date", "use_case", "hour", "n_attempted", "n_success",
                         "n_config_err", "n_user_err"]]

    topups_df = pd.DataFrame(topups, columns=["merchant_id", "day", "amount", "kind"])
    topups_df = topups_df[topups_df["day"] < N_DAYS]
    topups_df["date"] = START + pd.to_timedelta(topups_df.pop("day"), unit="D")
    topups_df = topups_df[["merchant_id", "date", "amount", "kind"]].sort_values(["merchant_id", "date"])

    stop_df = pd.DataFrame(stops, columns=["merchant_id", "day", "true_reason"])
    stop_df["date"] = START + pd.to_timedelta(stop_df.pop("day"), unit="D")

    # ---- monthly POS status for adopters
    months = pd.period_range(START, periods=24, freq="M")
    inv = adopters.set_index("merchant_id")["invoices_30d"]
    st_rows = []
    for mid, close_day, sub_end in status_info:
        base = max(inv[mid], 5)
        noise = np.exp(rng.normal(0, 0.15, len(months)))
        for k, m in enumerate(months):
            m_start = (m.start_time - START).days
            m_end = (m.end_time.normalize() - START).days
            pos = base * noise[k]
            if m_start >= close_day:
                pos = 0
            elif m_end >= close_day:
                pos *= (close_day - m_start) / (m_end - m_start + 1)
            elif close_day - m_end <= 60:
                pos *= 0.55
            st_rows.append((mid, m, int(round(pos)), bool(m_start < sub_end)))
    status = pd.DataFrame(st_rows, columns=["merchant_id", "month", "pos_invoices", "core_sub_active"])

    return SyntheticData(merchants=merchants, messages=messages, topups=topups_df.reset_index(drop=True),
                         status=status, stop_events=stop_df)
