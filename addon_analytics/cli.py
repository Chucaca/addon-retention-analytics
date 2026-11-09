"""Command line entry point.

    python -m addon_analytics.cli all          # data -> analyses -> charts -> reports/findings.md
    python -m addon_analytics.cli all --seed 7 --merchants 50000
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import churn as ch  # noqa: E402
from . import lift as lf  # noqa: E402
from . import lookalike as lk  # noqa: E402
from . import plots  # noqa: E402
from . import retention as rt  # noqa: E402
from . import segmentation as sg  # noqa: E402
from . import synthetic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AS_OF = "2025-01-01"
SEEDS = ("S1", "S2", "S3")
# sizing assumptions for the memo (explicitly scenario inputs, not estimates)
RECOVERY = {"balance_depleted": 0.30, "config_errors": 0.50}
SAVED_LIFETIME_MONTHS = 12


def _log(msg, t0):
    print(f"[{time.time() - t0:5.1f}s] {msg}", flush=True)


def run_all(seed: int = 42, n_merchants: int = 180_000, n_adopters: int = 600, out: Path = ROOT) -> dict:
    t0 = time.time()
    rep, img = out / "reports", out / "docs" / "img"
    rep.mkdir(parents=True, exist_ok=True)
    img.mkdir(parents=True, exist_ok=True)

    data = synthetic.generate(n_merchants, n_adopters, seed)
    M, msg = data.merchants, data.messages
    _log(f"synthetic data: {len(M):,} merchants, {M['is_adopter'].sum()} adopters, {len(msg):,} log rows", t0)
    try:
        (out / "data").mkdir(exist_ok=True)
        msg.to_parquet(out / "data" / "messages.parquet", index=False)
        M.to_parquet(out / "data" / "merchants.parquet", index=False)
        data.topups.to_parquet(out / "data" / "topups.parquet", index=False)
    except Exception as e:  # pyarrow optional
        print(f"  (skipping parquet export: {e})")

    # 1. habits
    adopters = M.loc[M["is_adopter"], "merchant_id"]
    seg = sg.segment_merchants(msg, AS_OF, merchant_ids=adopters)
    prof = sg.profile_segments(seg, msg, data.topups, AS_OF)
    hours = sg.hour_profile(seg, msg, AS_OF)
    seg.to_csv(rep / "segments.csv", index=False)
    prof.round(3).to_csv(rep / "segment_profile.csv", index=False)
    plots.segment_profile(prof, hours, img / "segment_profile.png")
    _log("segmentation", t0)

    # 2. retention + use cases
    act = rt.monthly_activity(msg)
    dec = {d: rt.decompose(act, d) for d in rt.DEFINITIONS}
    coh = {f"{d}: active = {lab}": rt.cohort_retention(act, d, min_cohort=10) for d, lab in rt.DEFINITIONS.items()}
    ucr = rt.use_case_retention(msg, act)
    for d, t in dec.items():
        t.to_csv(rep / f"mom_decomposition_{d}.csv", index=False)
    ucr.round(3).to_csv(rep / "use_case_retention.csv", index=False)
    plots.mom_decomposition({f"active = {rt.DEFINITIONS[d]}": t for d, t in dec.items()}, img / "mom_decomposition.png")
    plots.retention_heatmap(coh, img / "retention_heatmap.png")
    plots.use_case_retention(ucr, img / "use_case_retention.png")
    _log("retention", t0)

    # 3. churn
    panel = ch.build_panel(act, data.topups, data.status, M)
    churned = panel[panel["churn"] == 1].copy()
    churned["reason"] = ch.attribute_reasons(churned)
    summ = ch.reason_summary(churned)
    summ.round(3).to_csv(rep / "churn_reasons.csv", index=False)
    truth = ch.match_ground_truth(churned, data.stop_events)
    truth["true_mapped"] = truth["true_reason"].replace(ch.TRUE_TO_RULE)
    pd.crosstab(truth["reason"], truth["true_reason"]).to_csv(rep / "churn_attribution_vs_truth.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml = ch.ml_crosscheck(panel)
    ml["importance"].round(4).to_csv(rep / "churn_model_importance.csv", index=False)
    plots.churn_reasons(summ, img / "churn_reasons.png")
    plots.churn_importance(ml["importance"], ch.FEATURE_TO_RULE, img / "churn_model_importance.png")
    _log(f"churn: {len(churned)} churn events, GBM AUC {ml['auc_gbm']:.3f}", t0)

    # 4. lift + lookalike
    feats = lf.add_bands(M)
    segmap = M["merchant_id"].map(seg.set_index("merchant_id")["segment"])
    seed_mask = segmap.isin(SEEDS).to_numpy()
    base_mask = (~M["is_adopter"]).to_numpy()
    single = (M["n_branches"] == 1).to_numpy()
    lift = lf.profile_lift(feats, seed_mask, base_mask)
    lift_single = lf.profile_lift(feats, seed_mask & single, base_mask & single)
    lift.round(4).to_csv(rep / "lift_all.csv", index=False)
    lift_single.round(4).to_csv(rep / "lift_single_store.csv", index=False)
    rules = lk.derive_rules(lift, lift_single)
    cov = lk.evaluate_coverage(M, seg, rules)
    ucs = lk.suggested_use_cases(M, seg, msg, rules)
    targets = lk.target_list(M, feats, lift, rules, ucs)
    targets.head(2000).to_csv(rep / "target_list_top2000.csv", index=False)
    cov.round(4).to_csv(rep / "rule_coverage.csv", index=False)
    plots.lift_bars(lift, img / "lift_bars.png")
    plots.rule_coverage(cov, img / "rule_coverage.png")
    _log("lift + lookalike", t0)

    # optional SQL cross-check
    try:
        from .sql_check import run_sql_checks
        sql_note = run_sql_checks(ROOT / "sql", msg, seg)
    except ImportError as e:
        sql_note = f"skipped ({e})"
    _log(f"SQL cross-check: {sql_note}", t0)

    facts = _facts(M, seg, prof, dec, ucr, panel, churned, summ, truth, ml, lift, rules, cov, ucs, targets)
    facts["sql_check"] = sql_note
    (rep / "findings.md").write_text(_memo(facts), encoding="utf-8")
    (rep / "metrics.json").write_text(json.dumps({k: v for k, v in facts.items() if not isinstance(v, pd.DataFrame)},
                                                 indent=2, default=str), encoding="utf-8")
    _log("wrote reports/findings.md", t0)
    return facts


def _facts(M, seg, prof, dec, ucr, panel, churned, summ, truth, ml, lift, rules, cov, ucs, targets) -> dict:
    p = prof.set_index("segment")
    s = summ.set_index("reason")
    last_lab = panel.dropna(subset=["churn"])["month"].max()
    recent = churned[churned["month"] > last_lab - 12]
    rs = recent["reason"].value_counts()
    spend = recent.groupby("reason")["success"].mean()
    sized = {}
    for r, rate in RECOVERY.items():
        n = int(rs.get(r, 0))
        sized[r] = {"events_12m": n, "recovery_assumption": rate, "merchants_saved": n * rate,
                    "credits_saved_per_year": n * rate * float(spend.get(r, 0)) * SAVED_LIFETIME_MONTHS}
    acc = (truth["reason"] == truth["true_mapped"])
    with_event = truth["true_reason"] != "no_stop_event"
    d_any, d_real = dec["any"], dec["real"]
    ucr_i = ucr.set_index("group")
    covi = cov.set_index("rules")
    seg_counts = seg["segment"].value_counts()
    return {
        "n_merchants": len(M), "n_adopters": int(M["is_adopter"].sum()),
        "adoption_rate": float(M["is_adopter"].mean()),
        "segments": {k: int(seg_counts.get(k, 0)) for k in ["S1", "S2", "S3", "S4", "S0"]},
        "profile": prof,
        "s1_msgs": float(p.loc["S1", "median_msgs_per_month"]), "s4_msgs": float(p.loc["S4", "median_msgs_per_month"]),
        "s1_topup": float(p.loc["S1", "topup_rate"]), "s3_topup": float(p.loc["S3", "topup_rate"]),
        "s4_topup": float(p.loc["S4", "topup_rate"]),
        "active_any_last": int(d_any["active"].iloc[-1]), "active_real_last": int(d_real["active"].iloc[-1]),
        "churn_any_avg12": float(d_any["churn_rate"].tail(12).mean()),
        "churn_real_avg12": float(d_real["churn_rate"].tail(12).mean()),
        "react_share_any": float(d_any["reactivated"].tail(12).sum() / d_any["active"].tail(12).sum()),
        "ucr": ucr_i,
        "churn_events": len(churned), "panel_rows": int(panel["churn"].notna().sum()),
        "monthly_churn_rate": float(panel["churn"].mean()),
        "reason_summary": summ,
        "share_direct": float(s.loc[["balance_depleted", "config_errors"], "share"].sum()),
        "share_broad": float(s.loc[[r for r, a in ch.ADDRESSABLE.items() if a in ("yes", "partly")], "share"].sum()),
        "share_exogenous": float(s.loc[["store_stopped_selling", "core_subscription_expired"], "share"].sum()),
        "share_unexplained": float(s.loc["unexplained", "share"]),
        "recent_months": f"{last_lab - 11} .. {last_lab}", "sized": sized,
        "attr_acc_with_event": float(acc[with_event].mean()), "n_with_event": int(with_event.sum()),
        "no_event_share": float((~with_event).mean()),
        "no_event_reasons": truth.loc[~with_event, "reason"].value_counts().to_dict(),
        "ml": {k: v for k, v in ml.items() if k not in ("importance", "logit_coefs")},
        "ml_top": ml["importance"].head(6).assign(rule=lambda d: d["feature"].map(ch.FEATURE_TO_RULE)),
        "lift_top": lift[lift["keep"] & (lift["lift"] > 1)].nlargest(8, "z"),
        "rules": rules, "coverage": cov, "use_cases": ucs,
        "cov_abc": covi.loc["A+B+C"].to_dict(), "n_targets": len(targets),
    }


def _pct(x):
    return f"{x:.0%}"


def _memo(f: dict) -> str:
    r = f["rules"]
    ucr = f["ucr"]
    cov = f["cov_abc"]
    sz = f["sized"]
    bal, cfg = sz["balance_depleted"], sz["config_errors"]
    reason_rows = "\n".join(
        f"| {row.reason} | {row.churn_events} | {_pct(row.share)} | {row.addressable} | {row.lever} |"
        for row in f["reason_summary"].itertuples())
    prof_rows = "\n".join(
        f"| {row.segment} | {row.merchants} | {row.median_msgs_per_month:,.0f} | {_pct(row.topup_rate)} | "
        f"{row.median_topup_amount:,.0f} | {row.mean_send_hour:.1f} | {_pct(row.weekend_share)} | "
        f"{_pct(row.share_transactional)} / {_pct(row.share_customer_care)} / {_pct(row.share_marketing)} |"
        for row in f["profile"].itertuples())
    lift_rows = "\n".join(f"| {row.feature} = {row.value} | {row.lift:.1f}x | {row.z:+.1f} | {row.n_seed} |"
                          for row in f["lift_top"].itertuples())
    cov_rows = "\n".join(
        f"| {row.rules} | {_pct(row.base_share)} ({row.base_selected:,}) | {_pct(row.seed_recall)} | "
        f"{_pct(row.recall_S1)} / {_pct(row.recall_S2)} / {_pct(row.recall_S3)} | {row.concentration:.1f}x |"
        for row in f["coverage"].itertuples())
    ml_rows = "\n".join(f"| {row.feature} | {row.importance:.3f} | {row.rule} |" for row in f["ml_top"].itertuples())
    ml = f["ml"]
    tx, cc, mk = (ucr.loc[g, "retained_m6"] for g in ("transactional", "customer_care", "marketing"))
    multi, single = ucr.loc["multi_use_case=True", "retained_m6"], ucr.loc["multi_use_case=False", "retained_m6"]
    if multi > single + 0.03:
        breadth = (f"Using 2+ use cases early also helps ({_pct(multi)} vs {_pct(single)} at 6 months, "
                   "trial merchants excluded).")
    else:
        breadth = (f"Breadth alone does not help: 2+ early use cases retain {_pct(multi)} vs {_pct(single)} for "
                   "single-use-case merchants (trial merchants excluded) - the *type* of use case matters more.")
    uc = f["use_cases"]
    trial = ucr.loc["trial (<=10 msgs)", "retained_m6"] if "trial (<=10 msgs)" in ucr.index else float("nan")
    p_s3_amt = f["profile"].set_index("segment").loc["S3", "median_topup_amount"]
    return f"""# Findings memo - prepaid messaging add-on: habits, retention, churn, rollout targeting

> **All numbers below come from a seeded SYNTHETIC dataset** generated by `addon_analytics/synthetic.py`.
> They illustrate the method and are not real business results.
> Reproduce with `python -m addon_analytics.cli all`.

## TL;DR

* **{f['n_adopters']} of {f['n_merchants']:,} merchants ({f['adoption_rate']:.2%})** use the add-on. In the last 3 months,
  **{f['segments']['S1']} are regular high-volume users (S1)**: median {f['s1_msgs']:,.0f} messages/month, {_pct(f['s1_topup'])} top up.
  The long tail (S4, {f['segments']['S4']} merchants) sends ~{f['s4_msgs']:.0f} messages/month and only {_pct(f['s4_topup'])} top up.
* **Always-on use cases retain; campaign-only use does not.** 6-month "real use" retention: transactional {_pct(tx)},
  customer care {_pct(cc)}, marketing-led {_pct(mk)}. {breadth}
* **{_pct(f['share_direct'])} of churn is directly addressable by the add-on team** (balance ran out and no top-up, or
  blocked by template-configuration errors); {_pct(f['share_broad'])} including onboarding and usage decay.
  Only {_pct(f['share_exogenous'])} is exogenous (store closed / core subscription lapsed).
* **Lookalike rules A+B+C select {_pct(cov['base_share'])} of non-adopters yet cover {_pct(cov['seed_recall'])} of current good
  adopters** ({cov['concentration']:.1f}x concentration) -> {f['n_targets']:,} prioritised targets.

## 1. Usage habits and top-up behaviour

Rule-based segments over the 3 months / 14 weeks before {AS_OF} (S0 = adopters with no message in the window:
{f['segments']['S0']}).

| Segment | Merchants | Median msgs / month | Paid top-up rate | Median top-up (credits) | Mean send hour | Weekend share | Mix tx / care / mkt |
|---|---|---|---|---|---|---|---|
{prof_rows}

* S1 is the revenue core: high volume, near-universal top-ups, daytime transactional sends.
* S2 (irregular high volume) is campaign-driven: the highest marketing share and bursty send days.
* S3 is the "reminder" pattern - small but steady monthly volumes; top-ups are small (median {p_s3_amt:,.0f} credits).
* S4 rarely tops up: these merchants live on the starter credit and are the first to run dry.

![Segment profile](../docs/img/segment_profile.png)

## 2. Which use cases drive retention

* Active base at the end of the period: **{f['active_any_last']}** merchants with >=1 message vs **{f['active_real_last']}** with
  >=3 send days ("real use"). Average month-over-month churn over the last 12 months: {_pct(f['churn_any_avg12'])} (any)
  vs {_pct(f['churn_real_avg12'])} (real use). Reactivations are {_pct(f['react_share_any'])} of monthly active merchants,
  so a share of "churn" is really pauses.
* Retention by dominant use case in the first 60 days (6-month, real use): transactional {_pct(tx)}, customer care {_pct(cc)},
  marketing {_pct(mk)}. Merchants whose first 60 days were trial-only (<=10 messages) almost never stay
  ({_pct(trial)} at 6 months).

![Use-case retention](../docs/img/use_case_retention.png)
![Retention heatmap](../docs/img/retention_heatmap.png)

## 3. Why merchants churn, and how much is addressable

Churn at month M = active in M, nothing in M+1 and M+2. {f['churn_events']} churn events over {f['panel_rows']:,} labelled
active merchant-months ({_pct(f['monthly_churn_rate'])} per month). Reasons are attributed by the first rule that fires:

| Reason (priority order) | Events | Share | Addressable | Lever |
|---|---|---|---|---|
{reason_rows}

![Churn reasons](../docs/img/churn_reasons.png)

**Validation.** Because the data is simulated, the true stop reason is known. Where the simulator recorded a stop event
(n={f['n_with_event']}), the heuristic picks the true reason **{_pct(f['attr_acc_with_event'])}** of the time.
{_pct(f['no_event_share'])} of "churn" events have no stop event at all - low-frequency merchants whose natural gaps exceed
two months; they land mostly in {', '.join(f'{k} ({v})' for k, v in f['no_event_reasons'].items())}. This is the main
weakness of a fixed 2-month churn window for low-frequency users.

**ML cross-check.** A gradient-boosting model on a time split (train before {ml['test_from']}, test after;
{ml['test_rows']} test rows, churn rate {_pct(ml['test_churn_rate'])}) reaches **AUC {ml['auc_gbm']:.2f}**
(logistic regression {ml['auc_logit']:.2f}). Its top permutation importances map onto the rule families:

| Feature | Importance (AUC drop) | Rule family |
|---|---|---|
{ml_rows}

![Churn model importance](../docs/img/churn_model_importance.png)

**Sizing (scenario, last 12 labelled months {f['recent_months']}).**

| Lever | Churn events / year | Assumed recovery | Merchants saved / year | Credits retained / year* |
|---|---|---|---|---|
| Low-balance top-up nudge + auto top-up | {bal['events_12m']} | {_pct(bal['recovery_assumption'])} | {bal['merchants_saved']:.0f} | {bal['credits_saved_per_year']:,.0f} |
| Config-error alerting + guided fix | {cfg['events_12m']} | {_pct(cfg['recovery_assumption'])} | {cfg['merchants_saved']:.0f} | {cfg['credits_saved_per_year']:,.0f} |

\\* saved merchants x their average monthly spend in the churn month x {SAVED_LIFETIME_MONTHS} months. Recovery rates are
assumptions to be replaced by an A/B test, not estimates.

## 4. Lookalike targeting for the rollout

Lift = share among good adopters (S1-S3) / share among non-adopters, two-proportion z-test, kept if |z| >= 2 and n >= 5.
Strongest over-representation:

| Feature value | Lift | z | n (seed) |
|---|---|---|---|
{lift_rows}

![Lift](../docs/img/lift_bars.png)

Rules derived from the lift tables (thresholds read off the lowest significantly over-represented band). The suggested
use case is the one most over-indexed in the message mix of the good adopters each rule already covers:

* **A - chains:** >= 2 branches -> suggested use case: **{uc.get('A', 'n/a')}**
* **B - organized single stores:** >= {r['min_users_B']} users, e-invoice in use, >= {r['min_invoices_B']} invoices / 30 days -> **{uc.get('B', 'n/a')}**
* **C - service businesses with staff:** {', '.join(r['service_industries'])}; >= {r['min_employees_C']} employees and >= {r['min_invoices_C']} invoices / 30 days -> **{uc.get('C', 'n/a')}**

| Rules | Base selected | Recall of good adopters | Recall S1 / S2 / S3 | Concentration |
|---|---|---|---|---|
{cov_rows}

![Rule coverage](../docs/img/rule_coverage.png)

The top 2,000 targets are in `reports/target_list_top2000.csv` (priority A > B > C, then lift score).

## Recommendations

1. **Low-balance nudges and one-click / auto top-up.** Trigger when runway < ~1 week of typical spend; "balance ran out
   and no top-up" is the single largest churn reason ({_pct(f['reason_summary'].set_index('reason').loc['balance_depleted', 'share'])}).
2. **Config-error alerting.** Alert the merchant (and support) when the template-configuration failure rate exceeds 10%
   in a day; offer a guided fix. These merchants want to send but cannot.
3. **Onboard into always-on flows first.** Transactional and customer-care merchants retain best;
   set up order and appointment notifications before marketing blasts, and follow up trial-only merchants within the first week.
4. **Roll out to A -> B -> C segments**, with the suggested use case in the pitch, and measure with a holdout so the
   descriptive lift can be turned into a causal conversion estimate.
5. **Use "real use" (>=3 send days) as the health KPI** and a longer churn window for low-frequency merchants.

## Limitations

* Lift is **descriptive over-representation, not causal uplift**; targeting effects need an experiment.
* Churn-reason attribution is a **priority-ordered heuristic**; reasons overlap and the order is a judgement call.
* The balance is *estimated* (cumulative credits - successful sends); real systems may have promos, refunds and expiry.
* Synthetic data encodes the mechanisms the analysis looks for; the value here is the method, not the numbers.

_Optional SQL cross-check: {f['sql_check']}_
"""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="addon_analytics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all", help="generate data, run every analysis, write charts and reports/findings.md")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--merchants", type=int, default=180_000)
    a.add_argument("--adopters", type=int, default=600)
    args = ap.parse_args(argv)
    if args.cmd == "all":
        run_all(args.seed, args.merchants, args.adopters)


if __name__ == "__main__":
    main()
