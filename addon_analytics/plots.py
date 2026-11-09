"""Static charts (matplotlib, Agg backend). All charts are labelled as synthetic data."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# fixed categorical order (validated default palette); never cycled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEG_COLOR = {"S1": SERIES[0], "S2": SERIES[1], "S3": SERIES[2], "S4": SERIES[3]}
ADDR_COLOR = {"yes": SERIES[0], "partly": SERIES[2], "no": "#b5b3ab", "unknown": SERIES[3]}
SYNTH = "Synthetic data - illustrative only"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
})


def _finish(fig, path: Path, title: str):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.99, 0.005, SYNTH, ha="right", va="bottom", fontsize=8, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def segment_profile(prof: pd.DataFrame, hours: pd.DataFrame, path: Path):
    prof = prof.set_index("segment")
    segs = [s for s in ["S1", "S2", "S3", "S4"] if s in prof.index]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    panels = [("median_msgs_per_month", "Median messages / month", "{:,.0f}"),
              ("merchants", "Merchants in segment", "{:,.0f}"),
              ("topup_rate", "Paid top-up rate (3 months)", "{:.0%}"),
              ("median_topup_amount", "Median top-up (credits)", "{:,.0f}"),
              ("weekend_share", "Weekend share of messages", "{:.0%}")]
    for ax, (col, title, fmt) in zip(axes.flat, panels):
        vals = prof.loc[segs, col].fillna(0)
        bars = ax.bar(segs, vals, color=[SEG_COLOR[s] for s in segs], width=0.6, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom",
                    fontsize=8.5, color=INK2)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
        if "{:.0%}" in fmt:
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax = axes.flat[5]
    for s in segs:
        if s in hours.columns:
            ax.plot(hours.index, hours[s], color=SEG_COLOR[s], lw=2, label=s)
    ax.set_title("Hour-of-day share of messages", loc="left")
    ax.set_xticks(range(6, 23, 2))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("hour")
    ax.legend(ncol=2, fontsize=8)
    _finish(fig, path, "Usage-habit segments: S1 regular high / S2 irregular high / S3 low monthly / S4 low irregular")


def lift_bars(lift: pd.DataFrame, path: Path, top: int = 22):
    k = lift[lift["keep"]].copy()
    k = pd.concat([k[k["lift"] > 1].nlargest(top - 6, "z"), k[k["lift"] < 1].nsmallest(6, "z")])
    k = k.sort_values("lift")
    labels = k["feature"] + " = " + k["value"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(k) + 1.5))
    colors = np.where(k["lift"] > 1, SERIES[0], SERIES[1])
    ax.barh(labels, np.log2(k["lift"]), color=colors, height=0.65, edgecolor=SURFACE, linewidth=1.5)
    for y, (l, z, n) in enumerate(zip(k["lift"], k["z"], k["n_seed"])):
        x = np.log2(l)
        ax.text(x + (0.08 if x > 0 else -0.08), y, f"{l:.1f}x  (z={z:+.1f}, n={n})", va="center",
                ha="left" if x > 0 else "right", fontsize=8, color=INK2)
    ticks = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    ax.set_xticks(ticks, [f"{2.0 ** t:g}x" for t in ticks])
    lo, hi = np.log2(k["lift"]).min(), np.log2(k["lift"]).max()
    ax.set_xlim(lo - 1.8, hi + 2.0)
    ax.axvline(0, color=AXIS, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("lift = share among good adopters (S1-S3) / share in non-adopter base (log scale)")
    _finish(fig, path, "Profile lift of good adopters vs non-adopters (|z| >= 2, n >= 5) - descriptive, not causal")


def retention_heatmap(cohorts: dict, path: Path):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(16, 6.5), gridspec_kw={"wspace": 0.42})
    for ax, (name, tab) in zip(np.atleast_1d(axes), cohorts.items()):
        data = tab.drop(columns="cohort_size")
        im = ax.imshow(data.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(data.shape[1]), data.columns)
        ax.set_yticks(range(len(data)), [f"{c} (n={n})" for c, n in zip(data.index.astype(str), tab["cohort_size"])])
        ax.set_xlabel("months since first active month")
        ax.set_title(name, loc="left")
        ax.grid(False)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data.iat[i, j]
                if pd.notna(v) and j in (1, 3, 6, 12):
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6.5,
                            color="#ffffff" if v > 0.55 else INK)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, format=lambda x, _: f"{x:.0%}")
    fig.suptitle("Monthly cohort retention under two activity definitions", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.99, 0.005, SYNTH, ha="right", va="bottom", fontsize=8, color=MUTED, style="italic")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def mom_decomposition(decomps: dict, path: Path):
    fig, axes = plt.subplots(len(decomps), 1, figsize=(12, 6.5), sharex=True)
    parts = [("retained", SERIES[0]), ("new", SERIES[2]), ("reactivated", SERIES[3])]
    for ax, (name, d) in zip(np.atleast_1d(axes), decomps.items()):
        x = np.arange(len(d))
        bottom = np.zeros(len(d))
        for col, c in parts:
            ax.bar(x, d[col], bottom=bottom, color=c, label=col, width=0.75, edgecolor=SURFACE, linewidth=1)
            bottom += d[col].to_numpy()
        ax.bar(x, -d["churned"], color=SERIES[1], label="churned (vs last month)", width=0.75,
               edgecolor=SURFACE, linewidth=1)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_title(name, loc="left")
        ax.set_ylabel("merchants")
        ax.grid(axis="x", visible=False)
        ax.set_xticks(x, d["month"].astype(str), rotation=60, fontsize=8)
    np.atleast_1d(axes)[0].legend(ncol=4, loc="upper left")
    _finish(fig, path, "Month-over-month active-base decomposition")


def churn_reasons(summary: pd.DataFrame, path: Path):
    s = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    colors = [ADDR_COLOR[a] for a in s["addressable"]]
    ax.barh(s["reason"], s["share"], color=colors, height=0.6, edgecolor=SURFACE, linewidth=1.5)
    for y, (v, n) in enumerate(zip(s["share"], s["churn_events"])):
        ax.text(v + 0.005, y, f"{v:.0%}  ({n})", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, s["share"].max() * 1.25)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of churn events (rows listed in attribution priority order, top = checked first)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f"addressable: {k}") for k, c in ADDR_COLOR.items()],
              loc="lower right", fontsize=8)
    _finish(fig, path, "Why merchants churn - priority-ordered rule attribution")


def churn_importance(imp: pd.DataFrame, feature_to_rule: dict, path: Path):
    d = imp.sort_values("importance")
    rules = list(dict.fromkeys(feature_to_rule[f] for f in imp["feature"]))
    cmap = {r: SERIES[i % len(SERIES)] for i, r in enumerate(rules)}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(d["feature"], d["importance"], xerr=d["std"], color=[cmap[feature_to_rule[f]] for f in d["feature"]],
            height=0.6, edgecolor=SURFACE, linewidth=1.5, error_kw={"ecolor": MUTED, "lw": 1})
    ax.axvline(0, color=AXIS, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("permutation importance (drop in test AUC)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cmap[r], label=f"rule: {r}") for r in rules], loc="lower right", fontsize=8)
    _finish(fig, path, "ML cross-check: what the gradient-boosting churn model relies on")


def rule_coverage(cov: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = cov["base_share"].to_numpy()
    for i, (col, lab) in enumerate([("seed_recall", "all good adopters"), ("recall_S1", "S1"),
                                    ("recall_S2", "S2"), ("recall_S3", "S3")]):
        if col not in cov:
            continue
        c = INK if col == "seed_recall" else SEG_COLOR[col[-2:]]
        ax.plot(x, cov[col], marker="o", ms=8, lw=2.5 if col == "seed_recall" else 1.6, color=c, label=lab,
                markeredgecolor=SURFACE, markeredgewidth=2)
    for xi, yi, r in zip(x, cov["seed_recall"], cov["rules"]):
        ax.annotate(r, (xi, yi), textcoords="offset points", xytext=(-8, 10), ha="right", fontsize=9,
                    color=INK, fontweight="bold")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="random targeting")
    ax.set_xlim(0, max(0.35, x.max() * 1.3))
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("share of non-adopter base selected")
    ax.set_ylabel("recall of current good adopters")
    ax.legend(loc="lower right", fontsize=8.5)
    _finish(fig, path, "Lookalike rules: adopter coverage vs selection size")


def use_case_retention(tab: pd.DataFrame, path: Path):
    d = tab[~tab["group"].str.startswith("multi")].copy()
    d = pd.concat([d.sort_values("retained_m6", ascending=False), tab[tab["group"].str.startswith("multi")]])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(d))
    w = 0.38
    for k, (col, c) in enumerate([("retained_m3", SERIES[0]), ("retained_m6", SERIES[6])]):
        ax.bar(x + (k - 0.5) * w, d[col], width=w, color=c, label=col.replace("retained_m", "month ") + " retention",
               edgecolor=SURFACE, linewidth=1.5)
    for xi, v in zip(x, d["retained_m6"]):
        ax.text(xi + w / 2, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"{g}\n(n={n})" for g, n in zip(d["group"], d["merchants"])], fontsize=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax.text(0.99, 0.80, "multi_use_case excludes trial merchants", transform=ax.transAxes, ha="right",
            fontsize=8, color=MUTED)
    _finish(fig, path, "Retention ('real use') by dominant use case in the first 60 days")
