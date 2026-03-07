#!/usr/bin/env python3
"""Generate decision-oriented error-type plots and conclusions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "OpenAI",
    "Anthropic Claude Sonnet 4.6",
    "DeepSeek Reasoner",
    "Mistral Large",
]


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    base_dir = script_path.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=base_dir / "error_repair_insights",
        help="Folder containing error_repair_insights/{data,figures}.",
    )
    return parser.parse_args()


def draw_burden_pareto(summary: pd.DataFrame, out: Path) -> None:
    df = summary.sort_values("total_extra_iterations", ascending=False).copy()
    df["cum_extra_pct"] = 100.0 * df["total_extra_iterations"].cumsum() / df["total_extra_iterations"].sum()
    df["cum_time_pct"] = 100.0 * df["total_repair_sec"].cumsum() / df["total_repair_sec"].sum()
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(11, 5.2), dpi=220)
    ax1.bar(x, df["total_extra_iterations"].to_numpy(dtype=float), color="#4C78A8")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["family"].tolist(), rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("Total Additional Iterations", fontsize=9)
    ax1.set_title("Pareto of Repair Burden by Error Family", fontsize=11)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, df["cum_extra_pct"].to_numpy(dtype=float), color="#E45756", marker="o", linewidth=1.8, label="Cumulative extra-iteration share")
    ax2.plot(x, df["cum_time_pct"].to_numpy(dtype=float), color="#54A24B", marker="s", linewidth=1.6, label="Cumulative repair-time share")
    ax2.set_ylabel("Cumulative Share (%)", fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.legend(frameon=False, fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_priority_matrix(summary: pd.DataFrame, out: Path, label_top: int = 10) -> None:
    df = summary.copy()
    x = df["episodes"].to_numpy(dtype=float)
    y = df["mean_iterations_to_repair"].to_numpy(dtype=float)
    size = 80 + 12.0 * df["total_extra_iterations"].to_numpy(dtype=float)
    color = df["exact_recurrence_rate_persist"].fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=220)
    sc = ax.scatter(
        x, y, s=size, c=color, cmap="YlOrRd", vmin=0.0, vmax=1.0, alpha=0.82, edgecolors="white", linewidths=0.5
    )
    ax.set_xscale("log")
    ax.set_xlabel("Episodes (log scale)", fontsize=9)
    ax.set_ylabel("Mean Additional Iterations to Repair", fontsize=9)
    ax.set_title("Priority Matrix: Frequency × Difficulty × Repeatability", fontsize=11)
    ax.grid(alpha=0.25)

    top = df.sort_values("total_extra_iterations", ascending=False).head(label_top)
    for _, r in top.iterrows():
        ax.text(float(r["episodes"]) * 1.04, float(r["mean_iterations_to_repair"]) + 0.01, str(r["family"]), fontsize=7)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Exact Recurrence Rate (persistent episodes)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_model_concentration(episodes: pd.DataFrame, out: Path) -> None:
    fam_group = episodes.copy()
    fam_group["bucket"] = np.where(
        fam_group["family"] == "parsing-error",
        "Parsing",
        np.where(fam_group["family"] == "reference-error", "Reference", "Other"),
    )
    agg = (
        fam_group.groupby(["model_label", "bucket"])["iterations_to_repair"]
        .sum()
        .rename("total_extra_iterations")
        .reset_index()
    )
    pivot = (
        agg.pivot(index="model_label", columns="bucket", values="total_extra_iterations")
        .fillna(0.0)
        .reindex(index=MODEL_ORDER)
        .reindex(columns=["Parsing", "Reference", "Other"])
    )
    pct = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0

    x = np.arange(len(pct.index))
    fig, ax = plt.subplots(figsize=(9.8, 4.8), dpi=220)
    bottom = np.zeros(len(x), dtype=float)
    colors = {"Parsing": "#4C78A8", "Reference": "#F58518", "Other": "#54A24B"}

    for col in ["Parsing", "Reference", "Other"]:
        vals = pct[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=colors[col], label=col)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(pct.index.tolist(), rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of Model's Total Extra Iterations (%)", fontsize=9)
    ax.set_title("Burden Concentration by Model: Parsing/Reference vs Everything Else", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_top10_burden_and_recurrence(summary: pd.DataFrame, out: Path) -> None:
    top = summary.sort_values("total_extra_iterations", ascending=False).head(10).copy()
    y = np.arange(len(top))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.8, 5.4), dpi=220, gridspec_kw={"width_ratios": [1.2, 1.0]}
    )

    ax1.barh(y, top["share_extra_iterations_pct"].to_numpy(dtype=float), color="#4C78A8")
    ax1.set_yticks(y)
    ax1.set_yticklabels(top["family"].tolist(), fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Share of Total Extra Iterations (%)", fontsize=9)
    ax1.set_title("Top-10 Families by Iteration Burden", fontsize=10)
    ax1.grid(axis="x", alpha=0.25)
    for i, v in enumerate(top["share_extra_iterations_pct"].to_numpy(dtype=float)):
        ax1.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=7)

    ax2.barh(y, 100.0 * top["exact_recurrence_rate_persist"].fillna(0.0).to_numpy(dtype=float), color="#E45756")
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(y))
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Exact Recurrence Rate in Persistent Episodes (%)", fontsize=9)
    ax2.set_title("Repeatability Signal (Same Exact Diagnostic)", fontsize=10)
    ax2.grid(axis="x", alpha=0.25)
    for i, v in enumerate(100.0 * top["exact_recurrence_rate_persist"].fillna(0.0).to_numpy(dtype=float)):
        ax2.text(v + 1.0, i, f"{v:.0f}%", va="center", fontsize=7)

    fig.suptitle("What to Fix First: Burden and Repeatability Together", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def build_priority_summary(episodes: pd.DataFrame, identity_eps: pd.DataFrame) -> pd.DataFrame:
    fam = (
        episodes.groupby("family")
        .agg(
            episodes=("family", "count"),
            total_extra_iterations=("iterations_to_repair", "sum"),
            mean_iterations_to_repair=("iterations_to_repair", "mean"),
            median_iterations_to_repair=("iterations_to_repair", "median"),
            p90_iterations_to_repair=("iterations_to_repair", lambda s: s.quantile(0.9)),
            total_repair_sec=("elapsed_sec_to_repair", "sum"),
            mean_repair_sec=("elapsed_sec_to_repair", "mean"),
        )
        .reset_index()
    )

    persisted = identity_eps[identity_eps["iterations_with_family"] >= 2].copy()
    recur = (
        persisted.groupby("family")
        .agg(
            persistent_episodes=("family", "count"),
            exact_recurrence_rate_persist=("has_exact_recurrence", "mean"),
            template_recurrence_rate_persist=("has_template_recurrence", "mean"),
            churn_rate_persist=("persistence_mode", lambda s: float((s == "family_only_churn").mean())),
        )
        .reset_index()
    )

    out = fam.merge(recur, on="family", how="left")
    out["persistent_episodes"] = out["persistent_episodes"].fillna(0).astype(int)
    out["exact_recurrence_rate_persist"] = out["exact_recurrence_rate_persist"].fillna(0.0)
    out["template_recurrence_rate_persist"] = out["template_recurrence_rate_persist"].fillna(0.0)
    out["churn_rate_persist"] = out["churn_rate_persist"].fillna(0.0)

    out["share_extra_iterations_pct"] = 100.0 * out["total_extra_iterations"] / out["total_extra_iterations"].sum()
    out["share_repair_time_pct"] = 100.0 * out["total_repair_sec"] / out["total_repair_sec"].sum()

    # priority score emphasizes burden + repeatability
    out["priority_score"] = (
        0.5 * out["share_extra_iterations_pct"]
        + 0.3 * out["share_repair_time_pct"]
        + 0.2 * (100.0 * out["exact_recurrence_rate_persist"])
    )

    return out.sort_values("priority_score", ascending=False).reset_index(drop=True)


def write_conclusions(summary: pd.DataFrame, out_path: Path) -> None:
    s = summary.sort_values("total_extra_iterations", ascending=False).copy()
    top1 = s.head(1)
    top2 = s.head(2)
    top3 = s.head(3)
    top5 = s.head(5)
    high_repeat = s[s["persistent_episodes"] >= 3].sort_values("exact_recurrence_rate_persist", ascending=False).head(5)
    slow = s[s["episodes"] >= 3].sort_values("mean_repair_sec", ascending=False).head(5)

    lines: List[str] = []
    lines.append("# Actionable Conclusions: Which Error Types Matter Most")
    lines.append("")
    lines.append("## What To Care About")
    lines.append("")
    lines.append("- Prioritize error types by **total extra iterations** and **total repair time** (not raw count alone).")
    lines.append("- Favor high-repeatability families first: repeated exact diagnostics are best candidates for deterministic fixes.")
    lines.append("- Separate rare-but-expensive families from frequent families; each needs different intervention.")
    lines.append("")
    lines.append("## High-Value Findings")
    lines.append("")
    lines.append(
        f"- `parsing-error` alone accounts for **{float(top1['share_extra_iterations_pct'].sum()):.1f}%** of extra iterations and **{float(top1['share_repair_time_pct'].sum()):.1f}%** of repair time."
    )
    lines.append(
        f"- `parsing-error` + `reference-error` together account for **{float(top2['share_extra_iterations_pct'].sum()):.1f}%** of extra iterations and **{float(top2['share_repair_time_pct'].sum()):.1f}%** of repair time."
    )
    lines.append(
        f"- Top-3 families reach **{float(top3['share_extra_iterations_pct'].sum()):.1f}%** of extra-iteration burden; top-5 reach **{float(top5['share_extra_iterations_pct'].sum()):.1f}%**."
    )
    lines.append("")
    lines.append("Most repeatable families (persistent episodes >=3):")
    for _, r in high_repeat.iterrows():
        lines.append(
            f"- `{r['family']}`: exact recurrence {100.0*float(r['exact_recurrence_rate_persist']):.1f}%, persistent episodes {int(r['persistent_episodes'])}, burden share {float(r['share_extra_iterations_pct']):.1f}%."
        )
    lines.append("")
    lines.append("Slow-to-repair families (episodes >=3):")
    for _, r in slow.iterrows():
        lines.append(
            f"- `{r['family']}`: mean repair time {float(r['mean_repair_sec']):.1f}s, mean additional iterations {float(r['mean_iterations_to_repair']):.2f}."
        )
    lines.append("")
    lines.append("## Suggested Intervention Priority")
    lines.append("")
    lines.append("1. Build hard-coded repair rules and prompt checks for parsing/reference diagnostics first (highest ROI).")
    lines.append("2. Add targeted remediation templates for feature/connector families with high repeatability but lower frequency.")
    lines.append("3. Add specialized handling for rare high-latency families to cut wall-clock tail.")
    lines.append("")
    lines.append("## Figure Map")
    lines.append("")
    lines.append("- `figures/figE11_error_burden_pareto_iterations_time.png`")
    lines.append("- `figures/figE12_error_priority_matrix_bubble.png`")
    lines.append("- `figures/figE13_model_burden_concentration_stacked.png`")
    lines.append("- `figures/figE14_top10_burden_vs_exact_recurrence.png`")
    lines.append("")
    lines.append("## Data Map")
    lines.append("")
    lines.append("- `data/error_priority_summary.csv`")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.analysis_root.resolve()
    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(data_dir / "error_family_repair_episodes.csv")
    identity_eps = pd.read_csv(data_dir / "error_identity_episodes.csv")

    summary = build_priority_summary(episodes=episodes, identity_eps=identity_eps)
    summary.to_csv(data_dir / "error_priority_summary.csv", index=False)

    draw_burden_pareto(summary, fig_dir / "figE11_error_burden_pareto_iterations_time.png")
    draw_priority_matrix(summary, fig_dir / "figE12_error_priority_matrix_bubble.png")
    draw_model_concentration(episodes, fig_dir / "figE13_model_burden_concentration_stacked.png")
    draw_top10_burden_and_recurrence(summary, fig_dir / "figE14_top10_burden_vs_exact_recurrence.png")

    write_conclusions(summary, root / "actionable_priority_conclusions.md")

    print(f"[ok] summary rows: {len(summary)}")
    print(f"[ok] wrote: {data_dir / 'error_priority_summary.csv'}")
    print("[ok] wrote 4 figures (figE11-figE14)")
    print(f"[ok] wrote: {root / 'actionable_priority_conclusions.md'}")


if __name__ == "__main__":
    main()

