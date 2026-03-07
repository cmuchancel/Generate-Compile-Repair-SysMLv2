#!/usr/bin/env python3
"""Create user-focused figures: common errors, quickest repairs, and hard-repair behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    base_dir = script_path.parents[1]
    repo_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=base_dir / "error_repair_insights",
        help="Folder with error_repair_insights/{data,figures}.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root (for raw iteration metrics).",
    )
    return parser.parse_args()


def parse_family_counts(raw: object) -> dict[str, int]:
    if not isinstance(raw, str):
        return {}
    try:
        obj = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in obj.items():
        try:
            vi = int(v)
        except Exception:
            continue
        if vi > 0:
            out[str(k)] = vi
    return out


def load_single_shot_counts(repo_root: Path) -> pd.DataFrame:
    iter_path = repo_root / "paper" / "results" / "data" / "iteration_level_syntax_metrics.csv"
    df = pd.read_csv(iter_path)
    df["iteration_index"] = pd.to_numeric(df["iteration_index"], errors="coerce")
    first = df[df["iteration_index"] == 1].copy()

    counts: dict[str, int] = {}
    for raw in first["error_families_json"].tolist():
        fam = parse_family_counts(raw)
        for f, c in fam.items():
            counts[f] = counts.get(f, 0) + int(c)

    out = pd.DataFrame(
        {
            "family": list(counts.keys()),
            "single_shot_error_instances": list(counts.values()),
        }
    )
    return out


def build_family_table(
    summary: pd.DataFrame,
    episodes: pd.DataFrame,
    identity_eps: pd.DataFrame,
    single_shot_counts: pd.DataFrame,
) -> pd.DataFrame:
    stats = (
        episodes.groupby("family")
        .agg(
            episodes=("family", "count"),
            mean_iter=("iterations_to_repair", "mean"),
            median_iter=("iterations_to_repair", "median"),
            p90_iter=("iterations_to_repair", lambda s: s.quantile(0.9)),
            mean_sec=("elapsed_sec_to_repair", "mean"),
            median_sec=("elapsed_sec_to_repair", "median"),
            p_repair_le1=("iterations_to_repair", lambda s: float((s <= 1).mean())),
            p_repair_le2=("iterations_to_repair", lambda s: float((s <= 2).mean())),
            long_tail_count=("iterations_to_repair", lambda s: int((s >= 3).sum())),
        )
        .reset_index()
    )

    info = summary[
        [
            "family",
            "total_error_instances",
            "prompts_affected",
            "share_total_error_instances_pct",
        ]
    ].copy()

    persisted = identity_eps[identity_eps["iterations_with_family"] >= 2].copy()
    mode = (
        persisted.groupby("family")
        .agg(
            persist_episodes=("family", "count"),
            exact_recur_rate=("has_exact_recurrence", "mean"),
            template_recur_rate=("has_template_recurrence", "mean"),
            churn_rate=("persistence_mode", lambda s: float((s == "family_only_churn").mean())),
        )
        .reset_index()
    )

    out = (
        info.merge(single_shot_counts, on="family", how="left")
        .merge(stats, on="family", how="left")
        .merge(mode, on="family", how="left")
    )
    out["single_shot_error_instances"] = out["single_shot_error_instances"].fillna(0).astype(int)
    out["long_tail_share_within_family"] = out["long_tail_count"] / out["episodes"].clip(lower=1)
    for c in ["persist_episodes", "exact_recur_rate", "template_recur_rate", "churn_rate"]:
        out[c] = out[c].fillna(0.0)

    return out.sort_values("total_error_instances", ascending=False).reset_index(drop=True)


def fig_common_dual_bar(tbl: pd.DataFrame, out: Path, top_k: int = 12) -> None:
    df = tbl.head(top_k).copy()
    y = np.arange(len(df))

    fig, (ax0, ax1, ax2) = plt.subplots(
        1,
        3,
        figsize=(19.5, 6.2),
        dpi=220,
        gridspec_kw={"width_ratios": [1.25, 1.35, 1.0]},
    )

    left_vals = df["single_shot_error_instances"].to_numpy(dtype=float)
    ax0.barh(y, left_vals, color="#F58518")
    ax0.set_yticks(y)
    ax0.set_yticklabels(df["family"].tolist(), fontsize=8)
    ax0.invert_yaxis()
    ax0.set_xlabel("Single-Shot Error Instances", fontsize=9)
    ax0.set_title(f"Single-Shot Error Volume (Top {top_k})", fontsize=10)
    ax0.grid(axis="x", alpha=0.25)
    max0 = float(left_vals.max()) if len(left_vals) else 1.0
    xpad0 = max(1.0, max0 * 0.02)
    ax0.set_xlim(0, max0 + 5.0 * xpad0)
    for i, v in enumerate(left_vals):
        ax0.text(v + xpad0, i, f"{int(v)}", va="center", fontsize=7)

    mid_vals = df["total_error_instances"].to_numpy(dtype=float)
    ax1.barh(y, mid_vals, color="#4C78A8")
    ax1.set_yticks(y)
    ax1.set_yticklabels([""] * len(y))
    ax1.invert_yaxis()
    ax1.set_xlabel("Total Error Instances", fontsize=9)
    ax1.set_title(f"Cumulative Errors Across All Iterations (Top {top_k})", fontsize=10)
    ax1.grid(axis="x", alpha=0.25)
    max1 = float(mid_vals.max()) if len(mid_vals) else 1.0
    xpad1 = max(1.0, max1 * 0.02)
    ax1.set_xlim(0, max1 + 5.0 * xpad1)
    for i, v in enumerate(mid_vals):
        ax1.text(v + xpad1, i, f"{int(v)}", va="center", fontsize=7)

    right_vals = df["prompts_affected"].to_numpy(dtype=float)
    ax2.barh(y, right_vals, color="#72B7B2")
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(y))
    ax2.invert_yaxis()
    ax2.set_xlabel("Prompts Affected", fontsize=9)
    ax2.set_title(f"Coverage Across Prompt Set (Top {top_k})", fontsize=10)
    ax2.grid(axis="x", alpha=0.25)
    max2 = float(right_vals.max()) if len(right_vals) else 1.0
    xpad2 = max(0.35, max2 * 0.03)
    ax2.set_xlim(0, max2 + 5.0 * xpad2)
    for i, v in enumerate(right_vals):
        ax2.text(v + xpad2, i, f"{int(v)}", va="center", fontsize=7)

    fig.suptitle(f"Common Errors: Single-Shot vs Total vs Coverage (Top {top_k} Families)", fontsize=11)
    fig.tight_layout(rect=[0.035, 0.02, 0.995, 0.95], w_pad=2.8)
    fig.savefig(out)
    plt.close(fig)


def fig_fast_vs_slow(tbl: pd.DataFrame, out: Path, min_episodes: int = 5, k: int = 8) -> None:
    df = tbl[tbl["episodes"] >= min_episodes].copy()
    fast = df.sort_values("median_sec", ascending=True).head(k)
    slow = df.sort_values("median_sec", ascending=False).head(k)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=220)
    y1 = np.arange(len(fast))
    y2 = np.arange(len(slow))

    ax1.barh(y1, fast["median_sec"].to_numpy(dtype=float), color="#54A24B")
    ax1.set_yticks(y1)
    ax1.set_yticklabels(fast["family"].tolist(), fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Median Repair Time (sec)", fontsize=9)
    ax1.set_title(f"Quickest-to-Repair Families (episodes >= {min_episodes})", fontsize=10)
    ax1.grid(axis="x", alpha=0.25)
    for i, (sec, p1) in enumerate(zip(fast["median_sec"].to_numpy(dtype=float), fast["p_repair_le1"].to_numpy(dtype=float))):
        ax1.text(sec + 1.0, i, f"{sec:.1f}s | <=1 iter {100*p1:.0f}%", va="center", fontsize=7)

    ax2.barh(y2, slow["median_sec"].to_numpy(dtype=float), color="#E45756")
    ax2.set_yticks(y2)
    ax2.set_yticklabels(slow["family"].tolist(), fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Median Repair Time (sec)", fontsize=9)
    ax2.set_title(f"Slowest-to-Repair Families (episodes >= {min_episodes})", fontsize=10)
    ax2.grid(axis="x", alpha=0.25)
    for i, (sec, p1) in enumerate(zip(slow["median_sec"].to_numpy(dtype=float), slow["p_repair_le1"].to_numpy(dtype=float))):
        ax2.text(sec + 1.0, i, f"{sec:.1f}s | <=1 iter {100*p1:.0f}%", va="center", fontsize=7)

    fig.suptitle("Repair Speed Rankings", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_completion_curves(episodes: pd.DataFrame, tbl: pd.DataFrame, out: Path, top_k: int = 5) -> None:
    top_fams = tbl.sort_values("total_error_instances", ascending=False)["family"].head(top_k).tolist()
    max_k = int(max(1, episodes["iterations_to_repair"].max()))
    ks = np.arange(1, max_k + 1)

    fig, ax = plt.subplots(figsize=(9.8, 6.3), dpi=220)
    plotted_curves: list[np.ndarray] = []
    for fam in top_fams:
        s = episodes.loc[episodes["family"] == fam, "iterations_to_repair"].dropna().to_numpy(dtype=float)
        if len(s) == 0:
            continue
        cdf = np.array([(s <= k).mean() for k in ks], dtype=float)
        plotted_curves.append(cdf)
        ax.plot(ks, cdf, marker="o", markersize=4.0, linewidth=2.1, label=f"{fam} (n={len(s)})")

    ax.set_xlabel("Additional Iterations Allowed", fontsize=9)
    ax.set_ylabel("Share Repaired by This Iteration", fontsize=9)
    ax.set_xticks(ks)
    if plotted_curves:
        vals = np.concatenate(plotted_curves)
        y_min = max(0.0, float(vals.min()) - 0.08)
        y_max = min(1.05, float(vals.max()) + 0.03)
        if y_max - y_min < 0.2:
            y_min = max(0.0, y_max - 0.25)
        if y_max <= 1.0:
            y_max = 1.03
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(0.0, 1.03)
    ax.set_title(f"How Quickly Top-{top_k} Common Error Families Get Repaired", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_unrepaired_and_modes(identity_eps: pd.DataFrame, out: Path, top_k: int = 5) -> None:
    persisted = identity_eps[identity_eps["iterations_with_family"] >= 2].copy()
    mode_share = persisted["persistence_mode"].value_counts(normalize=True).reindex(
        ["same_exact_error_recurs", "same_error_template_recurs", "family_only_churn"], fill_value=0.0
    )
    mode_labels = ["Same exact recurs", "Same template recurs", "Family-only churn"]
    mode_colors = ["#2E7D32", "#F9A825", "#C62828"]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14.2, 5.2),
        dpi=220,
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )

    # Left panel: overall mode mix (same plot style user liked).
    vals = 100.0 * mode_share.to_numpy(dtype=float)
    x = np.arange(len(vals))
    bars = ax1.bar(x, vals, color=mode_colors, width=0.62)
    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_labels, rotation=12, ha="right", fontsize=8)
    ax1.set_ylim(0, max(100.0, float(vals.max()) + 8.0))
    ax1.set_ylabel("Share of Persistent Episodes (%)", fontsize=9)
    ax1.set_title("If Not Fixed Immediately, What Happens Next?", fontsize=10)
    ax1.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    # Right panel: top-K families, split by same green/yellow/red modes.
    mode_order = ["same_exact_error_recurs", "same_error_template_recurs", "family_only_churn"]
    top_fams = persisted["family"].value_counts().head(top_k).index.tolist()
    fam_mode = (
        persisted[persisted["family"].isin(top_fams)]
        .groupby(["family", "persistence_mode"])
        .size()
        .rename("n")
        .reset_index()
    )
    fam_tot = fam_mode.groupby("family")["n"].sum().rename("total").reset_index()
    fam_mode = fam_mode.merge(fam_tot, on="family", how="left")
    fam_mode["share_pct"] = 100.0 * fam_mode["n"] / fam_mode["total"].clip(lower=1)
    piv = (
        fam_mode.pivot(index="family", columns="persistence_mode", values="share_pct")
        .fillna(0.0)
        .reindex(index=top_fams)
        .reindex(columns=mode_order, fill_value=0.0)
    )

    y = np.arange(len(top_fams))
    left = np.zeros(len(top_fams), dtype=float)
    legend_handles = []
    for mode, color in zip(mode_order, mode_colors):
        v = piv[mode].to_numpy(dtype=float)
        h = ax2.barh(y, v, left=left, color=color, edgecolor="white", linewidth=0.6)
        legend_handles.append(h[0])
        left += v

    ax2.set_yticks(y)
    ax2.set_yticklabels(top_fams, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Share within Family's Persistent Episodes (%)", fontsize=9)
    ax2.set_title(f"Mode Breakdown by Error Family (Top {top_k})", fontsize=10)
    ax2.grid(axis="x", alpha=0.25)
    for i, total in enumerate(fam_tot.set_index("family").loc[top_fams, "total"].to_numpy(dtype=int)):
        ax2.text(101.0, i, f"n={int(total)}", va="center", fontsize=7)

    ax2.legend(
        legend_handles,
        ["Same exact recurs", "Same template recurs", "Family-only churn"],
        frameon=False,
        fontsize=7,
        loc="lower right",
    )

    fig.suptitle("Hard-Repair Behavior: Overall vs Top Error Families", fontsize=11)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95], w_pad=2.2)
    fig.savefig(out)
    plt.close(fig)


def fig_long_tail_by_family(tbl: pd.DataFrame, out: Path, min_episodes: int = 5, top_k: int = 10) -> None:
    df = tbl[tbl["episodes"] >= min_episodes].copy()
    df = df.sort_values(["long_tail_count", "total_error_instances"], ascending=[False, False]).head(top_k)
    y = np.arange(len(df))
    vals = 100.0 * df["long_tail_share_within_family"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=220)
    ax.barh(y, vals, color="#9467BD")
    ax.set_yticks(y)
    ax.set_yticklabels(df["family"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Share of Episodes Requiring >=3 Additional Iterations (%)", fontsize=9)
    ax.set_title("Hard-Repair Rate by Error Family", fontsize=11)
    ax.grid(axis="x", alpha=0.25)

    for i, (pct, n, total) in enumerate(
        zip(vals, df["long_tail_count"].to_numpy(dtype=int), df["episodes"].to_numpy(dtype=int))
    ):
        ax.text(pct + 0.7, i, f"{pct:.1f}% ({n}/{total})", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_user_facing_takeaways(
    tbl: pd.DataFrame,
    episodes: pd.DataFrame,
    identity_eps: pd.DataFrame,
    out: Path,
) -> None:
    top_common = tbl.sort_values("total_error_instances", ascending=False).head(5)
    quick = tbl[tbl["episodes"] >= 5].sort_values("median_sec").head(5)
    slow = tbl[tbl["episodes"] >= 5].sort_values("median_sec", ascending=False).head(5)

    unresolved = int((~episodes["resolved"]).sum())
    total = int(len(episodes))
    persisted = identity_eps[identity_eps["iterations_with_family"] >= 2].copy()
    mode = persisted["persistence_mode"].value_counts(normalize=True).to_dict()

    lines: List[str] = []
    lines.append("# Common vs Quick vs Hard Repair: What To Take Away")
    lines.append("")
    lines.append("This section answers three direct questions:")
    lines.append("1. Which errors are common?")
    lines.append("2. Which errors are repaired quickest?")
    lines.append("3. If not repaired immediately, what does repair behavior look like?")
    lines.append("")
    lines.append("## 1) Common Errors")
    lines.append("")
    for _, r in top_common.iterrows():
        lines.append(
            f"- `{r['family']}`: {int(r['single_shot_error_instances'])} single-shot instances, {int(r['total_error_instances'])} total instances, across {int(r['prompts_affected'])} prompts."
        )
    lines.append("")
    lines.append("## 2) Quickest Repairs (episodes >= 5)")
    lines.append("")
    for _, r in quick.iterrows():
        lines.append(
            f"- `{r['family']}`: median {float(r['median_sec']):.1f}s, repaired in <=1 iteration {100.0*float(r['p_repair_le1']):.0f}% of episodes."
        )
    lines.append("")
    lines.append("## 3) Hard/Not-Immediate Repairs")
    lines.append("")
    lines.append(f"- Unrepaired by campaign end: **{unresolved}/{total} episodes**.")
    lines.append("- Because unresolved-by-end is zero here, hard repair is best viewed as multi-iteration persistence.")
    lines.append(
        f"- Among persistent episodes: same exact diagnostic recurs in **{100.0*mode.get('same_exact_error_recurs',0.0):.1f}%**, template recurrence in **{100.0*mode.get('same_error_template_recurs',0.0):.1f}%**, family-only churn in **{100.0*mode.get('family_only_churn',0.0):.1f}%**."
    )
    lines.append("")
    lines.append("Slowest repairs by median time (episodes >= 5):")
    for _, r in slow.iterrows():
        lines.append(
            f"- `{r['family']}`: median {float(r['median_sec']):.1f}s, mean iterations {float(r['mean_iter']):.2f}."
        )
    lines.append("")
    lines.append("## Figure Map")
    lines.append("")
    lines.append("- `figures/figE15_common_errors_volume_and_coverage.png`")
    lines.append("- `figures/figE16_repair_speed_fast_vs_slow.png`")
    lines.append("- `figures/figE17_repair_completion_curves_top_common.png`")
    lines.append("- `figures/figE18_unrepaired_status_and_persistence_behavior.png`")
    lines.append("- `figures/figE19_hard_repair_rate_by_family.png`")
    lines.append("")
    lines.append("## Data Map")
    lines.append("")
    lines.append("- `data/error_common_fast_hard_summary.csv`")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.analysis_root.resolve()
    repo_root = args.repo_root.resolve()
    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(data_dir / "error_family_summary.csv")
    episodes = pd.read_csv(data_dir / "error_family_repair_episodes.csv")
    identity_eps = pd.read_csv(data_dir / "error_identity_episodes.csv")
    single_shot_counts = load_single_shot_counts(repo_root)

    tbl = build_family_table(summary, episodes, identity_eps, single_shot_counts)
    tbl.to_csv(data_dir / "error_common_fast_hard_summary.csv", index=False)

    fig_common_dual_bar(tbl, fig_dir / "figE15_common_errors_volume_and_coverage.png")
    fig_fast_vs_slow(tbl, fig_dir / "figE16_repair_speed_fast_vs_slow.png")
    fig_completion_curves(episodes, tbl, fig_dir / "figE17_repair_completion_curves_top_common.png")
    fig_unrepaired_and_modes(identity_eps, fig_dir / "figE18_unrepaired_status_and_persistence_behavior.png")
    fig_long_tail_by_family(tbl, fig_dir / "figE19_hard_repair_rate_by_family.png")

    write_user_facing_takeaways(
        tbl=tbl,
        episodes=episodes,
        identity_eps=identity_eps,
        out=root / "common_fast_hard_takeaways.md",
    )

    print(f"[ok] summary rows: {len(tbl)}")
    print(f"[ok] wrote: {data_dir / 'error_common_fast_hard_summary.csv'}")
    print("[ok] wrote 5 figures (figE15-figE19)")
    print(f"[ok] wrote: {root / 'common_fast_hard_takeaways.md'}")


if __name__ == "__main__":
    main()
