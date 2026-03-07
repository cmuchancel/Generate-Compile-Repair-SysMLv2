#!/usr/bin/env python3
"""Generate subcategory analysis (domain/grammar/difficulty) across 4 LLMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors


MODEL_LABELS = {
    "anthropic": "Anthropic Claude Sonnet 4.6",
    "deepseek_reasoner": "DeepSeek Reasoner",
    "mistral_large": "Mistral Large",
    "openai": "OpenAI",
}

MODEL_ORDER = [
    "OpenAI",
    "Anthropic Claude Sonnet 4.6",
    "DeepSeek Reasoner",
    "Mistral Large",
]


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    base_dir = script_path.parents[1]  # repsone to taylans comments/
    repo_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=base_dir,
        help="Root output directory for data/figures/writeup.",
    )
    parser.add_argument(
        "--top-grammar-k",
        type=int,
        default=15,
        help="How many grammar labels to show in grammar heatmaps.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def difficulty_bucket(lines: int) -> int:
    # Matches evaluation_scripts/get_difficult_metrics.py bucket logic.
    if lines < 30:
        return 1
    if lines < 60:
        return 2
    if lines < 90:
        return 3
    if lines < 120:
        return 4
    return 5


def make_label_frame(dataset_path: Path) -> pd.DataFrame:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    for idx, sample in enumerate(data, start=1):
        design = str(sample.get("design", ""))
        line_count = len(design.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
        rows.append(
            {
                "prompt_id": idx,
                "domain": str(sample.get("domain", "")),
                "grammar": str(sample.get("grammar", "")),
                "reference_sysml_line_count": line_count,
                "difficulty": difficulty_bucket(line_count),
            }
        )
    return pd.DataFrame(rows)


def make_merged_frame(prompt_metrics_path: Path, labels: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(prompt_metrics_path)
    merged = df.merge(labels, on="prompt_id", how="left", validate="many_to_one")

    merged["model_label"] = merged["provider"].map(MODEL_LABELS).fillna(merged["model"])
    merged["iterations_to_success"] = pd.to_numeric(merged["iterations_to_success"], errors="coerce")
    merged["iterations_run"] = pd.to_numeric(merged["iterations_run"], errors="coerce")

    bool_cols = [
        "first_shot_pass",
        "eventual_pass",
        "unresolved_within_cap",
        "first_failed_then_recovered",
    ]
    for col in bool_cols:
        merged[col] = merged[col].astype(bool)
    return merged


def summarize_by_category(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["model_label", category_col], dropna=False)
        .agg(
            n_prompts=("prompt_id", "count"),
            first_shot_pass_rate=("first_shot_pass", "mean"),
            eventual_pass_rate=("eventual_pass", "mean"),
            unresolved_rate=("unresolved_within_cap", "mean"),
            first_failed_then_recovered_rate=("first_failed_then_recovered", "mean"),
            mean_first_iteration_error_count=("first_iteration_error_count", "mean"),
            mean_total_error_count_across_iterations=(
                "total_error_count_across_iterations",
                "mean",
            ),
            mean_iterations_run=("iterations_run", "mean"),
            mean_iterations_to_success=("iterations_to_success", "mean"),
            median_iterations_to_success=("iterations_to_success", "median"),
        )
        .reset_index()
    )

    for rate_col in [
        "first_shot_pass_rate",
        "eventual_pass_rate",
        "unresolved_rate",
        "first_failed_then_recovered_rate",
    ]:
        grouped[rate_col] = grouped[rate_col] * 100.0
    return grouped


def select_top_categories(df: pd.DataFrame, category_col: str, top_k: int) -> List[str]:
    counts = (
        df[[category_col, "prompt_id"]]
        .drop_duplicates()
        .groupby(category_col, dropna=False)["prompt_id"]
        .count()
        .sort_values(ascending=False)
    )
    return [str(x) for x in counts.head(top_k).index.tolist()]


def pivot_for_plot(
    summary_df: pd.DataFrame,
    category_col: str,
    value_col: str,
    category_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    pivot = summary_df.pivot(index="model_label", columns=category_col, values=value_col)
    model_order = [m for m in MODEL_ORDER if m in pivot.index] + [m for m in pivot.index if m not in MODEL_ORDER]
    pivot = pivot.reindex(model_order)
    if category_order is not None:
        pivot = pivot.reindex(columns=list(category_order))
    return pivot


def draw_heatmap(
    pivot: pd.DataFrame,
    output_path: Path,
    title: str,
    cbar_label: str,
    value_fmt: str = ".1f",
    fixed_range: Optional[Tuple[float, float]] = None,
    cmap: str = "viridis",
) -> None:
    arr = pivot.to_numpy(dtype=float)
    rows, cols = arr.shape
    fig_w = max(8.0, 0.75 * cols + 2.5)
    fig_h = max(3.0, 0.65 * rows + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

    if fixed_range is None:
        valid = arr[~np.isnan(arr)]
        if valid.size:
            vmin, vmax = float(valid.min()), float(valid.max())
        else:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = fixed_range

    cmap_obj = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(arr, cmap=cmap_obj, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(rows))
    ax.set_yticklabels([str(r) for r in pivot.index], fontsize=8)

    # Annotate cells.
    for i in range(rows):
        for j in range(cols):
            val = arr[i, j]
            text = "NA" if np.isnan(val) else format(val, value_fmt)
            if np.isnan(val):
                color = "black"
            else:
                r, g, b, _ = cmap_obj(norm(val))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                color = "black" if luminance > 0.57 else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def draw_grouped_bar(
    summary_df: pd.DataFrame,
    category_col: str,
    value_col: str,
    output_path: Path,
    title: str,
    y_label: str,
    category_order: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    x = np.arange(len(category_order))
    models = [m for m in MODEL_ORDER if m in summary_df["model_label"].unique()]
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(models))

    for idx, model in enumerate(models):
        sub = summary_df[summary_df["model_label"] == model]
        values = []
        for cat in category_order:
            row = sub[sub[category_col].astype(str) == str(cat)]
            if row.empty:
                values.append(np.nan)
            else:
                values.append(float(row.iloc[0][value_col]))
        ax.bar(x + offsets[idx], values, width=width, label=model)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(category_col.replace("_", " ").title(), fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in category_order], fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def draw_grammar_iterations_combo(
    grammar_summary: pd.DataFrame,
    output_path: Path,
    top_grammar: Sequence[str],
    top_k: int,
) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(14, 8.8), dpi=220, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )

    # Top panel: model x grammar heatmap for mean iterations-to-success.
    pivot = pivot_for_plot(
        grammar_summary, "grammar", "mean_iterations_to_success", category_order=top_grammar
    )
    arr = pivot.to_numpy(dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size:
        vmin, vmax = float(valid.min()), float(valid.max())
    else:
        vmin, vmax = 0.0, 1.0

    # Lighter lows than viridis for better label readability.
    cmap_obj = plt.get_cmap("YlGnBu")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax_top.imshow(arr, cmap=cmap_obj, aspect="auto", vmin=vmin, vmax=vmax)
    ax_top.set_title(
        f"Mean Iterations-to-Success by Top-{top_k} Grammar Labels and Model",
        fontsize=11,
        pad=10,
    )
    ax_top.set_xticks(np.arange(arr.shape[1]))
    ax_top.set_xticklabels([str(c) for c in pivot.columns], rotation=35, ha="right", fontsize=8)
    ax_top.set_yticks(np.arange(arr.shape[0]))
    ax_top.set_yticklabels([str(r) for r in pivot.index], fontsize=8)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            text = "NA" if np.isnan(val) else format(val, ".2f")
            if np.isnan(val):
                color = "black"
            else:
                r, g, b, _ = cmap_obj(norm(val))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                color = "black" if luminance > 0.57 else "white"
            ax_top.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax_top, fraction=0.022, pad=0.02)
    cbar.set_label("Iterations", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Bottom panel: all-model average across the same grammar labels.
    macro = (
        grammar_summary.groupby("grammar", dropna=False)["mean_iterations_to_success"]
        .mean()
        .to_dict()
    )
    x = np.arange(len(top_grammar))
    y = np.array([float(macro.get(str(g), np.nan)) for g in top_grammar], dtype=float)
    bars = ax_bottom.bar(x, y, color="#4C78A8")
    ax_bottom.set_title(
        "All-Model Average Iterations-to-Success (Same Top-15 Grammar Labels)",
        fontsize=10,
    )
    ax_bottom.set_ylabel("Mean Iterations", fontsize=9)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([str(g) for g in top_grammar], rotation=35, ha="right", fontsize=8)
    ax_bottom.grid(axis="y", alpha=0.25)

    for b, val in zip(bars, y):
        if np.isnan(val):
            continue
        ax_bottom.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def draw_domain_iterations_combo(
    domain_summary: pd.DataFrame,
    merged_df: pd.DataFrame,
    output_path: Path,
    domain_order: Sequence[str],
) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(14, 8.4), dpi=220, gridspec_kw={"height_ratios": [2.1, 1.0]}
    )

    # Top panel: model x domain heatmap.
    pivot = pivot_for_plot(
        domain_summary, "domain", "mean_iterations_to_success", category_order=domain_order
    )
    arr = pivot.to_numpy(dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size:
        vmin, vmax = float(valid.min()), float(valid.max())
    else:
        vmin, vmax = 0.0, 1.0

    cmap_obj = plt.get_cmap("YlGnBu")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax_top.imshow(arr, cmap=cmap_obj, aspect="auto", vmin=vmin, vmax=vmax)
    ax_top.set_title("Mean Iterations-to-Success by Domain and Model", fontsize=11, pad=10)
    ax_top.set_xticks(np.arange(arr.shape[1]))
    ax_top.set_xticklabels([str(c) for c in pivot.columns], rotation=35, ha="right", fontsize=8)
    ax_top.set_yticks(np.arange(arr.shape[0]))
    ax_top.set_yticklabels([str(r) for r in pivot.index], fontsize=8)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            text = "NA" if np.isnan(val) else format(val, ".2f")
            if np.isnan(val):
                color = "black"
            else:
                r, g, b, _ = cmap_obj(norm(val))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                color = "black" if luminance > 0.57 else "white"
            ax_top.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax_top, fraction=0.022, pad=0.02)
    cbar.set_label("Iterations", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Bottom panel: all-model average across domains using the same bar style as grammar panel (+/- SE).
    merged = merged_df.copy()
    merged = merged.dropna(subset=["domain", "iterations_to_success"])
    agg = (
        merged.groupby("domain", dropna=False)["iterations_to_success"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["se"] = agg["std"] / np.sqrt(agg["count"])
    agg = agg.set_index("domain").reindex([str(d) for d in domain_order]).reset_index()
    agg = agg.dropna(subset=["mean"]).copy()

    x_pos = np.arange(len(agg))
    means = agg["mean"].to_numpy(dtype=float)
    ses = agg["se"].to_numpy(dtype=float)

    bars = ax_bottom.bar(
        x_pos,
        means,
        color="#4C78A8",
        yerr=ses,
        capsize=3,
        error_kw={"elinewidth": 1.2, "ecolor": "#1f1f1f", "capthick": 1.2},
    )
    ax_bottom.set_title("All-Model Average Iterations-to-Success by Domain (mean ± SE)", fontsize=10)
    ax_bottom.set_ylabel("Mean Iterations", fontsize=9)
    ax_bottom.set_xticks(x_pos)
    ax_bottom.set_xticklabels(agg["domain"].astype(str).tolist(), rotation=35, ha="right", fontsize=8)
    ax_bottom.grid(axis="y", alpha=0.25)

    max_y = float(np.nanmax(means + ses)) if len(means) else 1.0
    min_y = float(np.nanmin(means - ses)) if len(means) else 0.0
    span = max(max_y - min_y, 0.2)
    label_offset = 0.04 * span
    ax_bottom.set_ylim(max(0.0, min_y - 0.08 * span), max_y + 0.28 * span)

    for b, m, se in zip(bars, means, ses):
        ax_bottom.text(
            b.get_x() + b.get_width() / 2.0,
            m + se + label_offset,
            f"{m:.2f}",
            va="bottom",
            ha="center",
            fontsize=7.5,
            color="#1f1f1f",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.75, edgecolor="none"),
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def draw_difficulty_iterations_combo(
    difficulty_summary: pd.DataFrame,
    merged_df: pd.DataFrame,
    output_path: Path,
    category_order: Sequence[str],
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), dpi=220)

    # Left panel: keep original grouped bars by model.
    x = np.arange(len(category_order))
    models = [m for m in MODEL_ORDER if m in difficulty_summary["model_label"].unique()]
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(models))
    for idx, model in enumerate(models):
        sub = difficulty_summary[difficulty_summary["model_label"] == model]
        values = []
        for cat in category_order:
            row = sub[sub["difficulty"].astype(str) == str(cat)]
            if row.empty:
                values.append(np.nan)
            else:
                values.append(float(row.iloc[0]["mean_iterations_to_success"]))
        ax1.bar(x + offsets[idx], values, width=width, label=model)

    ax1.set_title("Mean Iterations-to-Success by Difficulty and Model", fontsize=11)
    ax1.set_xlabel("Difficulty Bucket", fontsize=9)
    ax1.set_ylabel("Mean Iterations-to-Success", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in category_order], fontsize=8)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, fontsize=8)

    # Right panel: all-model average with trendline + R^2.
    merged = merged_df.copy()
    merged = merged.dropna(subset=["difficulty", "iterations_to_success"])
    merged["difficulty"] = merged["difficulty"].astype(int)
    agg = (
        merged.groupby("difficulty", dropna=False)["iterations_to_success"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("difficulty")
    )
    agg["se"] = agg["std"] / np.sqrt(agg["count"])

    x2 = agg["difficulty"].to_numpy(dtype=float)
    y2 = agg["mean"].to_numpy(dtype=float)
    yerr = agg["se"].to_numpy(dtype=float)

    ax2.errorbar(
        x2,
        y2,
        yerr=yerr,
        fmt="o-",
        linewidth=2,
        markersize=5,
        capsize=4,
        label="All-model mean +/- SE",
    )

    # Linear trendline and R^2.
    if len(x2) >= 2:
        slope, intercept = np.polyfit(x2, y2, 1)
        y_hat = slope * x2 + intercept
        ss_res = float(np.sum((y2 - y_hat) ** 2))
        ss_tot = float(np.sum((y2 - np.mean(y2)) ** 2))
        r2 = float("nan") if ss_tot == 0 else 1.0 - (ss_res / ss_tot)
        ax2.plot(x2, y_hat, "--", linewidth=1.8, label="Linear trend")
        note = f"y = {slope:.3f}x + {intercept:.3f}\n$R^2$ = {r2:.3f}"
    else:
        note = "Insufficient points for trendline"

    ax2.set_title("All-Model Average by Difficulty", fontsize=11)
    ax2.set_xlabel("Difficulty Bucket", fontsize=9)
    ax2.set_ylabel("Mean Iterations-to-Success", fontsize=9)
    ax2.set_xticks([int(v) for v in category_order])
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    ax2.text(
        0.03,
        0.97,
        note,
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="0.7"),
    )

    fig.suptitle("Difficulty vs Iterations-to-Success: Model-wise and Combined View", fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def draw_count_panels(
    labels: pd.DataFrame,
    top_grammar: Sequence[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=200)

    domain_counts = labels["domain"].value_counts().sort_values(ascending=False)
    axes[0].bar(np.arange(len(domain_counts)), domain_counts.values)
    axes[0].set_title("Domain label counts", fontsize=10)
    axes[0].set_xticks(np.arange(len(domain_counts)))
    axes[0].set_xticklabels(domain_counts.index, rotation=75, ha="right", fontsize=7)
    axes[0].set_ylabel("Prompt count", fontsize=8)

    grammar_counts = labels["grammar"].value_counts().reindex(top_grammar)
    axes[1].bar(np.arange(len(grammar_counts)), grammar_counts.values)
    axes[1].set_title("Top grammar label counts", fontsize=10)
    axes[1].set_xticks(np.arange(len(grammar_counts)))
    axes[1].set_xticklabels(grammar_counts.index, rotation=75, ha="right", fontsize=7)

    diff_counts = labels["difficulty"].value_counts().sort_index()
    axes[2].bar(diff_counts.index.astype(str), diff_counts.values)
    axes[2].set_title("Difficulty bucket counts", fontsize=10)
    axes[2].set_xlabel("Difficulty bucket", fontsize=8)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def safe_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def make_findings_md(
    domain_summary: pd.DataFrame,
    grammar_summary: pd.DataFrame,
    difficulty_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    # Aggregate over category by averaging model-level metrics (macro over models).
    dom_macro = (
        domain_summary.groupby("domain", dropna=False)
        .agg(
            first_shot_pass_rate=("first_shot_pass_rate", "mean"),
            mean_iterations_to_success=("mean_iterations_to_success", "mean"),
            mean_first_iteration_error_count=("mean_first_iteration_error_count", "mean"),
        )
        .reset_index()
        .sort_values("first_shot_pass_rate", ascending=False)
    )
    gram_macro = (
        grammar_summary.groupby("grammar", dropna=False)
        .agg(
            first_shot_pass_rate=("first_shot_pass_rate", "mean"),
            mean_iterations_to_success=("mean_iterations_to_success", "mean"),
            mean_first_iteration_error_count=("mean_first_iteration_error_count", "mean"),
        )
        .reset_index()
        .sort_values("first_shot_pass_rate", ascending=False)
    )
    diff_macro = (
        difficulty_summary.groupby("difficulty", dropna=False)
        .agg(
            first_shot_pass_rate=("first_shot_pass_rate", "mean"),
            mean_iterations_to_success=("mean_iterations_to_success", "mean"),
            mean_first_iteration_error_count=("mean_first_iteration_error_count", "mean"),
        )
        .reset_index()
        .sort_values("difficulty")
    )

    top_dom = dom_macro.head(3)
    low_dom = dom_macro.tail(3)
    top_gram = gram_macro.head(3)
    low_gram = gram_macro.tail(3)

    lines: List[str] = []
    lines.append("# Response To Taylan's Comments: Subcategory Analysis Draft")
    lines.append("")
    lines.append("This note adds a label-conditioned analysis using SysMBench metadata:")
    lines.append("- Domain label (`domain`)")
    lines.append("- Key grammar label (`grammar`)")
    lines.append("- Difficulty bucket (computed from reference SysML length, buckets 1-5)")
    lines.append("")
    lines.append("Data source for the 4-model comparison:")
    lines.append("- `paper/results/data/prompt_level_syntax_metrics.csv` (151 prompts x 4 models = 604 rows)")
    lines.append("")
    lines.append("## Suggested Results Text (Short)")
    lines.append("")
    lines.append(
        "Across all four LLM pipelines, first-shot behavior and repair burden vary systematically by domain, key grammar label, and difficulty."
    )
    lines.append(
        "Because eventual compile pass is near-ceiling under iterative repair, first-shot pass, iterations-to-success, and first-iteration error burden are the most informative differentiators."
    )
    lines.append(
        "Domain-conditioned and grammar-conditioned results are heterogeneous: some labels compile on first shot consistently, while others require materially more repair iterations."
    )
    lines.append(
        "Difficulty buckets (proxied by reference SysML line-count complexity) also show non-uniform first-shot pass and error burden."
    )
    lines.append(
        "These findings support reporting subcategory-stratified performance rather than only aggregate eventual compile rates."
    )
    lines.append("")
    lines.append("## Quick Numerical Highlights (Macro-averaged across models)")
    lines.append("")
    lines.append("Top domains by first-shot pass rate:")
    for _, r in top_dom.iterrows():
        lines.append(
            f"- {r['domain']}: first-shot {safe_float(r['first_shot_pass_rate']):.1f}% | mean iters-to-success {safe_float(r['mean_iterations_to_success']):.2f} | mean first-iter errors {safe_float(r['mean_first_iteration_error_count']):.2f}"
        )
    lines.append("")
    lines.append("Lowest domains by first-shot pass rate:")
    for _, r in low_dom.iterrows():
        lines.append(
            f"- {r['domain']}: first-shot {safe_float(r['first_shot_pass_rate']):.1f}% | mean iters-to-success {safe_float(r['mean_iterations_to_success']):.2f} | mean first-iter errors {safe_float(r['mean_first_iteration_error_count']):.2f}"
        )
    lines.append("")
    lines.append("Top grammar labels by first-shot pass rate:")
    for _, r in top_gram.iterrows():
        lines.append(
            f"- {r['grammar']}: first-shot {safe_float(r['first_shot_pass_rate']):.1f}% | mean iters-to-success {safe_float(r['mean_iterations_to_success']):.2f} | mean first-iter errors {safe_float(r['mean_first_iteration_error_count']):.2f}"
        )
    lines.append("")
    lines.append("Lowest grammar labels by first-shot pass rate:")
    for _, r in low_gram.iterrows():
        lines.append(
            f"- {r['grammar']}: first-shot {safe_float(r['first_shot_pass_rate']):.1f}% | mean iters-to-success {safe_float(r['mean_iterations_to_success']):.2f} | mean first-iter errors {safe_float(r['mean_first_iteration_error_count']):.2f}"
        )
    lines.append("")
    lines.append("Difficulty trend (macro-averaged across models):")
    for _, r in diff_macro.iterrows():
        lines.append(
            f"- Difficulty {int(r['difficulty'])}: first-shot {safe_float(r['first_shot_pass_rate']):.1f}% | mean iterations-to-success {safe_float(r['mean_iterations_to_success']):.2f} | mean first-iter errors {safe_float(r['mean_first_iteration_error_count']):.2f}"
        )
    lines.append("")
    lines.append("## Figure/Data Map")
    lines.append("")
    lines.append("- `figures/fig01_domain_eventual_pass_heatmap.png`")
    lines.append("- `figures/fig02_domain_first_shot_pass_heatmap.png`")
    lines.append("- `figures/fig03_domain_iterations_to_success_heatmap.png` (combo: heatmap + all-model average bar chart with ±SE)")
    lines.append("- `figures/fig04_grammar_topK_eventual_pass_heatmap.png`")
    lines.append("- `figures/fig05_grammar_topK_first_shot_pass_heatmap.png`")
    lines.append("- `figures/fig06_grammar_topK_iterations_to_success_heatmap.png` (combo: heatmap + all-model average bar chart)")
    lines.append("- `figures/fig07_difficulty_eventual_pass_grouped_bar.png`")
    lines.append("- `figures/fig08_difficulty_first_shot_pass_grouped_bar.png`")
    lines.append("- `figures/fig09_difficulty_mean_iterations_to_success_grouped_bar.png` (combo: grouped bars + all-model trendline with $R^2$)")
    lines.append("- `figures/fig10_label_sample_size_panels.png`")
    lines.append("- `figures/fig11_domain_first_iteration_error_heatmap.png`")
    lines.append("- `figures/fig12_grammar_topK_first_iteration_error_heatmap.png`")
    lines.append("- `figures/fig13_difficulty_first_iteration_error_grouped_bar.png`")
    lines.append("- `data/merged_prompt_metrics_with_labels.csv`")
    lines.append("- `data/summary_domain_by_model.csv`")
    lines.append("- `data/summary_grammar_by_model.csv`")
    lines.append("- `data/summary_difficulty_by_model.csv`")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()

    data_dir = output_root / "data"
    fig_dir = output_root / "figures"
    ensure_dir(data_dir)
    ensure_dir(fig_dir)

    prompt_metrics_path = repo_root / "paper" / "results" / "data" / "prompt_level_syntax_metrics.csv"
    dataset_path = repo_root / "sysmbench_original_upstream" / "dataset" / "sysml" / "dataset.json"

    labels = make_label_frame(dataset_path)
    merged = make_merged_frame(prompt_metrics_path, labels)
    merged.to_csv(data_dir / "merged_prompt_metrics_with_labels.csv", index=False)

    domain_summary = summarize_by_category(merged, "domain").sort_values(["model_label", "domain"])
    grammar_summary = summarize_by_category(merged, "grammar").sort_values(["model_label", "grammar"])
    difficulty_summary = summarize_by_category(merged, "difficulty").sort_values(["model_label", "difficulty"])

    domain_summary.to_csv(data_dir / "summary_domain_by_model.csv", index=False)
    grammar_summary.to_csv(data_dir / "summary_grammar_by_model.csv", index=False)
    difficulty_summary.to_csv(data_dir / "summary_difficulty_by_model.csv", index=False)

    # Choose top-K grammar labels for figures.
    top_grammar = select_top_categories(merged, "grammar", args.top_grammar_k)

    # Heatmaps: domain.
    dom_eventual = pivot_for_plot(domain_summary, "domain", "eventual_pass_rate")
    dom_first = pivot_for_plot(domain_summary, "domain", "first_shot_pass_rate")
    domain_order = (
        merged[["domain", "prompt_id"]]
        .drop_duplicates()
        .groupby("domain", dropna=False)["prompt_id"]
        .count()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    draw_heatmap(
        dom_eventual,
        fig_dir / "fig01_domain_eventual_pass_heatmap.png",
        "Eventual Compile Pass Rate by Domain and Model",
        "Pass Rate (%)",
        fixed_range=(0, 100),
    )
    draw_heatmap(
        dom_first,
        fig_dir / "fig02_domain_first_shot_pass_heatmap.png",
        "First-Shot Compile Pass Rate by Domain and Model",
        "Pass Rate (%)",
        fixed_range=(0, 100),
    )
    draw_domain_iterations_combo(
        domain_summary=domain_summary,
        merged_df=merged,
        output_path=fig_dir / "fig03_domain_iterations_to_success_heatmap.png",
        domain_order=domain_order,
    )

    # Heatmaps: grammar (top-K).
    gram_eventual = pivot_for_plot(
        grammar_summary, "grammar", "eventual_pass_rate", category_order=top_grammar
    )
    gram_first = pivot_for_plot(
        grammar_summary, "grammar", "first_shot_pass_rate", category_order=top_grammar
    )
    draw_heatmap(
        gram_eventual,
        fig_dir / "fig04_grammar_topK_eventual_pass_heatmap.png",
        f"Eventual Compile Pass Rate by Top-{args.top_grammar_k} Grammar Labels and Model",
        "Pass Rate (%)",
        fixed_range=(0, 100),
    )
    draw_heatmap(
        gram_first,
        fig_dir / "fig05_grammar_topK_first_shot_pass_heatmap.png",
        f"First-Shot Compile Pass Rate by Top-{args.top_grammar_k} Grammar Labels and Model",
        "Pass Rate (%)",
        fixed_range=(0, 100),
    )
    draw_grammar_iterations_combo(
        grammar_summary=grammar_summary,
        output_path=fig_dir / "fig06_grammar_topK_iterations_to_success_heatmap.png",
        top_grammar=top_grammar,
        top_k=args.top_grammar_k,
    )

    # Difficulty grouped bars.
    diff_order = ["1", "2", "3", "4", "5"]
    difficulty_for_plot = difficulty_summary.copy()
    difficulty_for_plot["difficulty"] = difficulty_for_plot["difficulty"].astype(str)
    draw_grouped_bar(
        difficulty_for_plot,
        "difficulty",
        "eventual_pass_rate",
        fig_dir / "fig07_difficulty_eventual_pass_grouped_bar.png",
        "Eventual Compile Pass Rate by Difficulty and Model",
        "Pass Rate (%)",
        diff_order,
    )
    draw_grouped_bar(
        difficulty_for_plot,
        "difficulty",
        "first_shot_pass_rate",
        fig_dir / "fig08_difficulty_first_shot_pass_grouped_bar.png",
        "First-Shot Compile Pass Rate by Difficulty and Model",
        "Pass Rate (%)",
        diff_order,
    )
    draw_difficulty_iterations_combo(
        difficulty_summary=difficulty_for_plot,
        merged_df=merged,
        output_path=fig_dir / "fig09_difficulty_mean_iterations_to_success_grouped_bar.png",
        category_order=diff_order,
    )

    draw_count_panels(
        labels,
        top_grammar=top_grammar,
        output_path=fig_dir / "fig10_label_sample_size_panels.png",
    )

    # Error-burden plots.
    dom_first_err = pivot_for_plot(domain_summary, "domain", "mean_first_iteration_error_count")
    gram_first_err = pivot_for_plot(
        grammar_summary,
        "grammar",
        "mean_first_iteration_error_count",
        category_order=top_grammar,
    )
    draw_heatmap(
        dom_first_err,
        fig_dir / "fig11_domain_first_iteration_error_heatmap.png",
        "Mean First-Iteration Error Count by Domain and Model",
        "Errors",
        value_fmt=".2f",
    )
    draw_heatmap(
        gram_first_err,
        fig_dir / "fig12_grammar_topK_first_iteration_error_heatmap.png",
        f"Mean First-Iteration Error Count by Top-{args.top_grammar_k} Grammar Labels and Model",
        "Errors",
        value_fmt=".2f",
    )
    draw_grouped_bar(
        difficulty_for_plot,
        "difficulty",
        "mean_first_iteration_error_count",
        fig_dir / "fig13_difficulty_first_iteration_error_grouped_bar.png",
        "Mean First-Iteration Error Count by Difficulty and Model",
        "Errors",
        diff_order,
    )

    make_findings_md(
        domain_summary=domain_summary,
        grammar_summary=grammar_summary,
        difficulty_summary=difficulty_summary,
        output_path=output_root / "short_writeup.md",
    )

    print(f"[ok] output root: {output_root}")
    print(f"[ok] data: {data_dir}")
    print(f"[ok] figures: {fig_dir}")
    print("[ok] writeup: short_writeup.md")


if __name__ == "__main__":
    main()
