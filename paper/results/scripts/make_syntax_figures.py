#!/usr/bin/env python3
"""Generate a large syntax-only figure set for paper selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo_root / "paper" / "results" / "data")
    parser.add_argument("--figures-dir", type=Path, default=repo_root / "paper" / "results" / "figures")
    return parser.parse_args()


def pct(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def savefig(path: Path, dpi: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False)


def fig_placeholder(path: Path, title: str, message: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.axis("off")
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    savefig(path)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    prompt_df = pd.read_csv(data_dir / "prompt_level_syntax_metrics.csv")
    iter_df = pd.read_csv(data_dir / "iteration_level_syntax_metrics.csv")
    model_df = pd.read_csv(data_dir / "model_level_syntax_summary.csv")
    err_df = pd.read_csv(data_dir / "error_taxonomy_summary.csv")

    prompt_df["first_shot_pass"] = bool_series(prompt_df["first_shot_pass"])
    prompt_df["eventual_pass"] = bool_series(prompt_df["eventual_pass"])
    prompt_df["prompt_id"] = pd.to_numeric(prompt_df["prompt_id"], errors="coerce")
    prompt_df["iterations_to_success"] = pd.to_numeric(prompt_df["iterations_to_success"], errors="coerce")
    prompt_df["iterations_run"] = pd.to_numeric(prompt_df["iterations_run"], errors="coerce")
    prompt_df["first_iteration_error_count"] = pd.to_numeric(prompt_df["first_iteration_error_count"], errors="coerce")
    prompt_df["final_error_count"] = pd.to_numeric(prompt_df["final_error_count"], errors="coerce")
    prompt_df["total_error_count_across_iterations"] = pd.to_numeric(prompt_df["total_error_count_across_iterations"], errors="coerce")
    prompt_df["wall_time_sec"] = pd.to_numeric(prompt_df["wall_time_sec"], errors="coerce")
    prompt_df["token_total"] = pd.to_numeric(prompt_df["token_total"], errors="coerce")
    prompt_df["estimated_cost_usd"] = pd.to_numeric(prompt_df["estimated_cost_usd"], errors="coerce")

    iter_df["prompt_id"] = pd.to_numeric(iter_df["prompt_id"], errors="coerce")
    iter_df["iteration_index"] = pd.to_numeric(iter_df["iteration_index"], errors="coerce")
    iter_df["error_count"] = pd.to_numeric(iter_df["error_count"], errors="coerce")
    iter_df["warning_count"] = pd.to_numeric(iter_df.get("warning_count", pd.Series(dtype=float)), errors="coerce")
    iter_df["pass_at_iteration"] = bool_series(iter_df["pass_at_iteration"])

    sns.set_theme(style="whitegrid", context="talk")

    catalog: List[Tuple[str, str, str]] = []

    # 1. Baseline vs pipeline compile rate overall
    overall_first = prompt_df["first_shot_pass"].mean() * 100.0
    overall_final = prompt_df["eventual_pass"].mean() * 100.0
    plt.figure(figsize=(7, 5))
    overall_df = pd.DataFrame(
        {
            "label": ["Single-shot (iter 1)", "Pipeline (final)"],
            "rate": [overall_first, overall_final],
        }
    )
    sns.barplot(
        data=overall_df,
        x="label",
        y="rate",
        hue="label",
        palette=["#d95f02", "#1b9e77"],
        legend=False,
    )
    plt.ylabel("Compile success rate (%)")
    plt.ylim(0, 105)
    plt.title("Overall Compile Success: Baseline vs Pipeline")
    p = figures_dir / "01_baseline_vs_pipeline_compile_rate_overall.png"
    savefig(p)
    catalog.append((p.name, "Overall baseline vs pipeline compile rate", "Main headline figure"))

    # 2. Baseline vs pipeline by model
    by_model = prompt_df.groupby(["provider", "model"], as_index=False).agg(
        first_shot_pass_rate=("first_shot_pass", "mean"),
        eventual_pass_rate=("eventual_pass", "mean"),
    )
    by_model["label"] = by_model["provider"] + "\n" + by_model["model"]
    plot_df = by_model.melt(id_vars=["label"], value_vars=["first_shot_pass_rate", "eventual_pass_rate"], var_name="metric", value_name="rate")
    plot_df["rate"] = plot_df["rate"] * 100.0
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="label", y="rate", hue="metric", palette=["#d95f02", "#1b9e77"])
    plt.ylabel("Compile success rate (%)")
    plt.xlabel("Provider / Model")
    plt.ylim(0, 105)
    plt.title("Compile Success by Model: Baseline vs Pipeline")
    plt.legend(title="")
    p = figures_dir / "02_baseline_vs_pipeline_compile_rate_by_model.png"
    savefig(p)
    catalog.append((p.name, "Baseline vs pipeline by model", "Model comparison"))

    # 3. Absolute and relative improvement by model
    imp_df = by_model.copy()
    imp_df["absolute_gain_pp"] = (imp_df["eventual_pass_rate"] - imp_df["first_shot_pass_rate"]) * 100.0
    imp_df["relative_gain_pct"] = np.where(imp_df["first_shot_pass_rate"] > 0, (imp_df["eventual_pass_rate"] - imp_df["first_shot_pass_rate"]) / imp_df["first_shot_pass_rate"] * 100.0, np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=imp_df, x="label", y="absolute_gain_pp", ax=axes[0], color="#377eb8")
    axes[0].set_title("Absolute Gain (pp)")
    axes[0].set_ylabel("Percentage points")
    axes[0].set_xlabel("Provider / Model")
    sns.barplot(data=imp_df, x="label", y="relative_gain_pct", ax=axes[1], color="#4daf4a")
    axes[1].set_title("Relative Gain (%)")
    axes[1].set_ylabel("Percent")
    axes[1].set_xlabel("Provider / Model")
    p = figures_dir / "03_absolute_relative_improvement_by_model.png"
    savefig(p)
    catalog.append((p.name, "Absolute and relative improvement by model", "Improvement framing"))

    # 4. Paired outcome matrix overall
    fs = prompt_df["first_shot_pass"]
    ev = prompt_df["eventual_pass"]
    matrix = np.array([
        [int((fs & ev).sum()), int((fs & (~ev)).sum())],
        [int(((~fs) & ev).sum()), int(((~fs) & (~ev)).sum())],
    ])
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Final pass", "Final fail"], yticklabels=["First pass", "First fail"])
    plt.title("Paired Outcomes: First Iteration vs Final")
    p = figures_dir / "04_paired_outcome_matrix_overall.png"
    savefig(p)
    catalog.append((p.name, "Paired first-vs-final outcome matrix", "Transition structure"))

    # 5. Iterations-to-success histogram overall
    plt.figure(figsize=(8, 5))
    sns.histplot(prompt_df["iterations_to_success"].dropna(), discrete=True, stat="count", color="#1b9e77")
    plt.xlabel("Iterations to first successful compile")
    plt.ylabel("Prompt count")
    plt.title("Iterations-to-Success Distribution (Overall)")
    p = figures_dir / "05_iterations_to_success_hist_overall.png"
    savefig(p)
    catalog.append((p.name, "Iterations-to-success histogram", "Convergence profile"))

    # 6. ECDF overall
    vals = np.sort(prompt_df["iterations_to_success"].dropna().to_numpy())
    if vals.size > 0:
        y = np.arange(1, vals.size + 1) / vals.size
        plt.figure(figsize=(8, 5))
        plt.step(vals, y, where="post", color="#1b9e77")
        plt.xlabel("Iterations to success")
        plt.ylabel("ECDF")
        plt.title("ECDF of Iterations-to-Success")
        p = figures_dir / "06_iterations_to_success_ecdf_overall.png"
        savefig(p)
        catalog.append((p.name, "ECDF of iterations-to-success", "Convergence compactness"))

    # 7. Cumulative success by iteration
    max_iter = int(np.nanmax(prompt_df["iterations_run"])) if len(prompt_df) else 1
    xs = np.arange(1, max_iter + 1)
    cum_success = []
    for i in xs:
        cum_success.append((prompt_df["iterations_to_success"].fillna(np.inf) <= i).mean() * 100.0)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, cum_success, marker="o", color="#1b9e77")
    plt.ylim(0, 102)
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative success rate (%)")
    plt.title("Cumulative Compile Success by Iteration")
    p = figures_dir / "07_cumulative_success_by_iteration_overall.png"
    savefig(p)
    catalog.append((p.name, "Cumulative success by iteration", "Pipeline convergence"))

    # 8. Unresolved survival curve
    unresolved = [100.0 - c for c in cum_success]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, unresolved, marker="o", color="#d95f02")
    plt.ylim(0, 102)
    plt.xlabel("Iteration")
    plt.ylabel("Unresolved prompts (%)")
    plt.title("Unresolved-Survival Curve")
    p = figures_dir / "08_unresolved_survival_by_iteration_overall.png"
    savefig(p)
    catalog.append((p.name, "Unresolved-survival by iteration", "Remaining failure surface"))

    # 9. Error drop first -> final
    plt.figure(figsize=(8, 5))
    plt.scatter(prompt_df["first_iteration_error_count"], prompt_df["final_error_count"], alpha=0.65, color="#377eb8")
    max_err = max(prompt_df["first_iteration_error_count"].max(), prompt_df["final_error_count"].max()) if len(prompt_df) else 1
    plt.plot([0, max_err], [0, max_err], "k--", linewidth=1)
    plt.xlabel("First-iteration error count")
    plt.ylabel("Final-iteration error count")
    plt.title("Compiler Error Reduction per Prompt")
    p = figures_dir / "09_error_count_drop_first_to_final.png"
    savefig(p)
    catalog.append((p.name, "First-vs-final error counts", "Error elimination behavior"))

    # 10. First error count vs iterations to success
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=prompt_df, x="first_iteration_error_count", y="iterations_to_success", hue="provider", alpha=0.7)
    plt.title("Initial Error Burden vs Iterations to Success")
    p = figures_dir / "10_first_error_vs_iters_scatter.png"
    savefig(p)
    catalog.append((p.name, "Initial errors vs iterations", "Difficulty signal"))

    # 11. Error taxonomy Pareto first iteration
    err_first = err_df[(err_df["provider"] == "ALL") & (err_df["model"] == "ALL") & (err_df["scope"] == "first_iteration")].sort_values("error_count", ascending=False).head(20)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=err_first, x="error_family", y="error_count", color="#e41a1c")
    plt.xticks(rotation=60, ha="right")
    plt.title("Top Error Families on First Iteration")
    plt.ylabel("Count")
    p = figures_dir / "11_error_taxonomy_pareto_first_iteration.png"
    savefig(p)
    catalog.append((p.name, "Top first-iteration error families", "Failure mode taxonomy"))

    # 12. Error taxonomy Pareto all iterations
    err_all = err_df[(err_df["provider"] == "ALL") & (err_df["model"] == "ALL") & (err_df["scope"] == "all_iterations")].sort_values("error_count", ascending=False).head(20)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=err_all, x="error_family", y="error_count", color="#984ea3")
    plt.xticks(rotation=60, ha="right")
    plt.title("Top Error Families Across All Iterations")
    plt.ylabel("Count")
    p = figures_dir / "12_error_taxonomy_pareto_all_iterations.png"
    savefig(p)
    catalog.append((p.name, "Top all-iteration error families", "Aggregate failure modes"))

    # 13. Prompt x iteration heatmap
    heat = (
        iter_df.groupby(["prompt_id", "iteration_index"], as_index=False)["error_count"].sum()
        .pivot(index="prompt_id", columns="iteration_index", values="error_count")
        .sort_index()
        .fillna(0)
    )
    plt.figure(figsize=(12, 10))
    sns.heatmap(heat, cmap="mako", cbar_kws={"label": "Error count"})
    plt.title("Prompt-Level Error Heatmap Across Iterations (Pooled Models)")
    plt.xlabel("Iteration")
    plt.ylabel("Prompt ID")
    p = figures_dir / "13_prompt_iteration_error_heatmap_all_models.png"
    savefig(p)
    catalog.append((p.name, "Prompt x iteration error heatmap", "Difficulty heterogeneity"))

    # 14. Hardest prompts by total errors
    hard = prompt_df.groupby("prompt_id", as_index=False)["total_error_count_across_iterations"].sum().sort_values("total_error_count_across_iterations", ascending=False).head(25)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=hard, x="prompt_id", y="total_error_count_across_iterations", color="#ff7f00")
    plt.xticks(rotation=60, ha="right")
    plt.title("Hardest Prompts by Cumulative Error Count")
    plt.xlabel("Prompt ID")
    plt.ylabel("Total errors (pooled)")
    p = figures_dir / "14_hardest_prompts_total_errors.png"
    savefig(p)
    catalog.append((p.name, "Hardest prompts by total errors", "Outlier prompts"))

    # 15. Runtime distribution by model
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=prompt_df, x="model", y="wall_time_sec")
    plt.xticks(rotation=30, ha="right")
    plt.title("Runtime Distribution by Model")
    plt.xlabel("Model")
    plt.ylabel("Wall time (seconds)")
    p = figures_dir / "15_runtime_boxplot_by_model.png"
    savefig(p)
    catalog.append((p.name, "Runtime boxplot by model", "Efficiency tradeoff"))

    # 16. Token distribution by model
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=prompt_df, x="model", y="token_total")
    plt.xticks(rotation=30, ha="right")
    plt.title("Token Distribution by Model")
    plt.xlabel("Model")
    plt.ylabel("Total tokens per prompt")
    p = figures_dir / "16_tokens_boxplot_by_model.png"
    savefig(p)
    catalog.append((p.name, "Token boxplot by model", "Token efficiency"))

    # 17. Cost distribution by model (or placeholder)
    if prompt_df["estimated_cost_usd"].notna().any():
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=prompt_df, x="model", y="estimated_cost_usd")
        plt.xticks(rotation=30, ha="right")
        plt.title("Estimated Cost Distribution by Model")
        plt.xlabel("Model")
        plt.ylabel("Estimated cost (USD)")
        p = figures_dir / "17_cost_boxplot_by_model.png"
        savefig(p)
    else:
        p = figures_dir / "17_cost_boxplot_by_model.png"
        fig_placeholder(p, "Estimated Cost Distribution by Model", "Cost data unavailable in current artifacts.")
    catalog.append((p.name, "Cost distribution (or availability placeholder)", "Cost reporting"))

    # 18. Recovery-only subset distribution
    recovery = prompt_df[(~prompt_df["first_shot_pass"]) & (prompt_df["eventual_pass"])]
    if len(recovery) > 0:
        plt.figure(figsize=(8, 5))
        sns.histplot(recovery["iterations_to_success"].dropna(), discrete=True, color="#4daf4a")
        plt.title("Iterations-to-Success for Recovered Cases")
        plt.xlabel("Iterations to first success")
        plt.ylabel("Recovered prompt count")
        p = figures_dir / "18_recovery_only_iterations_hist.png"
        savefig(p)
    else:
        p = figures_dir / "18_recovery_only_iterations_hist.png"
        fig_placeholder(p, "Recovered Cases", "No recovered cases in current dataset.")
    catalog.append((p.name, "Recovery-only iterations histogram", "Repair behavior"))

    # 19. Iterations-to-success by model
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=prompt_df, x="model", y="iterations_to_success")
    plt.xticks(rotation=30, ha="right")
    plt.title("Iterations-to-Success by Model")
    plt.xlabel("Model")
    plt.ylabel("Iterations")
    p = figures_dir / "19_iterations_to_success_box_by_model.png"
    savefig(p)
    catalog.append((p.name, "Iterations-to-success by model", "Convergence by model"))

    # 20. First-shot failure rate by model
    fail_df = by_model.copy()
    fail_df["first_shot_fail_rate"] = 100.0 - fail_df["first_shot_pass_rate"] * 100.0
    plt.figure(figsize=(12, 6))
    sns.barplot(data=fail_df, x="label", y="first_shot_fail_rate", color="#d95f02")
    plt.ylim(0, 100)
    plt.ylabel("First-shot fail rate (%)")
    plt.xlabel("Provider / Model")
    plt.title("First-Shot Failure Rate by Model")
    p = figures_dir / "20_first_shot_failure_rate_by_model.png"
    savefig(p)
    catalog.append((p.name, "First-shot failure rate by model", "Baseline weakness"))

    # 21. Stacked first-shot vs recovered vs unresolved by model
    stack_rows = []
    for (provider, model), sub in prompt_df.groupby(["provider", "model"]):
        n = len(sub)
        first_pass = int(sub["first_shot_pass"].sum())
        recovered = int(((~sub["first_shot_pass"]) & sub["eventual_pass"]).sum())
        unresolved = int((~sub["eventual_pass"]).sum())
        stack_rows.append({"label": f"{provider}\n{model}", "first_pass": first_pass / n * 100.0, "recovered": recovered / n * 100.0, "unresolved": unresolved / n * 100.0})
    stack_df = pd.DataFrame(stack_rows)
    plt.figure(figsize=(12, 6))
    plt.bar(stack_df["label"], stack_df["first_pass"], label="First-shot pass", color="#1b9e77")
    plt.bar(stack_df["label"], stack_df["recovered"], bottom=stack_df["first_pass"], label="Recovered in loop", color="#377eb8")
    plt.bar(stack_df["label"], stack_df["unresolved"], bottom=stack_df["first_pass"] + stack_df["recovered"], label="Unresolved", color="#e41a1c")
    plt.ylabel("Share of prompts (%)")
    plt.ylim(0, 105)
    plt.title("Outcome Composition by Model")
    plt.legend()
    p = figures_dir / "21_outcome_composition_stacked_by_model.png"
    savefig(p)
    catalog.append((p.name, "Outcome composition stacked bars", "How final success is achieved"))

    # 22. Mean error count by iteration and model
    trend = iter_df.groupby(["provider", "model", "iteration_index"], as_index=False)["error_count"].mean()
    trend["label"] = trend["provider"] + "/" + trend["model"]
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=trend, x="iteration_index", y="error_count", hue="label", marker="o")
    plt.title("Mean Error Count by Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Mean error count")
    p = figures_dir / "22_iteration_error_trend_by_model.png"
    savefig(p)
    catalog.append((p.name, "Error trend by iteration and model", "Repair dynamics"))

    # 23. Warning vs error scatter per iteration
    if "warning_count" in iter_df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=iter_df, x="warning_count", y="error_count", hue="provider", alpha=0.6)
        plt.title("Warnings vs Errors per Iteration")
        p = figures_dir / "23_warning_vs_error_scatter.png"
        savefig(p)
    else:
        p = figures_dir / "23_warning_vs_error_scatter.png"
        fig_placeholder(p, "Warnings vs Errors", "warning_count unavailable.")
    catalog.append((p.name, "Warning-vs-error scatter", "Diagnostic only"))

    # 24. Prompt difficulty rank curve by model
    plt.figure(figsize=(12, 6))
    for (provider, model), sub in prompt_df.groupby(["provider", "model"]):
        vals = np.sort(sub["total_error_count_across_iterations"].fillna(0).to_numpy())[::-1]
        plt.plot(np.arange(1, len(vals) + 1), vals, label=f"{provider}/{model}")
    plt.xlabel("Prompt rank (hardest to easiest)")
    plt.ylabel("Total error count")
    plt.title("Prompt Difficulty Rank Curves by Model")
    plt.legend()
    p = figures_dir / "24_prompt_difficulty_rank_curves_by_model.png"
    savefig(p)
    catalog.append((p.name, "Prompt difficulty rank curves", "Tail difficulty behavior"))

    # 25. Runtime vs tokens scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=prompt_df, x="token_total", y="wall_time_sec", hue="provider", alpha=0.6)
    plt.xlabel("Total tokens")
    plt.ylabel("Wall time (s)")
    plt.title("Runtime vs Token Usage")
    p = figures_dir / "25_runtime_vs_tokens_scatter.png"
    savefig(p)
    catalog.append((p.name, "Runtime vs token scatter", "Cost/runtime scaling"))

    # Figure catalog markdown
    lines = [
        "# Figure Catalog",
        "",
        "| Figure | Purpose | When To Use |",
        "|---|---|---|",
    ]
    for name, purpose, use in catalog:
        lines.append(f"| `{name}` | {purpose} | {use} |")
    (figures_dir / "FIGURE_CATALOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] wrote {len(catalog)} figures to {figures_dir}")
    print(f"[ok] wrote {figures_dir / 'FIGURE_CATALOG.md'}")


if __name__ == "__main__":
    main()
