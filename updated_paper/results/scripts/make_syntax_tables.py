#!/usr/bin/env python3
"""Generate LaTeX tables for syntax-only campaign results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo_root / "paper" / "results" / "data")
    parser.add_argument("--tables-dir", type=Path, default=repo_root / "paper" / "results" / "tables")
    return parser.parse_args()


def pct(x: Any, d: int = 2) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{d}f}\\%"


def num(x: Any, d: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{d}f}"


def int_or_na(x: Any) -> str:
    if pd.isna(x):
        return "NA"
    return str(int(x))


def write_tex_table(path: Path, df: pd.DataFrame, caption: str, label: str) -> None:
    tex = df.to_latex(index=False, escape=False)
    wrapped = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{tex}\n"
        "\\end{table}\n"
    )
    path.write_text(wrapped, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    tables_dir = args.tables_dir.resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)

    prompt_df = pd.read_csv(data_dir / "prompt_level_syntax_metrics.csv")
    model_df = pd.read_csv(data_dir / "model_level_syntax_summary.csv")
    iter_df = pd.read_csv(data_dir / "iteration_level_syntax_metrics.csv")
    error_df = pd.read_csv(data_dir / "error_taxonomy_summary.csv")
    stat_tests = json.loads((data_dir / "stat_tests.json").read_text(encoding="utf-8"))

    # Table 1: Overall compile outcomes
    overall = model_df[(model_df["provider"] == "ALL") & (model_df["model"] == "ALL")].iloc[0]
    tbl_overall = pd.DataFrame(
        [
            {
                "Metric": "Prompts",
                "Value": int_or_na(overall["total_prompts"]),
            },
            {
                "Metric": "First-shot pass",
                "Value": f"{int_or_na(overall['first_shot_pass_count'])} ({pct(overall['first_shot_pass_rate_pct'])})",
            },
            {
                "Metric": "First-shot fail",
                "Value": f"{int_or_na(overall['first_shot_fail_count'])} ({pct(overall['first_shot_fail_rate_pct'])})",
            },
            {
                "Metric": "Eventual pass (pipeline)",
                "Value": f"{int_or_na(overall['eventual_pass_count'])} ({pct(overall['eventual_pass_rate_pct'])})",
            },
            {
                "Metric": "Absolute gain",
                "Value": pct(overall["absolute_gain_pp"]),
            },
            {
                "Metric": "Relative gain over first-shot",
                "Value": pct(overall["relative_gain_pct"]),
            },
            {
                "Metric": "Unresolved",
                "Value": f"{int_or_na(overall['unresolved_count'])} ({pct(overall['unresolved_rate_pct'])})",
            },
            {
                "Metric": "Recovered after first-shot failure",
                "Value": int_or_na(overall["recovered_count"]),
            },
            {
                "Metric": "First-shot pass 95\\% Wilson CI",
                "Value": f"[{pct(overall['first_shot_pass_ci95_low_pct'])}, {pct(overall['first_shot_pass_ci95_high_pct'])}]",
            },
            {
                "Metric": "Eventual pass 95\\% Wilson CI",
                "Value": f"[{pct(overall['eventual_pass_ci95_low_pct'])}, {pct(overall['eventual_pass_ci95_high_pct'])}]",
            },
            {
                "Metric": "Mean iterations-to-success",
                "Value": num(overall["iterations_to_success_mean"]),
            },
            {
                "Metric": "Mean iterations 95\\% bootstrap CI",
                "Value": f"[{num(overall['iterations_to_success_bootstrap_ci95_low'])}, {num(overall['iterations_to_success_bootstrap_ci95_high'])}]",
            },
        ]
    )
    write_tex_table(
        tables_dir / "table_overall_compile_outcomes.tex",
        tbl_overall,
        caption="Overall compiler-gated outcomes for single-shot baseline vs iterative pipeline.",
        label="tab:overall_compile",
    )

    # Table 2: Model comparison
    model_rows = model_df[~((model_df["provider"] == "ALL") & (model_df["model"] == "ALL"))].copy()
    model_rows = model_rows.sort_values(["provider", "model"]) 
    tbl_model = pd.DataFrame(
        {
            "Provider": model_rows["provider"],
            "Model": model_rows["model"],
            "N": model_rows["total_prompts"].map(int_or_na),
            "First-shot pass": model_rows["first_shot_pass_rate_pct"].map(pct),
            "Eventual pass": model_rows["eventual_pass_rate_pct"].map(pct),
            "Gain (pp)": model_rows["absolute_gain_pp"].map(pct),
            "Unresolved": model_rows["unresolved_rate_pct"].map(pct),
            "Mean iters": model_rows["iterations_to_success_mean"].map(num),
            "Median iters": model_rows["iterations_to_success_median"].map(num),
            "Max iters": model_rows["iterations_to_success_max"].map(num),
        }
    )
    write_tex_table(
        tables_dir / "table_model_comparison.tex",
        tbl_model,
        caption="Per-model syntactic reliability summary.",
        label="tab:model_comparison",
    )

    # Table 3: Iteration distribution (overall + per model)
    tmp = prompt_df.copy()
    tmp["iterations_to_success"] = pd.to_numeric(tmp["iterations_to_success"], errors="coerce")
    dist_all = (
        tmp.groupby("iterations_to_success", dropna=True)
        .size()
        .reset_index(name="count")
        .sort_values("iterations_to_success")
    )
    dist_all["Group"] = "ALL/ALL"
    dist_all["Share"] = dist_all["count"] / dist_all["count"].sum() * 100.0

    dist_model = (
        tmp.groupby(["provider", "model", "iterations_to_success"], dropna=True)
        .size()
        .reset_index(name="count")
        .sort_values(["provider", "model", "iterations_to_success"])
    )
    model_totals = dist_model.groupby(["provider", "model"]) ["count"].transform("sum")
    dist_model["Share"] = dist_model["count"] / model_totals * 100.0
    dist_model["Group"] = dist_model["provider"] + "/" + dist_model["model"]

    dist_tbl = pd.concat(
        [
            dist_all[["Group", "iterations_to_success", "count", "Share"]],
            dist_model[["Group", "iterations_to_success", "count", "Share"]],
        ],
        ignore_index=True,
    )
    dist_tbl = dist_tbl.rename(
        columns={
            "iterations_to_success": "Iteration",
            "count": "Count",
        }
    )
    dist_tbl["Iteration"] = dist_tbl["Iteration"].map(num)
    dist_tbl["Share"] = dist_tbl["Share"].map(pct)
    write_tex_table(
        tables_dir / "table_iteration_distribution.tex",
        dist_tbl,
        caption="Distribution of iterations required to reach first successful compile.",
        label="tab:iteration_distribution",
    )

    # Table 4: Top error taxonomy (first iteration and first failed iteration)
    err_top = error_df[(error_df["provider"] == "ALL") & (error_df["model"] == "ALL") & (error_df["scope"] == "first_iteration")].copy()
    err_top = err_top.sort_values(["error_count", "error_family"], ascending=[False, True]).head(20)
    err_tbl = pd.DataFrame(
        {
            "Error family": err_top["error_family"],
            "Count": err_top["error_count"].map(int_or_na),
            "Prompts affected": err_top["prompts_affected"].map(int_or_na),
        }
    )
    write_tex_table(
        tables_dir / "table_error_taxonomy_top.tex",
        err_tbl,
        caption="Top compiler error families on first iteration (pooled across models).",
        label="tab:error_taxonomy_top",
    )

    # Table 5: Hardest prompts
    hardest = prompt_df.copy()
    hardest = hardest.sort_values(
        ["total_error_count_across_iterations", "iterations_run", "first_iteration_error_count", "prompt_id"],
        ascending=[False, False, False, True],
    ).head(25)
    hardest_tbl = pd.DataFrame(
        {
            "Provider": hardest["provider"],
            "Model": hardest["model"],
            "Prompt ID": hardest["prompt_id"].map(int_or_na),
            "First-iter errors": hardest["first_iteration_error_count"].map(int_or_na),
            "Total errors": hardest["total_error_count_across_iterations"].map(int_or_na),
            "Iterations run": hardest["iterations_run"].map(int_or_na),
            "Eventual pass": hardest["eventual_pass"].astype(str),
        }
    )
    write_tex_table(
        tables_dir / "table_hardest_prompts.tex",
        hardest_tbl,
        caption="High-burden prompts by cumulative compiler error volume.",
        label="tab:hardest_prompts",
    )

    # Table 6: Runtime/token/cost summary
    rt = prompt_df.copy()
    grouped = rt.groupby(["provider", "model"], as_index=False).agg(
        n=("prompt_id", "count"),
        n_runtime_available=("wall_time_sec", lambda s: int(s.notna().sum())),
        n_tokens_available=("token_total", lambda s: int(s.notna().sum())),
        wall_time_mean=("wall_time_sec", "mean"),
        wall_time_median=("wall_time_sec", "median"),
        wall_time_max=("wall_time_sec", "max"),
        token_total_mean=("token_total", "mean"),
        token_total_median=("token_total", "median"),
        token_total_max=("token_total", "max"),
        est_cost_mean=("estimated_cost_usd", "mean"),
        est_cost_total=("estimated_cost_usd", "sum"),
    )
    rt_tbl = pd.DataFrame(
        {
            "Provider": grouped["provider"],
            "Model": grouped["model"],
            "N": grouped["n"].map(int_or_na),
            "N runtime": grouped["n_runtime_available"].map(int_or_na),
            "N tokens": grouped["n_tokens_available"].map(int_or_na),
            "Mean wall time (s)": grouped["wall_time_mean"].map(num),
            "Median wall time (s)": grouped["wall_time_median"].map(num),
            "Max wall time (s)": grouped["wall_time_max"].map(num),
            "Mean tokens": grouped["token_total_mean"].map(num),
            "Median tokens": grouped["token_total_median"].map(num),
            "Max tokens": grouped["token_total_max"].map(num),
            "Mean cost (USD)": grouped["est_cost_mean"].map(num),
            "Total cost (USD)": grouped["est_cost_total"].map(num),
        }
    )
    write_tex_table(
        tables_dir / "table_runtime_token_cost_summary.tex",
        rt_tbl,
        caption="Runtime, token, and estimated cost summary by model (cost is NA when unavailable).",
        label="tab:runtime_token_cost",
    )

    print("[ok] wrote LaTeX tables to", tables_dir)


if __name__ == "__main__":
    main()
