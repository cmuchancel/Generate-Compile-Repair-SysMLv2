#!/usr/bin/env python3
"""Plot generated SysML line count vs iterations-to-converge with trendline and R^2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "ANTHROPIC": "Anthropic",
    "DEEPSEEK_REASONER": "DeepSeek",
    "MISTRAL_LARGE": "Mistral",
    "OPENAI": "OpenAI",
}

MODEL_COLORS = {
    "OpenAI": "#4C78A8",
    "Anthropic": "#F58518",
    "DeepSeek": "#54A24B",
    "Mistral": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    base_dir = script_path.parents[1]
    repo_root = script_path.parents[2]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=repo_root / "api_loop" / "analysis" / "sysml_lines_vs_iterations_all_models.csv",
        help="CSV with columns model_name, iterations_to_converge, sysml_line_count.",
    )
    parser.add_argument(
        "--output-fig",
        type=Path,
        default=base_dir / "figures" / "fig14_generated_output_lines_vs_iterations_scatter_trend.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=base_dir / "data" / "summary_generated_lines_vs_iterations_fit.csv",
        help="Output 1-row CSV with fit stats.",
    )
    parser.add_argument(
        "--output-model-summary",
        type=Path,
        default=base_dir / "data" / "summary_generated_lines_by_model.csv",
        help="Output CSV with average generated line counts by model.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    df = df.dropna(subset=["sysml_line_count", "iterations_to_converge"]).copy()
    df["sysml_line_count"] = pd.to_numeric(df["sysml_line_count"], errors="coerce")
    df["iterations_to_converge"] = pd.to_numeric(df["iterations_to_converge"], errors="coerce")
    df = df.dropna(subset=["sysml_line_count", "iterations_to_converge"]).copy()
    df = df.reset_index(drop=True)

    df["model_label"] = df["model_name"].map(MODEL_LABELS).fillna(df["model_name"])

    x = df["sysml_line_count"].to_numpy(dtype=float)
    y = df["iterations_to_converge"].to_numpy(dtype=float)

    # OLS linear fit: y = a*x + b
    a_lin, b_lin = np.polyfit(x, y, 1)
    y_hat_lin = a_lin * x + b_lin
    ss_res_lin = float(np.sum((y - y_hat_lin) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_lin = 1.0 - (ss_res_lin / ss_tot) if ss_tot > 0 else float("nan")

    # Jitter only for display (keep fit on raw values).
    rng = np.random.default_rng(7)
    y_jitter = y + rng.normal(0.0, 0.045, size=len(y))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(13.8, 5.8),
        dpi=220,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    legend_handles = []
    legend_labels = []

    for model in ["OpenAI", "Anthropic", "DeepSeek", "Mistral"]:
        sub = df[df["model_label"] == model]
        if sub.empty:
            continue
        idx = sub.index.to_numpy()
        h1 = ax1.scatter(
            sub["sysml_line_count"].to_numpy(dtype=float),
            y_jitter[idx],
            s=18,
            alpha=0.55,
            color=MODEL_COLORS[model],
            edgecolors="none",
            label=f"{model} (n={len(sub)})",
        )
        legend_handles.append(h1)
        legend_labels.append(f"{model} (n={len(sub)})")

    # Linear-x panel
    xs_lin = np.linspace(float(np.min(x)), float(np.max(x)), 300)
    ys_lin = a_lin * xs_lin + b_lin
    l1 = ax1.plot(xs_lin, ys_lin, color="#222222", linewidth=2.0, linestyle="--", label="Linear fit")[0]
    ax1.set_xlabel("Generated SysML Line Count (linear scale)", fontsize=10)
    ax1.set_ylabel("Iterations to Converge", fontsize=10)
    ax1.set_title("Linear X-Scale", fontsize=11)
    ax1.grid(alpha=0.25)
    txt_lin = f"y = {a_lin:.5f}x + {b_lin:.3f}\n$R^2$ = {r2_lin:.4f}"
    ax1.text(
        0.03,
        0.97,
        txt_lin,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#666666"},
    )

    # Right panel: average generated line count by model.
    model_order = ["OpenAI", "Anthropic", "DeepSeek", "Mistral"]
    agg = (
        df.groupby("model_label")["sysml_line_count"]
        .agg(["mean", "std", "count"])
        .reindex(model_order)
    )
    agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    x_idx = np.arange(len(agg))
    bars = ax2.bar(
        x_idx,
        agg["mean"].to_numpy(dtype=float),
        yerr=agg["se"].to_numpy(dtype=float),
        color=[MODEL_COLORS[m] for m in agg.index],
        capsize=3,
        alpha=0.9,
    )
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(agg.index.tolist(), rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Average Generated SysML Lines", fontsize=10)
    ax2.set_title("Average Output Length by Model", fontsize=11)
    ax2.grid(axis="y", alpha=0.25)
    ymax = float((agg["mean"] + agg["se"]).max()) if not agg.empty else 1.0
    ax2.set_ylim(0, ymax * 1.18)
    for i, (bar, mean_v) in enumerate(zip(bars, agg["mean"].to_numpy(dtype=float))):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            mean_v + float(agg["se"].iloc[i]) + ymax * 0.025,
            f"{mean_v:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    y_min = float(np.min(y)) - 0.2
    y_max = float(np.max(y)) + 0.25
    ax1.set_ylim(y_min, y_max)

    fig.suptitle(
        f"Generated Output Length and Convergence Behavior (N={len(df)})",
        fontsize=12,
    )
    fig.legend(
        legend_handles + [l1],
        legend_labels + ["Linear fit"],
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    ensure_parent(args.output_fig)
    fig.savefig(args.output_fig)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "n_points": int(len(df)),
                "linear_slope": float(a_lin),
                "linear_intercept": float(b_lin),
                "linear_r2": float(r2_lin),
                "x_min": float(np.min(x)),
                "x_max": float(np.max(x)),
                "y_min": float(np.min(y)),
                "y_max": float(np.max(y)),
            }
        ]
    )
    ensure_parent(args.output_summary)
    summary.to_csv(args.output_summary, index=False)
    ensure_parent(args.output_model_summary)
    (
        agg.reset_index()
        .rename(columns={"model_label": "model"})
        .to_csv(args.output_model_summary, index=False)
    )

    print(f"[ok] wrote figure: {args.output_fig}")
    print(f"[ok] wrote summary: {args.output_summary}")
    print(f"[ok] wrote model summary: {args.output_model_summary}")
    print(
        f"[ok] linear fit: slope={a_lin:.8f}, intercept={b_lin:.5f}, r2={r2_lin:.6f}, n={len(df)}"
    )


if __name__ == "__main__":
    main()
