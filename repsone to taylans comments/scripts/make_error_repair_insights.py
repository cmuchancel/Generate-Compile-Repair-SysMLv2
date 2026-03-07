#!/usr/bin/env python3
"""Build error-family persistence and time-to-repair insights from iteration readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        default=base_dir / "error_repair_insights",
        help="Output root for data/figures/writeup.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_family_counts(raw: object) -> Dict[str, int]:
    if not isinstance(raw, str):
        return {}
    try:
        obj = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, int] = {}
    for k, v in obj.items():
        try:
            vi = int(v)
        except Exception:
            continue
        if vi > 0:
            out[str(k)] = vi
    return out


def make_expanded_iteration_rows(iter_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for rec in iter_df.to_dict(orient="records"):
        fam = parse_family_counts(rec.get("error_families_json"))
        for family, cnt in fam.items():
            rows.append(
                {
                    "provider": rec["provider"],
                    "model": rec["model"],
                    "model_label": rec["model_label"],
                    "prompt_id": int(rec["prompt_id"]),
                    "iteration_index": int(rec["iteration_index"]),
                    "family": family,
                    "count": int(cnt),
                    "iteration_time_sec": float(rec["iteration_time_sec"]) if pd.notna(rec["iteration_time_sec"]) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_episodes(iter_df: pd.DataFrame) -> pd.DataFrame:
    episodes: List[Dict[str, object]] = []
    keys = ["provider", "model", "model_label", "prompt_id"]

    for key_vals, group in iter_df.groupby(keys, sort=False):
        provider, model, model_label, prompt_id = key_vals
        g = group.sort_values("iteration_index")
        rows: List[Dict[str, object]] = []
        all_families: set[str] = set()

        for _, row in g.iterrows():
            fam = parse_family_counts(row["error_families_json"])
            rows.append(
                {
                    "iteration_index": int(row["iteration_index"]),
                    "iteration_time_sec": float(row["iteration_time_sec"]) if pd.notna(row["iteration_time_sec"]) else 0.0,
                    "families": fam,
                }
            )
            all_families.update(fam.keys())

        for family in sorted(all_families):
            active = False
            start_iter = -1
            start_pos = -1
            last_present_iter = -1
            last_present_pos = -1

            for i, r in enumerate(rows):
                c = int(r["families"].get(family, 0))
                curr_iter = int(r["iteration_index"])
                if c > 0 and not active:
                    active = True
                    start_iter = curr_iter
                    start_pos = i
                    last_present_iter = curr_iter
                    last_present_pos = i
                elif c > 0 and active:
                    last_present_iter = curr_iter
                    last_present_pos = i
                elif c == 0 and active:
                    repair_iter = curr_iter
                    iterations_to_repair = repair_iter - start_iter
                    persisted_iterations = last_present_iter - start_iter + 1
                    elapsed_sec_to_repair = float(
                        sum(rows[j]["iteration_time_sec"] for j in range(start_pos, i + 1))
                    )
                    episodes.append(
                        {
                            "provider": provider,
                            "model": model,
                            "model_label": model_label,
                            "prompt_id": int(prompt_id),
                            "family": family,
                            "start_iteration": start_iter,
                            "repair_iteration": repair_iter,
                            "iterations_to_repair": int(iterations_to_repair),
                            "persisted_iterations": int(persisted_iterations),
                            "elapsed_sec_to_repair": elapsed_sec_to_repair,
                            "resolved": True,
                        }
                    )
                    active = False
                    start_iter = -1
                    start_pos = -1
                    last_present_iter = -1
                    last_present_pos = -1

            if active:
                elapsed_sec = float(
                    sum(rows[j]["iteration_time_sec"] for j in range(start_pos, last_present_pos + 1))
                )
                episodes.append(
                    {
                        "provider": provider,
                        "model": model,
                        "model_label": model_label,
                        "prompt_id": int(prompt_id),
                        "family": family,
                        "start_iteration": start_iter,
                        "repair_iteration": np.nan,
                        "iterations_to_repair": np.nan,
                        "persisted_iterations": int(last_present_iter - start_iter + 1),
                        "elapsed_sec_to_repair": elapsed_sec,
                        "resolved": False,
                    }
                )

    return pd.DataFrame(episodes)


def summarize_families(expanded: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    if expanded.empty:
        return pd.DataFrame()

    volume = (
        expanded.groupby("family")
        .agg(
            total_error_instances=("count", "sum"),
            iteration_rows_with_family=("family", "count"),
            prompts_affected=("prompt_id", "nunique"),
        )
        .reset_index()
    )

    resolved = episodes[episodes["resolved"]].copy()
    repair = (
        resolved.groupby("family")
        .agg(
            episodes=("family", "count"),
            mean_iterations_to_repair=("iterations_to_repair", "mean"),
            median_iterations_to_repair=("iterations_to_repair", "median"),
            p90_iterations_to_repair=("iterations_to_repair", lambda s: s.quantile(0.9)),
            max_iterations_to_repair=("iterations_to_repair", "max"),
            std_iterations_to_repair=("iterations_to_repair", "std"),
            mean_persisted_iterations=("persisted_iterations", "mean"),
            p90_persisted_iterations=("persisted_iterations", lambda s: s.quantile(0.9)),
            max_persisted_iterations=("persisted_iterations", "max"),
            mean_elapsed_sec_to_repair=("elapsed_sec_to_repair", "mean"),
            median_elapsed_sec_to_repair=("elapsed_sec_to_repair", "median"),
            p90_elapsed_sec_to_repair=("elapsed_sec_to_repair", lambda s: s.quantile(0.9)),
            share_two_plus_iterations=("iterations_to_repair", lambda s: float((s >= 2).mean())),
        )
        .reset_index()
    )
    repair["ci95_iterations_to_repair"] = 1.96 * (
        repair["std_iterations_to_repair"] / np.sqrt(repair["episodes"].clip(lower=1))
    )

    out = volume.merge(repair, on="family", how="left")
    out["share_total_error_instances_pct"] = 100.0 * out["total_error_instances"] / out["total_error_instances"].sum()
    out["share_two_plus_iterations_pct"] = 100.0 * out["share_two_plus_iterations"].fillna(0.0)
    return out.sort_values("total_error_instances", ascending=False)


def summarize_model_family(episodes: pd.DataFrame, min_episodes: int = 3) -> pd.DataFrame:
    resolved = episodes[episodes["resolved"]].copy()
    mf = (
        resolved.groupby(["model_label", "family"])
        .agg(
            episodes=("family", "count"),
            mean_iterations_to_repair=("iterations_to_repair", "mean"),
            median_iterations_to_repair=("iterations_to_repair", "median"),
            mean_elapsed_sec_to_repair=("elapsed_sec_to_repair", "mean"),
        )
        .reset_index()
    )
    return mf[mf["episodes"] >= min_episodes].copy()


def draw_pareto(summary: pd.DataFrame, out: Path) -> None:
    df = summary.sort_values("total_error_instances", ascending=False).copy()
    df["cum_pct"] = 100.0 * df["total_error_instances"].cumsum() / df["total_error_instances"].sum()
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(11, 5), dpi=220)
    bars = ax1.bar(x, df["total_error_instances"].to_numpy(dtype=float), color="#4C78A8")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["family"].tolist(), rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("Total Error Instances", fontsize=9)
    ax1.set_title("Error Family Pareto (All Iterations, All Models)", fontsize=11)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, df["cum_pct"].to_numpy(dtype=float), color="#E45756", marker="o", linewidth=1.8)
    ax2.set_ylabel("Cumulative Share (%)", fontsize=9)
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_hardest_by_iterations(summary: pd.DataFrame, out: Path, min_episodes: int = 5, top_k: int = 10) -> None:
    df = summary[summary["episodes"] >= min_episodes].copy()
    df = df.sort_values("mean_iterations_to_repair", ascending=False).head(top_k)
    y = np.arange(len(df))
    vals = df["mean_iterations_to_repair"].to_numpy(dtype=float)
    errs = df["ci95_iterations_to_repair"].fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=220)
    ax.barh(y, vals, xerr=errs, color="#72B7B2", error_kw={"elinewidth": 1.2, "ecolor": "#1f1f1f", "capthick": 1.2}, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["family"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Additional Iterations to Repair", fontsize=9)
    ax.set_title(f"Hardest Error Families to Repair (n >= {min_episodes} episodes)", fontsize=11)
    ax.grid(axis="x", alpha=0.25)

    for i, (v, e, n) in enumerate(zip(vals, errs, df["episodes"].to_numpy(dtype=int))):
        ax.text(v + e + 0.02, i, f"{v:.2f} (n={n})", va="center", ha="left", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_time_to_repair_boxplot(episodes: pd.DataFrame, out: Path, top_k: int = 8) -> None:
    resolved = episodes[episodes["resolved"]].copy()
    counts = resolved["family"].value_counts().head(top_k)
    fams = counts.index.tolist()
    data = [resolved.loc[resolved["family"] == f, "elapsed_sec_to_repair"].to_numpy(dtype=float) for f in fams]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=220)
    ax.boxplot(data, tick_labels=fams, showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("Elapsed Time to Repair (sec, log scale)", fontsize=9)
    ax.set_title(f"Time-to-Repair Distributions for Top-{top_k} Error Families", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_model_family_heatmap(model_family: pd.DataFrame, summary: pd.DataFrame, out: Path, top_k: int = 10) -> None:
    top_fams = summary.sort_values("episodes", ascending=False)["family"].head(top_k).tolist()
    pivot = (
        model_family[model_family["family"].isin(top_fams)]
        .pivot(index="model_label", columns="family", values="mean_iterations_to_repair")
        .reindex(MODEL_ORDER)
        .reindex(columns=top_fams)
    )

    arr = pivot.to_numpy(dtype=float)
    valid = arr[~np.isnan(arr)]
    vmin = float(valid.min()) if valid.size else 0.0
    vmax = float(valid.max()) if valid.size else 1.0

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=220)
    cmap = plt.get_cmap("YlGnBu")
    im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title("Mean Iterations-to-Repair by Error Family and Model", fontsize=11)
    ax.set_xticks(np.arange(arr.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(arr.shape[0]))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            t = "NA" if np.isnan(v) else f"{v:.2f}"
            color = "black" if np.isnan(v) or v < (vmin + vmax) / 2 else "white"
            ax.text(j, i, t, ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Iterations", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_effort_map(summary: pd.DataFrame, out: Path, min_episodes: int = 3) -> None:
    df = summary[summary["episodes"] >= min_episodes].copy()
    x = df["total_error_instances"].to_numpy(dtype=float)
    y = df["mean_iterations_to_repair"].to_numpy(dtype=float)
    size = 20 + 6.0 * df["mean_elapsed_sec_to_repair"].fillna(0.0).to_numpy(dtype=float)
    color = df["share_two_plus_iterations_pct"].fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=220)
    sc = ax.scatter(x, y, s=size, c=color, cmap="magma", alpha=0.8, edgecolors="white", linewidths=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Total Error Instances (log scale)", fontsize=9)
    ax.set_ylabel("Mean Additional Iterations to Repair", fontsize=9)
    ax.set_title("Error Family Effort Map: Frequency vs Repair Difficulty", fontsize=11)
    ax.grid(alpha=0.25)

    top = df.sort_values(["mean_iterations_to_repair", "episodes"], ascending=[False, False]).head(8)
    for _, r in top.iterrows():
        ax.text(float(r["total_error_instances"]) * 1.03, float(r["mean_iterations_to_repair"]) + 0.01, str(r["family"]), fontsize=7)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Episodes requiring >=2 additional iterations (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_persistence_curves(episodes: pd.DataFrame, summary: pd.DataFrame, out: Path, top_k: int = 5) -> None:
    resolved = episodes[episodes["resolved"]].copy()
    top_fams = summary.sort_values("episodes", ascending=False)["family"].head(top_k).tolist()
    max_k = int(max(1, resolved["iterations_to_repair"].max()))

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=220)
    for fam in top_fams:
        s = resolved.loc[resolved["family"] == fam, "iterations_to_repair"].dropna().to_numpy(dtype=float)
        if len(s) == 0:
            continue
        ks = np.arange(1, max_k + 1)
        # Probability family still unresolved after k additional iterations.
        probs = np.array([(s >= k).mean() for k in ks], dtype=float)
        ax.plot(ks, probs, marker="o", linewidth=1.8, label=f"{fam} (n={len(s)})")

    ax.set_xlabel("Additional Iterations Since First Appearance (k)", fontsize=9)
    ax.set_ylabel("P(still unresolved after k)", fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Persistence Curves for Top-{top_k} Error Families", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_longest_persisting(summary: pd.DataFrame, out: Path, min_episodes: int = 5, top_k: int = 12) -> None:
    df = summary[summary["episodes"] >= min_episodes].copy()
    df = df.sort_values(
        ["p90_iterations_to_repair", "max_iterations_to_repair", "mean_iterations_to_repair"],
        ascending=False,
    ).head(top_k)
    y = np.arange(len(df))
    p90 = df["p90_iterations_to_repair"].to_numpy(dtype=float)
    max_v = df["max_iterations_to_repair"].to_numpy(dtype=float)
    mean_v = df["mean_iterations_to_repair"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=220)
    ax.barh(y, p90, color="#9ecae1", label="P90 additional iterations")
    ax.scatter(max_v, y, color="#08519c", s=30, label="Max observed additional iterations", zorder=3)
    ax.scatter(mean_v, y, color="#e6550d", s=26, label="Mean additional iterations", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["family"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Additional Iterations to Repair", fontsize=9)
    ax.set_title(
        f"Longest-Persisting Error Families (episodes >= {min_episodes})",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    for i, n in enumerate(df["episodes"].to_numpy(dtype=int)):
        ax.text(max_v[i] + 0.05, i, f"n={n}", va="center", ha="left", fontsize=7)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def make_writeup(summary: pd.DataFrame, episodes: pd.DataFrame, out: Path) -> None:
    resolved = episodes[episodes["resolved"]].copy()
    total_episodes = int(len(episodes))
    resolved_episodes = int(len(resolved))

    by_iters = summary[summary["episodes"] >= 5].sort_values("mean_iterations_to_repair", ascending=False).head(5)
    by_time = summary[summary["episodes"] >= 5].sort_values("mean_elapsed_sec_to_repair", ascending=False).head(5)
    by_tail = summary[summary["episodes"] >= 5].sort_values(
        ["p90_iterations_to_repair", "max_iterations_to_repair"],
        ascending=False,
    ).head(5)
    by_volume = summary.sort_values("total_error_instances", ascending=False).head(5)

    lines: List[str] = []
    lines.append("# Error Repair Insights (From Iteration Readouts)")
    lines.append("")
    lines.append("This analysis classifies actual compiler error families from iteration logs and quantifies repair effort.")
    lines.append("")
    lines.append("## Core Findings")
    lines.append("")
    lines.append(
        f"- Extracted **{total_episodes} error-family repair episodes**; resolved episodes: **{resolved_episodes} ({(100.0*resolved_episodes/max(total_episodes,1)):.1f}%)**."
    )
    lines.append("- Eventual convergence is 100% in this campaign, so repair burden is captured by additional iterations and elapsed seconds to repair.")
    lines.append("")
    lines.append("Highest-volume error families:")
    for _, r in by_volume.iterrows():
        lines.append(
            f"- `{r['family']}`: {int(r['total_error_instances'])} instances across {int(r['episodes'])} episodes; mean additional iterations-to-repair {float(r['mean_iterations_to_repair']):.2f}"
        )
    lines.append("")
    lines.append("Hardest families by additional iterations-to-repair (episodes >= 5):")
    for _, r in by_iters.iterrows():
        lines.append(
            f"- `{r['family']}`: mean {float(r['mean_iterations_to_repair']):.2f}, median {float(r['median_iterations_to_repair']):.2f}, P(requires >=2 extra iters) {float(r['share_two_plus_iterations_pct']):.1f}%"
        )
    lines.append("")
    lines.append("Longest-persisting families (tail burden, episodes >= 5):")
    for _, r in by_tail.iterrows():
        lines.append(
            f"- `{r['family']}`: P90 additional iterations {float(r['p90_iterations_to_repair']):.2f}, max additional iterations {int(r['max_iterations_to_repair'])}"
        )
    lines.append("")
    lines.append("Longest families by elapsed seconds-to-repair (episodes >= 5):")
    for _, r in by_time.iterrows():
        lines.append(
            f"- `{r['family']}`: mean {float(r['mean_elapsed_sec_to_repair']):.1f}s, median {float(r['median_elapsed_sec_to_repair']):.1f}s"
        )
    lines.append("")
    lines.append("## Figure Map")
    lines.append("")
    lines.append("- `figures/figE01_error_family_pareto_volume.png`")
    lines.append("- `figures/figE02_error_family_hardest_by_iterations.png`")
    lines.append("- `figures/figE03_error_family_time_to_repair_boxplot.png`")
    lines.append("- `figures/figE04_error_family_model_heatmap_mean_iterations.png`")
    lines.append("- `figures/figE05_error_family_effort_map.png`")
    lines.append("- `figures/figE06_error_family_persistence_curves_top5.png`")
    lines.append("- `figures/figE07_error_family_longest_persisting_tail.png`")
    lines.append("")
    lines.append("## Data Map")
    lines.append("")
    lines.append("- `data/error_family_iteration_expanded.csv`")
    lines.append("- `data/error_family_repair_episodes.csv`")
    lines.append("- `data/error_family_summary.csv`")
    lines.append("- `data/error_family_model_summary.csv`")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_root = args.output_root.resolve()
    data_dir = out_root / "data"
    fig_dir = out_root / "figures"
    ensure_dir(data_dir)
    ensure_dir(fig_dir)

    iter_path = repo_root / "paper" / "results" / "data" / "iteration_level_syntax_metrics.csv"
    iter_df = pd.read_csv(iter_path)
    iter_df["model_label"] = iter_df["provider"].map(MODEL_LABELS).fillna(iter_df["model"])
    iter_df["iteration_index"] = pd.to_numeric(iter_df["iteration_index"], errors="coerce")
    iter_df["iteration_time_sec"] = pd.to_numeric(iter_df["iteration_time_sec"], errors="coerce")
    iter_df = iter_df.dropna(subset=["iteration_index"]).copy()

    expanded = make_expanded_iteration_rows(iter_df)
    episodes = build_episodes(iter_df)
    summary = summarize_families(expanded, episodes)
    model_family = summarize_model_family(episodes)

    expanded.to_csv(data_dir / "error_family_iteration_expanded.csv", index=False)
    episodes.to_csv(data_dir / "error_family_repair_episodes.csv", index=False)
    summary.to_csv(data_dir / "error_family_summary.csv", index=False)
    model_family.to_csv(data_dir / "error_family_model_summary.csv", index=False)

    draw_pareto(summary, fig_dir / "figE01_error_family_pareto_volume.png")
    draw_hardest_by_iterations(summary, fig_dir / "figE02_error_family_hardest_by_iterations.png")
    draw_time_to_repair_boxplot(episodes, fig_dir / "figE03_error_family_time_to_repair_boxplot.png")
    draw_model_family_heatmap(
        model_family=model_family,
        summary=summary,
        out=fig_dir / "figE04_error_family_model_heatmap_mean_iterations.png",
    )
    draw_effort_map(summary, fig_dir / "figE05_error_family_effort_map.png")
    draw_persistence_curves(
        episodes=episodes,
        summary=summary,
        out=fig_dir / "figE06_error_family_persistence_curves_top5.png",
    )
    draw_longest_persisting(
        summary=summary,
        out=fig_dir / "figE07_error_family_longest_persisting_tail.png",
    )

    make_writeup(summary, episodes, out_root / "writeup.md")

    print(f"[ok] output_root: {out_root}")
    print(f"[ok] data files: {len(list(data_dir.glob('*.csv')))}")
    print(f"[ok] figure files: {len(list(fig_dir.glob('*.png')))}")
    print(f"[ok] writeup: {out_root / 'writeup.md'}")


if __name__ == "__main__":
    main()
