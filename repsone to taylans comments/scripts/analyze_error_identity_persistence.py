#!/usr/bin/env python3
"""Analyze whether persistent errors are the same exact diagnostics or just the same family."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
DIAG_RE = re.compile(
    r"^(?P<file>[^:\n]+):(?P<line>\d+):(?P<col>\d+):\s+"
    r"(?P<sev>error|warning)\s+\((?P<family>[^)]+)\):\s*(?P<msg>.*)$"
)

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
    base_dir = script_path.parents[1]
    repo_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=base_dir / "error_repair_insights",
        help="Writes to error_repair_insights/{data,figures}.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_exact_message(msg: str) -> str:
    msg = msg.strip()
    msg = re.sub(r"\s+", " ", msg)
    return msg


def normalize_template_message(msg: str) -> str:
    msg = normalize_exact_message(msg)
    msg = re.sub(r"'[^']*'", "<q>", msg)
    msg = re.sub(r'"[^"]*"', "<dq>", msg)
    msg = re.sub(r"\b\d+\b", "<n>", msg)
    msg = re.sub(r"\[[^\]]*\]", "[...]", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg


def parse_diagnostics(stderr: str) -> List[Dict[str, str]]:
    text = ANSI_RE.sub("", stderr or "")
    out: List[Dict[str, str]] = []
    for line in text.splitlines():
        m = DIAG_RE.match(line.strip())
        if not m:
            continue
        sev = m.group("sev")
        if sev != "error":
            continue
        family = m.group("family").strip()
        msg_raw = m.group("msg").strip()
        msg_exact = normalize_exact_message(msg_raw)
        msg_template = normalize_template_message(msg_raw)
        out.append(
            {
                "family": family,
                "message_raw": msg_raw,
                "message_exact": msg_exact,
                "message_template": msg_template,
                "signature_exact": f"{family}::{msg_exact}",
                "signature_template": f"{family}::{msg_template}",
            }
        )
    return out


def load_iteration_diagnostics(repo_root: Path) -> pd.DataFrame:
    iter_path = repo_root / "paper" / "results" / "data" / "iteration_level_syntax_metrics.csv"
    iter_df = pd.read_csv(iter_path)

    manifest_cache: Dict[str, str] = {}
    rows: List[Dict[str, object]] = []

    for rec in iter_df.to_dict(orient="records"):
        source_path = str(rec["source_path"])
        if source_path not in manifest_cache:
            try:
                manifest = json.loads(Path(source_path).read_text(encoding="utf-8"))
                manifest_cache[source_path] = manifest.get("run_log_path", "")
            except Exception:
                manifest_cache[source_path] = ""
        run_log_path = manifest_cache[source_path]
        if not run_log_path:
            continue

        try:
            run_log = json.loads(Path(run_log_path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(run_log, list):
            continue

        target_iter = int(rec["iteration_index"])
        entry = None
        for item in run_log:
            if int(item.get("iteration", -1)) == target_iter:
                entry = item
                break
        if entry is None:
            continue

        diags = parse_diagnostics(entry.get("compiler_stderr", ""))
        for d in diags:
            rows.append(
                {
                    "provider": rec["provider"],
                    "model": rec["model"],
                    "model_label": MODEL_LABELS.get(str(rec["provider"]), str(rec["model"])),
                    "prompt_id": int(rec["prompt_id"]),
                    "iteration_index": int(rec["iteration_index"]),
                    "family": d["family"],
                    "signature_exact": d["signature_exact"],
                    "signature_template": d["signature_template"],
                    "message_exact": d["message_exact"],
                    "message_template": d["message_template"],
                }
            )

    return pd.DataFrame(rows)


def build_episode_rows(diag_df: pd.DataFrame) -> pd.DataFrame:
    if diag_df.empty:
        return pd.DataFrame()

    # key -> iteration -> set(signatures)
    iter_exact: Dict[Tuple[str, str, int, str], Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))
    iter_templ: Dict[Tuple[str, str, int, str], Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))

    for r in diag_df.itertuples(index=False):
        key = (r.provider, r.model_label, int(r.prompt_id), r.family)
        iter_exact[key][int(r.iteration_index)].add(r.signature_exact)
        iter_templ[key][int(r.iteration_index)].add(r.signature_template)

    episodes: List[Dict[str, object]] = []
    for (provider, model_label, prompt_id, family), by_iter in iter_exact.items():
        iters = sorted(by_iter.keys())
        # split into contiguous blocks
        block_start = 0
        for i in range(1, len(iters) + 1):
            if i == len(iters) or iters[i] != iters[i - 1] + 1:
                block = iters[block_start:i]
                block_start = i

                exact_sets = [by_iter[k] for k in block]
                templ_sets = [iter_templ[(provider, model_label, prompt_id, family)][k] for k in block]

                start_exact = exact_sets[0]
                end_exact = exact_sets[-1]
                start_templ = templ_sets[0]
                end_templ = templ_sets[-1]

                all_exact = [s for st in exact_sets for s in st]
                all_templ = [s for st in templ_sets for s in st]

                c_exact = Counter(all_exact)
                c_templ = Counter(all_templ)
                exact_repeats = {k for k, v in c_exact.items() if v >= 2}
                templ_repeats = {k for k, v in c_templ.items() if v >= 2}

                # "same exact error persists" proxy:
                # at least one exact signature appears on >=2 iterations in this episode.
                has_exact_recurrence = len(exact_repeats) > 0
                has_template_recurrence = len(templ_repeats) > 0

                # Adjacent overlap captures uninterrupted persistence across iterations.
                adjacent_exact_overlap = False
                adjacent_template_overlap = False
                for j in range(len(block) - 1):
                    if exact_sets[j].intersection(exact_sets[j + 1]):
                        adjacent_exact_overlap = True
                    if templ_sets[j].intersection(templ_sets[j + 1]):
                        adjacent_template_overlap = True

                if has_exact_recurrence:
                    persistence_mode = "same_exact_error_recurs"
                elif has_template_recurrence:
                    persistence_mode = "same_error_template_recurs"
                else:
                    persistence_mode = "family_only_churn"

                episode = {
                    "provider": provider,
                    "model_label": model_label,
                    "prompt_id": int(prompt_id),
                    "family": family,
                    "start_iteration": int(block[0]),
                    "end_iteration_with_family": int(block[-1]),
                    "iterations_with_family": int(len(block)),
                    "additional_iterations_to_repair": int(len(block)),
                    "unique_exact_signatures": int(len(c_exact)),
                    "unique_template_signatures": int(len(c_templ)),
                    "has_exact_recurrence": bool(has_exact_recurrence),
                    "has_template_recurrence": bool(has_template_recurrence),
                    "adjacent_exact_overlap": bool(adjacent_exact_overlap),
                    "adjacent_template_overlap": bool(adjacent_template_overlap),
                    "start_end_exact_overlap": bool(len(start_exact.intersection(end_exact)) > 0),
                    "start_end_template_overlap": bool(len(start_templ.intersection(end_templ)) > 0),
                    "persistence_mode": persistence_mode,
                }
                episodes.append(episode)

    return pd.DataFrame(episodes)


def summarize_identity(episodes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if episodes.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Focus on non-trivial persistence where family appears for >=2 iterations.
    persisted = episodes[episodes["iterations_with_family"] >= 2].copy()

    overall = (
        persisted.groupby("persistence_mode")
        .agg(
            episodes=("persistence_mode", "count"),
            mean_additional_iterations=("additional_iterations_to_repair", "mean"),
        )
        .reset_index()
    )
    total = max(int(overall["episodes"].sum()), 1)
    overall["share_pct"] = 100.0 * overall["episodes"] / total

    by_family = (
        persisted.groupby(["family", "persistence_mode"])
        .size()
        .rename("episodes")
        .reset_index()
    )
    fam_tot = by_family.groupby("family")["episodes"].sum().rename("family_total").reset_index()
    by_family = by_family.merge(fam_tot, on="family", how="left")
    by_family["share_within_family_pct"] = 100.0 * by_family["episodes"] / by_family["family_total"].clip(lower=1)

    by_model = (
        persisted.groupby(["model_label", "persistence_mode"])
        .size()
        .rename("episodes")
        .reset_index()
    )
    mod_tot = by_model.groupby("model_label")["episodes"].sum().rename("model_total").reset_index()
    by_model = by_model.merge(mod_tot, on="model_label", how="left")
    by_model["share_within_model_pct"] = 100.0 * by_model["episodes"] / by_model["model_total"].clip(lower=1)

    return overall, by_family, by_model


def draw_overall_mode(overall: pd.DataFrame, out: Path) -> None:
    order = ["same_exact_error_recurs", "same_error_template_recurs", "family_only_churn"]
    label_map = {
        "same_exact_error_recurs": "Same exact diagnostic recurs",
        "same_error_template_recurs": "Same template, details changed",
        "family_only_churn": "Family-only churn",
    }
    color_map = {
        "same_exact_error_recurs": "#2E7D32",
        "same_error_template_recurs": "#F9A825",
        "family_only_churn": "#C62828",
    }

    s = overall.set_index("persistence_mode")
    vals = [float(s.loc[k, "episodes"]) if k in s.index else 0.0 for k in order]
    shares = [float(s.loc[k, "share_pct"]) if k in s.index else 0.0 for k in order]
    labels = [label_map[k] for k in order]
    colors = [color_map[k] for k in order]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=220)
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Episodes (family persisted >=2 iterations)", fontsize=9)
    ax.set_title("Do Persistent Errors Repeat as the Same Instance or Just Same Type?", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=12, labelsize=8)

    for b, v, p in zip(bars, vals, shares):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.5,
            f"{int(v)} ({p:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_family_mode(by_family: pd.DataFrame, out: Path, top_k: int = 12) -> None:
    mode_order = ["same_exact_error_recurs", "same_error_template_recurs", "family_only_churn"]
    color_map = {
        "same_exact_error_recurs": "#2E7D32",
        "same_error_template_recurs": "#F9A825",
        "family_only_churn": "#C62828",
    }
    label_map = {
        "same_exact_error_recurs": "Same exact",
        "same_error_template_recurs": "Same template",
        "family_only_churn": "Family churn",
    }

    totals = by_family.groupby("family")["episodes"].sum().sort_values(ascending=False).head(top_k)
    fams = totals.index.tolist()
    piv = (
        by_family[by_family["family"].isin(fams)]
        .pivot(index="family", columns="persistence_mode", values="share_within_family_pct")
        .fillna(0.0)
        .reindex(index=fams, columns=mode_order, fill_value=0.0)
    )

    y = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(10, 6), dpi=220)
    left = np.zeros(len(fams), dtype=float)
    for mode in mode_order:
        vals = piv[mode].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, color=color_map[mode], label=label_map[mode])
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(fams, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share within family (%)", fontsize=9)
    ax.set_title("Persistence Mode Mix by Error Family (Top Families by Persistent Episodes)", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def draw_model_mode(by_model: pd.DataFrame, out: Path) -> None:
    mode_order = ["same_exact_error_recurs", "same_error_template_recurs", "family_only_churn"]
    color_map = {
        "same_exact_error_recurs": "#2E7D32",
        "same_error_template_recurs": "#F9A825",
        "family_only_churn": "#C62828",
    }
    label_map = {
        "same_exact_error_recurs": "Same exact",
        "same_error_template_recurs": "Same template",
        "family_only_churn": "Family churn",
    }

    piv = (
        by_model.pivot(index="model_label", columns="persistence_mode", values="share_within_model_pct")
        .fillna(0.0)
        .reindex(index=MODEL_ORDER)
        .reindex(columns=mode_order, fill_value=0.0)
    )

    x = np.arange(len(piv.index))
    width = 0.23

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=220)
    for i, mode in enumerate(mode_order):
        vals = piv[mode].to_numpy(dtype=float)
        ax.bar(x + (i - 1) * width, vals, width=width, color=color_map[mode], label=label_map[mode])

    ax.set_xticks(x)
    ax.set_xticklabels(piv.index.tolist(), rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("Share within model (%)", fontsize=9)
    ax.set_title("Persistence Mode by Model", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_short_writeup(
    episodes: pd.DataFrame,
    overall: pd.DataFrame,
    by_family: pd.DataFrame,
    out_path: Path,
) -> None:
    persisted = episodes[episodes["iterations_with_family"] >= 2].copy()
    n_persisted = len(persisted)

    def pct(mode: str) -> float:
        sub = overall[overall["persistence_mode"] == mode]
        if sub.empty:
            return 0.0
        return float(sub.iloc[0]["share_pct"])

    top_families = (
        persisted.groupby("family")["prompt_id"]
        .count()
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )

    lines: List[str] = []
    lines.append("# Identity Persistence Analysis (Same Error vs Same Type)")
    lines.append("")
    lines.append(
        "Question: when an error family persists across iterations, is it the exact same diagnostic recurring, or only the same family label with changing diagnostics?"
    )
    lines.append("")
    lines.append("## Core Result")
    lines.append("")
    lines.append(f"- Persistent family episodes analyzed (`iterations_with_family >= 2`): **{n_persisted}**.")
    lines.append(
        f"- **Same exact diagnostic recurs:** {pct('same_exact_error_recurs'):.1f}% of persistent episodes."
    )
    lines.append(
        f"- **Same template recurs (details changed):** {pct('same_error_template_recurs'):.1f}% of persistent episodes."
    )
    lines.append(
        f"- **Family-only churn (no recurring exact/template diagnostic):** {pct('family_only_churn'):.1f}% of persistent episodes."
    )
    lines.append("")
    lines.append("Interpretation: persistent burden is mostly repeated concrete diagnostics, not just category-level churn.")
    lines.append("")
    lines.append("Top families by persistent episodes (for detailed breakdown):")
    for fam in top_families:
        sub = by_family[by_family["family"] == fam]
        exact = float(sub.loc[sub["persistence_mode"] == "same_exact_error_recurs", "share_within_family_pct"].sum())
        templ = float(sub.loc[sub["persistence_mode"] == "same_error_template_recurs", "share_within_family_pct"].sum())
        churn = float(sub.loc[sub["persistence_mode"] == "family_only_churn", "share_within_family_pct"].sum())
        n = int(sub["family_total"].max()) if not sub.empty else 0
        lines.append(
            f"- `{fam}` (n={n}): exact {exact:.1f}%, template {templ:.1f}%, churn {churn:.1f}%."
        )
    lines.append("")
    lines.append("## Figure Map")
    lines.append("")
    lines.append("- `figures/figE08_identity_persistence_overall.png`")
    lines.append("- `figures/figE09_identity_persistence_by_family_top12.png`")
    lines.append("- `figures/figE10_identity_persistence_by_model.png`")
    lines.append("")
    lines.append("## Data Map")
    lines.append("")
    lines.append("- `data/error_diagnostic_instances.csv`")
    lines.append("- `data/error_identity_episodes.csv`")
    lines.append("- `data/error_identity_summary_overall.csv`")
    lines.append("- `data/error_identity_summary_by_family.csv`")
    lines.append("- `data/error_identity_summary_by_model.csv`")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_root = args.output_root.resolve()
    data_dir = out_root / "data"
    fig_dir = out_root / "figures"
    ensure_dir(data_dir)
    ensure_dir(fig_dir)

    diag_df = load_iteration_diagnostics(repo_root)
    episodes = build_episode_rows(diag_df)
    overall, by_family, by_model = summarize_identity(episodes)

    diag_df.to_csv(data_dir / "error_diagnostic_instances.csv", index=False)
    episodes.to_csv(data_dir / "error_identity_episodes.csv", index=False)
    overall.to_csv(data_dir / "error_identity_summary_overall.csv", index=False)
    by_family.to_csv(data_dir / "error_identity_summary_by_family.csv", index=False)
    by_model.to_csv(data_dir / "error_identity_summary_by_model.csv", index=False)

    if not overall.empty:
        draw_overall_mode(overall, fig_dir / "figE08_identity_persistence_overall.png")
    if not by_family.empty:
        draw_family_mode(by_family, fig_dir / "figE09_identity_persistence_by_family_top12.png")
    if not by_model.empty:
        draw_model_mode(by_model, fig_dir / "figE10_identity_persistence_by_model.png")

    write_short_writeup(
        episodes=episodes,
        overall=overall,
        by_family=by_family,
        out_path=out_root / "identity_persistence_writeup.md",
    )

    print(f"[ok] diagnostics: {len(diag_df)}")
    print(f"[ok] episodes: {len(episodes)}")
    print(f"[ok] overall rows: {len(overall)}")
    print(f"[ok] family rows: {len(by_family)}")
    print(f"[ok] model rows: {len(by_model)}")


if __name__ == "__main__":
    main()

