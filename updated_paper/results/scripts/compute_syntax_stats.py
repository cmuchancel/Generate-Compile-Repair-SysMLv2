#!/usr/bin/env python3
"""Compute deterministic syntax-only statistics from extracted campaign CSVs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-data-dir",
        type=Path,
        default=repo_root / "paper" / "results" / "data",
    )
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=repo_root / "paper" / "results" / "data",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260220)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    return parser.parse_args()


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    return s.astype(str).str.strip().str.lower().map(mapping)


def wilson_ci(count: int, nobs: int, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    if nobs <= 0:
        return None, None
    lo, hi = proportion_confint(count, nobs, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_resamples: int = 10000) -> Tuple[Optional[float], Optional[float]]:
    if values.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = float(np.mean(sample))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def safe_pct(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return 100.0 * num / den


def _json_load_dict(text: Any) -> Dict[str, int]:
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        obj = json.loads(text)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, int] = {}
    for k, v in obj.items():
        try:
            out[str(k)] = int(v)
        except Exception:
            continue
    return out


@dataclass
class GroupStats:
    provider: str
    model: str
    total_prompts: int
    first_shot_pass_count: int
    first_shot_fail_count: int
    eventual_pass_count: int
    unresolved_count: int
    recovered_count: int
    first_shot_pass_rate_pct: Optional[float]
    first_shot_fail_rate_pct: Optional[float]
    eventual_pass_rate_pct: Optional[float]
    unresolved_rate_pct: Optional[float]
    absolute_gain_pp: Optional[float]
    relative_gain_pct: Optional[float]
    first_shot_pass_ci95_low_pct: Optional[float]
    first_shot_pass_ci95_high_pct: Optional[float]
    eventual_pass_ci95_low_pct: Optional[float]
    eventual_pass_ci95_high_pct: Optional[float]
    unresolved_ci95_low_pct: Optional[float]
    unresolved_ci95_high_pct: Optional[float]
    iterations_to_success_mean: Optional[float]
    iterations_to_success_median: Optional[float]
    iterations_to_success_std: Optional[float]
    iterations_to_success_max: Optional[float]
    iterations_to_success_q25: Optional[float]
    iterations_to_success_q75: Optional[float]
    iterations_to_success_bootstrap_ci95_low: Optional[float]
    iterations_to_success_bootstrap_ci95_high: Optional[float]
    mcnemar_b: int
    mcnemar_c: int
    mcnemar_statistic: Optional[float]
    mcnemar_pvalue: Optional[float]


def compute_group_stats(
    sub: pd.DataFrame,
    provider: str,
    model: str,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> GroupStats:
    total = int(len(sub))
    fs = to_bool_series(sub["first_shot_pass"]).fillna(False)
    ev = to_bool_series(sub["eventual_pass"]).fillna(False)

    first_pass = int(fs.sum())
    first_fail = total - first_pass
    eventual_pass = int(ev.sum())
    unresolved = total - eventual_pass
    recovered = int(((~fs) & ev).sum())

    first_rate = safe_pct(first_pass, total)
    first_fail_rate = safe_pct(first_fail, total)
    eventual_rate = safe_pct(eventual_pass, total)
    unresolved_rate = safe_pct(unresolved, total)
    abs_gain = (eventual_rate - first_rate) if (eventual_rate is not None and first_rate is not None) else None
    rel_gain = safe_pct((eventual_pass - first_pass), first_pass) if first_pass > 0 else None

    fs_lo, fs_hi = wilson_ci(first_pass, total)
    ev_lo, ev_hi = wilson_ci(eventual_pass, total)
    un_lo, un_hi = wilson_ci(unresolved, total)

    iters = pd.to_numeric(sub["iterations_to_success"], errors="coerce").dropna().to_numpy(dtype=float)
    if iters.size:
        mean_iters = float(np.mean(iters))
        median_iters = float(np.median(iters))
        std_iters = float(np.std(iters, ddof=1)) if iters.size > 1 else 0.0
        max_iters = float(np.max(iters))
        q25 = float(np.quantile(iters, 0.25))
        q75 = float(np.quantile(iters, 0.75))
        boot_lo, boot_hi = bootstrap_mean_ci(iters, bootstrap_seed, bootstrap_resamples)
    else:
        mean_iters = median_iters = std_iters = max_iters = q25 = q75 = None
        boot_lo = boot_hi = None

    # McNemar contingency table
    a = int((fs & ev).sum())
    b = int((fs & (~ev)).sum())
    c = int(((~fs) & ev).sum())
    d = int(((~fs) & (~ev)).sum())
    table = [[a, b], [c, d]]

    try:
        mres = mcnemar(table, exact=((b + c) <= 25), correction=True)
        m_stat = float(mres.statistic) if mres.statistic is not None else None
        m_p = float(mres.pvalue) if mres.pvalue is not None else None
    except Exception:
        m_stat = None
        m_p = None

    return GroupStats(
        provider=provider,
        model=model,
        total_prompts=total,
        first_shot_pass_count=first_pass,
        first_shot_fail_count=first_fail,
        eventual_pass_count=eventual_pass,
        unresolved_count=unresolved,
        recovered_count=recovered,
        first_shot_pass_rate_pct=first_rate,
        first_shot_fail_rate_pct=first_fail_rate,
        eventual_pass_rate_pct=eventual_rate,
        unresolved_rate_pct=unresolved_rate,
        absolute_gain_pp=abs_gain,
        relative_gain_pct=rel_gain,
        first_shot_pass_ci95_low_pct=safe_pct(fs_lo, 1.0) if fs_lo is not None else None,
        first_shot_pass_ci95_high_pct=safe_pct(fs_hi, 1.0) if fs_hi is not None else None,
        eventual_pass_ci95_low_pct=safe_pct(ev_lo, 1.0) if ev_lo is not None else None,
        eventual_pass_ci95_high_pct=safe_pct(ev_hi, 1.0) if ev_hi is not None else None,
        unresolved_ci95_low_pct=safe_pct(un_lo, 1.0) if un_lo is not None else None,
        unresolved_ci95_high_pct=safe_pct(un_hi, 1.0) if un_hi is not None else None,
        iterations_to_success_mean=mean_iters,
        iterations_to_success_median=median_iters,
        iterations_to_success_std=std_iters,
        iterations_to_success_max=max_iters,
        iterations_to_success_q25=q25,
        iterations_to_success_q75=q75,
        iterations_to_success_bootstrap_ci95_low=boot_lo,
        iterations_to_success_bootstrap_ci95_high=boot_hi,
        mcnemar_b=b,
        mcnemar_c=c,
        mcnemar_statistic=m_stat,
        mcnemar_pvalue=m_p,
    )


def explode_error_rows(iter_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in iter_df.to_dict(orient="records"):
        fam = _json_load_dict(rec.get("error_families_json"))
        if not fam:
            continue
        for k, v in fam.items():
            rows.append(
                {
                    "provider": rec["provider"],
                    "model": rec["model"],
                    "prompt_id": int(rec["prompt_id"]),
                    "iteration_index": int(rec["iteration_index"]),
                    "error_family": k,
                    "count": int(v),
                }
            )
    return pd.DataFrame(rows)


def summarize_error_taxonomy(iter_df: pd.DataFrame) -> pd.DataFrame:
    exploded = explode_error_rows(iter_df)
    if exploded.empty:
        cols = [
            "provider",
            "model",
            "scope",
            "error_family",
            "error_count",
            "prompts_affected",
            "iteration_rows_with_family",
        ]
        return pd.DataFrame(columns=cols)

    out_rows: List[Dict[str, Any]] = []

    groups = exploded.groupby(["provider", "model"], dropna=False)
    for (provider, model), g in groups:
        # scope: all iterations
        agg_all = (
            g.groupby("error_family")
            .agg(
                error_count=("count", "sum"),
                prompts_affected=("prompt_id", "nunique"),
                iteration_rows_with_family=("iteration_index", "count"),
            )
            .reset_index()
        )
        for rec in agg_all.to_dict(orient="records"):
            rec.update({"provider": provider, "model": model, "scope": "all_iterations"})
            out_rows.append(rec)

        # scope: first failed iteration per prompt
        first_failed_idx = (
            g[g["count"] > 0]
            .sort_values(["prompt_id", "iteration_index"])
            .groupby("prompt_id", as_index=False)
            .first()[["prompt_id", "iteration_index"]]
        )
        g_first_failed = g.merge(first_failed_idx, on=["prompt_id", "iteration_index"], how="inner")
        agg_first_failed = (
            g_first_failed.groupby("error_family")
            .agg(
                error_count=("count", "sum"),
                prompts_affected=("prompt_id", "nunique"),
                iteration_rows_with_family=("iteration_index", "count"),
            )
            .reset_index()
        )
        for rec in agg_first_failed.to_dict(orient="records"):
            rec.update({"provider": provider, "model": model, "scope": "first_failed_iteration"})
            out_rows.append(rec)

        # scope: first iteration only
        g_first_iter = g[g["iteration_index"] == 1]
        agg_first_iter = (
            g_first_iter.groupby("error_family")
            .agg(
                error_count=("count", "sum"),
                prompts_affected=("prompt_id", "nunique"),
                iteration_rows_with_family=("iteration_index", "count"),
            )
            .reset_index()
        )
        for rec in agg_first_iter.to_dict(orient="records"):
            rec.update({"provider": provider, "model": model, "scope": "first_iteration"})
            out_rows.append(rec)

    summary = pd.DataFrame(out_rows)

    # Add pooled ALL/ALL for each scope
    pooled_rows: List[Dict[str, Any]] = []
    for scope in sorted(summary["scope"].unique()):
        sub = summary[summary["scope"] == scope]
        pooled = (
            sub.groupby("error_family", as_index=False)
            .agg(
                error_count=("error_count", "sum"),
                prompts_affected=("prompts_affected", "sum"),
                iteration_rows_with_family=("iteration_rows_with_family", "sum"),
            )
        )
        for rec in pooled.to_dict(orient="records"):
            rec.update({"provider": "ALL", "model": "ALL", "scope": scope})
            pooled_rows.append(rec)

    summary = pd.concat([summary, pd.DataFrame(pooled_rows)], ignore_index=True)
    summary = summary.sort_values(["provider", "model", "scope", "error_count", "error_family"], ascending=[True, True, True, False, True])
    return summary


def main() -> None:
    args = parse_args()
    input_dir = args.input_data_dir.resolve()
    out_dir = args.output_data_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_csv = input_dir / "prompt_level_syntax_metrics.csv"
    iter_csv = input_dir / "iteration_level_syntax_metrics.csv"
    if not prompt_csv.exists() or not iter_csv.exists():
        raise SystemExit("Missing extracted CSVs. Run extract_syntax_metrics.py first.")

    prompt_df = pd.read_csv(prompt_csv)
    iter_df = pd.read_csv(iter_csv)

    # Normalize core types
    prompt_df["prompt_id"] = pd.to_numeric(prompt_df["prompt_id"], errors="coerce").astype("Int64")
    prompt_df = prompt_df.dropna(subset=["prompt_id"]).copy()
    prompt_df["prompt_id"] = prompt_df["prompt_id"].astype(int)

    model_stats: List[GroupStats] = []

    for (provider, model), sub in prompt_df.groupby(["provider", "model"], dropna=False):
        model_stats.append(
            compute_group_stats(
                sub.copy(),
                str(provider),
                str(model),
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_resamples=args.bootstrap_resamples,
            )
        )

    model_stats.append(
        compute_group_stats(
            prompt_df.copy(),
            provider="ALL",
            model="ALL",
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    )

    model_summary_df = pd.DataFrame([asdict(s) for s in model_stats])
    model_summary_df = model_summary_df.sort_values(["provider", "model"], ascending=[True, True])
    model_summary_path = out_dir / "model_level_syntax_summary.csv"
    model_summary_df.to_csv(model_summary_path, index=False)

    error_summary_df = summarize_error_taxonomy(iter_df)
    error_summary_path = out_dir / "error_taxonomy_summary.csv"
    error_summary_df.to_csv(error_summary_path, index=False)

    overall = model_summary_df[(model_summary_df["provider"] == "ALL") & (model_summary_df["model"] == "ALL")].iloc[0].to_dict()

    per_model = []
    for _, row in model_summary_df.iterrows():
        if row["provider"] == "ALL" and row["model"] == "ALL":
            continue
        per_model.append(row.to_dict())

    stat_tests = {
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "overall": overall,
        "per_model": per_model,
    }
    stat_tests_path = out_dir / "stat_tests.json"
    stat_tests_path.write_text(json.dumps(stat_tests, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] wrote {model_summary_path}")
    print(f"[ok] wrote {error_summary_path}")
    print(f"[ok] wrote {stat_tests_path}")


if __name__ == "__main__":
    main()
