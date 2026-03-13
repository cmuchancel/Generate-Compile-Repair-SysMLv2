#!/usr/bin/env python3
"""Extract deterministic syntax-only metrics from API loop run artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ERROR_FAMILY_RE = re.compile(r"\berror \(([^)]+)\):")
WARNING_RE = re.compile(r"\bwarning \(([^)]+)\):")
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")

MODEL_ROOT_PREFIX = "Generated_from_Prompts_API_LOOP_"
PROMPT_IDS = list(range(1, 152))


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--api-loop-root", type=Path, default=repo_root / "api_loop")
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=repo_root / "paper" / "results" / "data",
    )
    parser.add_argument(
        "--include-model-roots",
        nargs="*",
        default=[],
        help=(
            "Optional explicit list of model-root directory names under api_loop. "
            "If omitted, all Generated_from_Prompts_API_LOOP_* roots are used."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sanitize_compiler_text(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text or "")


def parse_error_families(*compiler_texts: str) -> Tuple[int, Dict[str, int], int, Dict[str, int]]:
    merged = "\n".join(sanitize_compiler_text(t) for t in compiler_texts if t)
    error_families = Counter(ERROR_FAMILY_RE.findall(merged))
    warning_families = Counter(WARNING_RE.findall(merged))
    return sum(error_families.values()), dict(error_families), sum(warning_families.values()), dict(warning_families)


def discover_model_roots(api_loop_root: Path, include_names: Iterable[str]) -> List[Path]:
    include = {name.strip() for name in include_names if name.strip()}
    roots: List[Path] = []
    for child in sorted(api_loop_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if not child.name.startswith(MODEL_ROOT_PREFIX):
            continue
        if include and child.name not in include:
            continue
        roots.append(child)
    return roots


def infer_provider_model(model_root: Path) -> Tuple[str, str]:
    suffix = model_root.name[len(MODEL_ROOT_PREFIX) :]
    inferred_provider = suffix.lower()
    inferred_model = suffix.lower()

    for prompt_id in PROMPT_IDS:
        manifest_path = model_root / str(prompt_id) / f"{prompt_id}_refine_manifest.json"
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            continue

        run_log_path = resolve_run_log_path(model_root, prompt_id, manifest)
        run_meta = None
        run_log = None
        if run_log_path and run_log_path.exists():
            run_meta = read_json(run_log_path.parent / "run_meta.json")
            run_log = read_json(run_log_path)

        provider = None
        model = None
        if isinstance(run_meta, dict):
            provider = run_meta.get("provider")
            model = run_meta.get("model")

        if (not provider or not model) and isinstance(run_log, list) and run_log:
            first = run_log[0] if isinstance(run_log[0], dict) else {}
            provider = provider or first.get("provider")
            model = model or first.get("model")

        provider = str(provider).strip() if provider else None
        model = str(model).strip() if model else None

        if provider and model:
            return provider, model
        if provider:
            return provider, inferred_model
        if model:
            return inferred_provider, model

    return inferred_provider, inferred_model


def resolve_run_log_path(model_root: Path, prompt_id: int, manifest: Dict[str, Any]) -> Optional[Path]:
    candidates: List[Path] = []

    run_log_path_raw = manifest.get("run_log_path")
    if isinstance(run_log_path_raw, str) and run_log_path_raw.strip():
        candidates.append(Path(run_log_path_raw))

    for key in ("archived_run_dir", "run_dir"):
        val = manifest.get(key)
        if isinstance(val, str) and val.strip():
            candidates.append(Path(val) / "run_log.json")

    case_dir = model_root / str(prompt_id)
    refine_runs_dir = case_dir / "refine_runs"
    if refine_runs_dir.exists():
        for p in sorted(refine_runs_dir.glob("*/run_log.json"), key=lambda x: x.parent.name):
            candidates.append(p)

    for path in candidates:
        if path.exists():
            return path

    return None


def get_git_commit(repo_root: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        return None
    return None


def json_dumps_sorted(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def parse_iteration_tokens(step: Dict[str, Any], run_log_dir: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    token_obj = step.get("tokens_used_this_iter")
    if isinstance(token_obj, dict):
        ti = token_obj.get("input_tokens")
        to = token_obj.get("output_tokens")
        tt = token_obj.get("total_tokens")
        if ti is not None or to is not None or tt is not None:
            return _to_int(ti), _to_int(to), _to_int(tt)

    response_path_raw = step.get("response_path")
    response_path = None
    if isinstance(response_path_raw, str) and response_path_raw.strip():
        response_path = Path(response_path_raw)
        if not response_path.exists():
            response_path = run_log_dir / response_path.name
    if response_path and response_path.exists():
        payload = read_json(response_path)
        if isinstance(payload, dict):
            usage = payload.get("usage")
            if isinstance(usage, dict):
                ti = usage.get("input_tokens") or usage.get("prompt_tokens")
                to = usage.get("output_tokens") or usage.get("completion_tokens")
                tt = usage.get("total_tokens")
                if tt is None and (ti is not None and to is not None):
                    try:
                        tt = int(ti) + int(to)
                    except Exception:
                        tt = None
                return _to_int(ti), _to_int(to), _to_int(tt)
    return None, None, None


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    api_loop_root = args.api_loop_root.resolve()
    output_data_dir = args.output_data_dir.resolve()
    output_data_dir.mkdir(parents=True, exist_ok=True)

    model_roots = discover_model_roots(api_loop_root, args.include_model_roots)
    if not model_roots:
        raise SystemExit("No model roots found under api_loop.")

    prompt_rows: List[Dict[str, Any]] = []
    iteration_rows: List[Dict[str, Any]] = []

    campaign_manifest: Dict[str, Any] = {
        "repo_root": str(repo_root),
        "api_loop_root": str(api_loop_root),
        "model_roots": [],
        "assumptions": [
            "One selected run per prompt is identified by <id>_refine_manifest.json when present.",
            "Error counts are derived from compiler text lines matching 'error (<family>):'.",
            "Warnings are recorded separately and do not affect pass/fail metrics.",
            "Costs are left null unless explicit pricing metadata is provided (none detected).",
        ],
        "git_commit": get_git_commit(repo_root),
    }

    for model_root in model_roots:
        provider, model = infer_provider_model(model_root)
        model_entry: Dict[str, Any] = {
            "model_root": str(model_root),
            "provider": provider,
            "model": model,
            "missing_manifests": [],
            "invalid_manifests": [],
            "missing_run_logs": [],
            "invalid_run_logs": [],
            "processed_prompts": 0,
        }

        for prompt_id in PROMPT_IDS:
            case_dir = model_root / str(prompt_id)
            manifest_path = case_dir / f"{prompt_id}_refine_manifest.json"
            if not manifest_path.exists():
                model_entry["missing_manifests"].append(prompt_id)
                continue

            manifest = read_json(manifest_path)
            if not isinstance(manifest, dict):
                model_entry["invalid_manifests"].append(prompt_id)
                continue

            run_log_path = resolve_run_log_path(model_root, prompt_id, manifest)
            if not run_log_path or not run_log_path.exists():
                model_entry["missing_run_logs"].append(prompt_id)
                continue

            run_log = read_json(run_log_path)
            if not isinstance(run_log, list) or not run_log:
                model_entry["invalid_run_logs"].append(prompt_id)
                continue

            run_meta = read_json(run_log_path.parent / "run_meta.json")
            if not isinstance(run_meta, dict):
                run_meta = {}

            norm_steps: List[Dict[str, Any]] = []
            for step in run_log:
                if not isinstance(step, dict):
                    continue
                iteration_index = _to_int(step.get("iteration"))
                if iteration_index is None:
                    continue

                compiler_stdout = str(step.get("compiler_stdout") or "")
                compiler_stderr = str(step.get("compiler_stderr") or "")
                error_count, error_families, warning_count, warning_families = parse_error_families(
                    compiler_stdout,
                    compiler_stderr,
                )
                tokens_in, tokens_out, tokens_total = parse_iteration_tokens(step, run_log_path.parent)

                iter_row = {
                    "provider": provider,
                    "model": model,
                    "prompt_id": prompt_id,
                    "iteration_index": iteration_index,
                    "pass_at_iteration": bool(step.get("success", False)),
                    "error_count": error_count,
                    "error_families_json": json_dumps_sorted(error_families),
                    "warning_count": warning_count,
                    "warning_families_json": json_dumps_sorted(warning_families),
                    "iteration_time_sec": _to_float(step.get("iteration_duration_seconds")),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "tokens_total": tokens_total,
                    "return_code": _to_int(step.get("return_code")),
                    "run_id": run_log_path.parent.name,
                    "session_id": run_log_path.parent.name,
                    "source_path": str(manifest_path),
                }
                iteration_rows.append(iter_row)
                norm_steps.append(iter_row)

            if not norm_steps:
                model_entry["invalid_run_logs"].append(prompt_id)
                continue

            norm_steps.sort(key=lambda r: int(r["iteration_index"]))
            first = norm_steps[0]
            last = norm_steps[-1]
            start_iter_raw = int(first["iteration_index"])
            end_iter_raw = int(last["iteration_index"])
            is_resumed_segment = start_iter_raw > 1
            first_success = bool(first["pass_at_iteration"])
            eventual_success = bool(manifest.get("final_iteration_success", last["pass_at_iteration"]))
            success_iters = [int(r["iteration_index"]) for r in norm_steps if bool(r["pass_at_iteration"])]
            iters_to_success = success_iters[0] if success_iters else None

            tokens_in_total = sum((r["tokens_in"] or 0) for r in norm_steps)
            tokens_out_total = sum((r["tokens_out"] or 0) for r in norm_steps)
            tokens_total = sum((r["tokens_total"] or 0) for r in norm_steps)
            if not tokens_total:
                tokens_total = _to_int(manifest.get("tokens_used_total")) or _to_int(run_meta.get("tokens_used_total")) or 0

            wall_time_sec = (
                _to_float(manifest.get("loop_duration_seconds"))
                or _to_float(run_meta.get("run_duration_seconds"))
                or sum((r["iteration_time_sec"] or 0.0) for r in norm_steps)
            )

            # If a persisted run starts at iteration > 1, it is a resumed segment.
            # Full-prompt wall time/token totals are not reconstructible from this segment alone.
            if is_resumed_segment:
                wall_time_sec = None
                tokens_in_total = None
                tokens_out_total = None
                tokens_total = None

            prompt_rows.append(
                {
                    "provider": provider,
                    "model": model,
                    "prompt_id": prompt_id,
                    "first_shot_pass": first_success,
                    "eventual_pass": eventual_success,
                    "iterations_run": len(norm_steps),
                    "iterations_to_success": iters_to_success,
                    "unresolved_within_cap": not eventual_success,
                    "first_iteration_error_count": int(first["error_count"]),
                    "final_error_count": int(last["error_count"]),
                    "total_error_count_across_iterations": int(sum(int(r["error_count"]) for r in norm_steps)),
                    "first_failed_then_recovered": (not first_success) and eventual_success,
                    "wall_time_sec": wall_time_sec,
                    "token_input": tokens_in_total if tokens_in_total is not None else None,
                    "token_output": tokens_out_total if tokens_out_total is not None else None,
                    "token_total": tokens_total if tokens_total is not None else None,
                    "estimated_cost_usd": None,
                    "run_start_iteration": start_iter_raw,
                    "run_end_iteration": end_iter_raw,
                    "is_resumed_segment": is_resumed_segment,
                    "run_id": run_log_path.parent.name,
                    "session_id": run_log_path.parent.name,
                    "source_path": str(manifest_path),
                }
            )
            model_entry["processed_prompts"] += 1

        campaign_manifest["model_roots"].append(model_entry)

    prompt_rows.sort(key=lambda r: (r["provider"], r["model"], int(r["prompt_id"])))
    iteration_rows.sort(
        key=lambda r: (
            r["provider"],
            r["model"],
            int(r["prompt_id"]),
            int(r["iteration_index"]),
        )
    )

    prompt_cols = [
        "provider",
        "model",
        "prompt_id",
        "first_shot_pass",
        "eventual_pass",
        "iterations_run",
        "iterations_to_success",
        "unresolved_within_cap",
        "first_iteration_error_count",
        "final_error_count",
        "total_error_count_across_iterations",
        "first_failed_then_recovered",
        "wall_time_sec",
        "token_input",
        "token_output",
        "token_total",
        "estimated_cost_usd",
        "run_start_iteration",
        "run_end_iteration",
        "is_resumed_segment",
        "run_id",
        "session_id",
        "source_path",
    ]
    iter_cols = [
        "provider",
        "model",
        "prompt_id",
        "iteration_index",
        "pass_at_iteration",
        "error_count",
        "error_families_json",
        "iteration_time_sec",
        "tokens_in",
        "tokens_out",
        "tokens_total",
        "return_code",
        "warning_count",
        "warning_families_json",
        "run_id",
        "session_id",
        "source_path",
    ]

    prompt_csv = output_data_dir / "prompt_level_syntax_metrics.csv"
    iter_csv = output_data_dir / "iteration_level_syntax_metrics.csv"

    def write_csv(path: Path, cols: List[str], rows: List[Dict[str, Any]]) -> None:
        import csv

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c) for c in cols})

    write_csv(prompt_csv, prompt_cols, prompt_rows)
    write_csv(iter_csv, iter_cols, iteration_rows)

    campaign_manifest["outputs"] = {
        "prompt_level_csv": str(prompt_csv),
        "iteration_level_csv": str(iter_csv),
    }
    campaign_manifest_path = output_data_dir / "campaign_manifest.json"
    campaign_manifest_path.write_text(
        json.dumps(campaign_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[ok] wrote {prompt_csv}")
    print(f"[ok] wrote {iter_csv}")
    print(f"[ok] wrote {campaign_manifest_path}")
    print(f"[summary] prompt rows={len(prompt_rows)} iteration rows={len(iteration_rows)}")


if __name__ == "__main__":
    main()
