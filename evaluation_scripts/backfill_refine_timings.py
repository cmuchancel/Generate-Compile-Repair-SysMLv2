#!/usr/bin/env python3
"""Backfill timing CSVs from existing refine run logs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    default_output = detect_default_output_root()
    p.add_argument(
        "--output-root",
        type=Path,
        default=default_output,
        help="Root containing per-ID output folders (default: %(default)s).",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "api_loop" / "runs" / "designbench_refine_api_loop",
        help="Root containing raw per-ID refine run folders (default: %(default)s).",
    )
    p.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Output directory for backfilled CSV/JSON. Defaults to "
        "<output-root>/_refine_sessions/backfill_<timestamp>.",
    )
    return p.parse_args()


def detect_default_output_root() -> Path:
    candidates = [
        REPO_ROOT / "api_loop" / "Generated_from_Prompts_API_LOOP_OPENAI",
        REPO_ROOT / "ai_agent" / "Generated_from_Prompts_AI_AGENT",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_run_log_files(output_root: Path, runs_root: Path) -> List[Path]:
    seen: Set[Path] = set()
    run_logs: List[Path] = []

    # Archived per-case runs copied under output root.
    for path in output_root.glob("[0-9]*/refine_runs/*/run_log.json"):
        rp = path.resolve()
        if rp not in seen:
            seen.add(rp)
            run_logs.append(rp)

    # Raw runs root.
    if runs_root.exists():
        for path in runs_root.glob("[0-9]*/[0-9]*-*/run_log.json"):
            rp = path.resolve()
            if rp not in seen:
                seen.add(rp)
                run_logs.append(rp)
        # Some runs may use plain timestamp folder names.
        for path in runs_root.glob("[0-9]*/[0-9]*/run_log.json"):
            rp = path.resolve()
            if rp not in seen:
                seen.add(rp)
                run_logs.append(rp)

    return run_logs


def read_json(path: Path) -> Optional[object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_model_id_from_run_log(path: Path) -> Optional[int]:
    # Expected: .../<id>/<run_id>/run_log.json
    parent = path.parent.parent.name
    if parent.isdigit():
        return int(parent)
    return None


def parse_run(
    run_log_path: Path,
) -> Tuple[Optional[Dict[str, object]], List[Dict[str, object]]]:
    payload = read_json(run_log_path)
    if not isinstance(payload, list) or not payload:
        return None, []

    model_id = parse_model_id_from_run_log(run_log_path)
    run_id = run_log_path.parent.name
    run_meta_path = run_log_path.parent / "run_meta.json"
    meta_obj = read_json(run_meta_path)
    meta = meta_obj if isinstance(meta_obj, dict) else {}

    first = payload[0] if isinstance(payload[0], dict) else {}
    last = payload[-1] if isinstance(payload[-1], dict) else {}

    run_start = meta.get("run_start") or first.get("iteration_start")
    run_end = meta.get("run_end") or last.get("iteration_end")
    run_duration_seconds = meta.get("run_duration_seconds")
    if run_duration_seconds is None:
        run_duration_seconds = sum(
            float(step.get("iteration_duration_seconds") or 0.0)
            for step in payload
            if isinstance(step, dict)
        )

    loop_row: Dict[str, object] = {
        "model_id": model_id,
        "run_id": run_id,
        "run_start": run_start,
        "run_end": run_end,
        "run_duration_seconds": run_duration_seconds,
        "iterations_completed": len(payload),
        "any_iteration_success": any(
            bool(step.get("success", False)) for step in payload if isinstance(step, dict)
        ),
        "final_iteration_success": bool(last.get("success", False)),
        "final_return_code": last.get("return_code"),
        "tokens_used_total": last.get("tokens_used_total"),
        "run_log_path": str(run_log_path),
    }

    iter_rows: List[Dict[str, object]] = []
    for step in payload:
        if not isinstance(step, dict):
            continue
        token_obj = step.get("tokens_used_this_iter") or {}
        if not isinstance(token_obj, dict):
            token_obj = {}
        iter_rows.append(
            {
                "model_id": model_id,
                "run_id": run_id,
                "iteration": step.get("iteration"),
                "iteration_start": step.get("iteration_start"),
                "iteration_end": step.get("iteration_end"),
                "iteration_duration_seconds": step.get("iteration_duration_seconds"),
                "success": step.get("success"),
                "return_code": step.get("return_code"),
                "tokens_used_this_iter_total": token_obj.get("total_tokens"),
                "tokens_used_total": step.get("tokens_used_total"),
                "sysml_path": step.get("sysml_path"),
                "run_log_path": str(run_log_path),
            }
        )

    return loop_row, iter_rows


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    runs_root = args.runs_root.resolve()

    if args.destination is None:
        destination = output_root / "_refine_sessions" / f"backfill_{utc_now_str()}"
    else:
        destination = args.destination.resolve()
    ensure_dir(destination)

    run_log_files = find_run_log_files(output_root, runs_root)
    loop_rows: List[Dict[str, object]] = []
    iter_rows: List[Dict[str, object]] = []

    for run_log in run_log_files:
        loop_row, run_iter_rows = parse_run(run_log)
        if loop_row is None:
            continue
        loop_rows.append(loop_row)
        iter_rows.extend(run_iter_rows)

    loop_rows.sort(key=lambda r: (r.get("model_id") or 0, str(r.get("run_id") or "")))
    iter_rows.sort(
        key=lambda r: (
            r.get("model_id") or 0,
            str(r.get("run_id") or ""),
            r.get("iteration") or 0,
        )
    )

    stamp = utc_now_str()
    loop_csv = destination / f"_refine_backfill_loop_timings_{stamp}.csv"
    iter_csv = destination / f"_refine_backfill_iteration_timings_{stamp}.csv"
    summary_json = destination / f"_refine_backfill_summary_{stamp}.json"

    write_csv(
        loop_csv,
        [
            "model_id",
            "run_id",
            "run_start",
            "run_end",
            "run_duration_seconds",
            "iterations_completed",
            "any_iteration_success",
            "final_iteration_success",
            "final_return_code",
            "tokens_used_total",
            "run_log_path",
        ],
        loop_rows,
    )
    write_csv(
        iter_csv,
        [
            "model_id",
            "run_id",
            "iteration",
            "iteration_start",
            "iteration_end",
            "iteration_duration_seconds",
            "success",
            "return_code",
            "tokens_used_this_iter_total",
            "tokens_used_total",
            "sysml_path",
            "run_log_path",
        ],
        iter_rows,
    )

    model_ids = sorted({int(r["model_id"]) for r in loop_rows if isinstance(r.get("model_id"), int)})
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "output_root": str(output_root),
        "runs_root": str(runs_root),
        "destination": str(destination),
        "run_logs_found": len(run_log_files),
        "loop_rows": len(loop_rows),
        "iteration_rows": len(iter_rows),
        "model_ids_count": len(model_ids),
        "model_ids_min": model_ids[0] if model_ids else None,
        "model_ids_max": model_ids[-1] if model_ids else None,
        "loop_csv": str(loop_csv),
        "iteration_csv": str(iter_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] loop csv: {loop_csv}")
    print(f"[done] iteration csv: {iter_csv}")
    print(f"[done] summary: {summary_json}")


if __name__ == "__main__":
    main()
