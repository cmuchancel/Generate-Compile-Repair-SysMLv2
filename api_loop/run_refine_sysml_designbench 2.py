#!/usr/bin/env python3
"""Batch runner for refine_sysml.py over SysMBench NL prompts (1..151)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
UPSTREAM_ROOT = SCRIPT_DIR.parent / "sysmbench_original_upstream"
DONE_LOG_RE = re.compile(r"\[done\] run details saved to (.+run_log\.json)")
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MISTRAL_LARGE_MODEL = "mistral-large-latest"
DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts-root",
        type=Path,
        default=SCRIPT_DIR / "nl_prompts",
        help="Root containing per-ID prompt folders with nl.txt.",
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=UPSTREAM_ROOT / "dataset" / "sysml" / "samples",
        help="Root containing ground-truth sample folders (01..151).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "Generated_from_Prompts_API_LOOP",
        help="Evaluation-ready output root.",
    )
    parser.add_argument(
        "--refine-runs-root",
        type=Path,
        default=SCRIPT_DIR / "runs" / "designbench_refine_api_loop",
        help="Raw refine_sysml run artifacts root.",
    )
    parser.add_argument(
        "--refine-script",
        type=Path,
        default=SCRIPT_DIR / "refine_sysml.py",
        help="Path to refine_sysml.py.",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=SCRIPT_DIR.parent / ".venv",
        help="Virtual environment root passed to refine_sysml.py --venv.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=SCRIPT_DIR.parent / ".env",
        help="Optional env file to load into subprocess env (e.g., OPENAI_API_KEY / ANTHROPIC_API_KEY).",
    )
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=151)
    parser.add_argument("--skip", type=int, nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="How many IDs to process concurrently inside each batch.",
    )
    parser.add_argument(
        "--id-retries",
        type=int,
        default=4,
        help="How many times to retry an ID if refine_sysml exits with failure.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "deepseek_reasoner", "mistral_large"),
        default="openai",
        help="LLM API provider to pass through to refine_sysml.py.",
    )
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-total-tokens", type=int, default=50000)
    parser.add_argument("--api-max-retries", type=int, default=8)
    parser.add_argument("--api-retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--api-retry-max-backoff-seconds", type=float, default=30.0)
    parser.add_argument("--api-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--anthropic-max-output-tokens",
        type=int,
        default=8192,
        help="Pass-through max output tokens for Anthropic calls.",
    )
    parser.add_argument(
        "--deepseek-base-url",
        default=DEFAULT_DEEPSEEK_BASE_URL,
        help="Pass-through DeepSeek OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--mistral-base-url",
        default=DEFAULT_MISTRAL_BASE_URL,
        help="Pass-through Mistral OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--syside-timeout-seconds",
        type=int,
        default=60,
        help="Timeout forwarded to refine_sysml.py for each syside call.",
    )
    parser.add_argument(
        "--syside-validate-with",
        choices=("format", "check"),
        default="format",
        help="Validation mode forwarded to refine_sysml.py.",
    )
    parser.add_argument(
        "--example",
        type=Path,
        default=None,
        help="Optional SysML example snippet path forwarded to refine_sysml.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate even if <id>.sysml already exists in output.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any ID fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forward dry-run to refine_sysml.py and skip API/compiler calls.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.parallelism <= 0:
        raise SystemExit("--parallelism must be > 0")
    if args.id_retries < 0:
        raise SystemExit("--id-retries must be >= 0")
    if args.start_id > args.end_id:
        raise SystemExit("--start-id must be <= --end-id")
    return args


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_venv_python(venv_root: Path) -> Path:
    python_path = venv_root / "bin" / "python"
    if not python_path.exists():
        raise FileNotFoundError(f"Could not find venv python at {python_path}")
    return python_path


def load_env_file(env_path: Path, env: Dict[str, str]) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if key == "OPENAI_API_KEY_BACKUP":
            # Never propagate backup keys to subprocess environments.
            continue
        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        if key in {
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
            "SILICONFLOW_API_KEY",
            "MISTRAL_API_KEY",
        }:
            # Force use of provider API keys from env file for reproducible runs.
            env[key] = value
        else:
            env.setdefault(key, value)


def discover_prompt_ids(prompts_root: Path) -> List[int]:
    ids: List[int] = []
    for child in sorted(prompts_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or not child.name.isdigit():
            continue
        if (child / "nl.txt").exists():
            ids.append(int(child.name))
    return sorted(ids)


def select_ids(all_ids: Sequence[int], start_id: int, end_id: int, skip_ids: Sequence[int]) -> List[int]:
    skip_set = set(skip_ids)
    return [i for i in all_ids if start_id <= i <= end_id and i not in skip_set]


def copy_groundtruth(samples_root: Path, model_id: int, case_dir: Path) -> Optional[Path]:
    candidates = [
        samples_root / f"{model_id:02d}" / "design.sysml",
        samples_root / str(model_id) / "design.sysml",
    ]
    for src in candidates:
        if src.exists():
            dest = case_dir / f"{model_id}_groundtruth.sysml"
            shutil.copy2(src, dest)
            return dest
    return None


def find_newest_run_dir(raw_runs_dir: Path, before: Sequence[str]) -> Optional[Path]:
    if not raw_runs_dir.exists():
        return None
    before_set = set(before)
    dirs = [p for p in raw_runs_dir.iterdir() if p.is_dir()]
    new_dirs = [p for p in dirs if p.name not in before_set]
    if new_dirs:
        return max(new_dirs, key=lambda p: p.stat().st_mtime)
    if dirs:
        return max(dirs, key=lambda p: p.stat().st_mtime)
    return None


def parse_run_log_path(stdout: str) -> Optional[Path]:
    for line in reversed(stdout.splitlines()):
        match = DONE_LOG_RE.search(line)
        if match:
            return Path(match.group(1).strip())
    return None


def has_success_manifest(case_dir: Path, model_id: int) -> bool:
    manifest_path = case_dir / f"{model_id}_refine_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        manifest.get("status") == "ok"
        and bool(manifest.get("final_iteration_success", False))
    )


def run_refine_for_id(
    args: argparse.Namespace,
    model_id: int,
    base_env: Dict[str, str],
) -> Dict[str, object]:
    prompt_path = args.prompts_root / str(model_id) / "nl.txt"
    case_dir = args.output_root / str(model_id)
    raw_runs_dir = args.refine_runs_root / str(model_id)
    final_sysml_path = case_dir / f"{model_id}.sysml"

    ensure_dir(case_dir)
    ensure_dir(raw_runs_dir)
    shutil.copy2(prompt_path, case_dir / "nl.txt")
    groundtruth_path = copy_groundtruth(args.samples_root, model_id, case_dir)

    if final_sysml_path.exists() and not args.overwrite and has_success_manifest(case_dir, model_id):
        return {
            "model_id": model_id,
            "status": "skipped",
            "reason": f"{final_sysml_path} already exists with successful manifest",
            "generated_path": str(final_sysml_path),
            "groundtruth_path": str(groundtruth_path) if groundtruth_path else None,
        }

    before_dirs = [p.name for p in raw_runs_dir.iterdir() if p.is_dir()]
    runner_python = resolve_venv_python(args.venv)

    cmd: List[str] = [
        str(runner_python),
        str(args.refine_script),
        "--provider",
        str(args.provider),
        "--input",
        str(prompt_path),
        "--output-dir",
        str(raw_runs_dir),
        "--model",
        str(args.model),
        "--max-iters",
        str(args.max_iters),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--venv",
        str(args.venv),
        "--syside-timeout-seconds",
        str(args.syside_timeout_seconds),
        "--syside-validate-with",
        str(args.syside_validate_with),
        "--api-max-retries",
        str(args.api_max_retries),
        "--api-retry-backoff-seconds",
        str(args.api_retry_backoff_seconds),
        "--api-retry-max-backoff-seconds",
        str(args.api_retry_max_backoff_seconds),
        "--api-timeout-seconds",
        str(args.api_timeout_seconds),
        "--anthropic-max-output-tokens",
        str(args.anthropic_max_output_tokens),
        "--deepseek-base-url",
        str(args.deepseek_base_url),
        "--mistral-base-url",
        str(args.mistral_base_url),
    ]
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.example is not None:
        cmd.extend(["--example", str(args.example)])
    if args.dry_run:
        cmd.append("--dry-run")

    loop_start_utc = utc_now()
    loop_start_wall = perf_counter()

    proc = subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        env=base_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    stdout_path = case_dir / f"{model_id}_refine_stdout.log"
    stderr_path = case_dir / f"{model_id}_refine_stderr.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    loop_end_utc = utc_now()
    loop_duration_seconds = perf_counter() - loop_start_wall

    run_log_path = parse_run_log_path(proc.stdout)
    run_dir = run_log_path.parent if run_log_path else find_newest_run_dir(raw_runs_dir, before_dirs)
    if run_dir:
        run_log_path = run_dir / "run_log.json"

    if proc.returncode != 0:
        return {
            "model_id": model_id,
            "status": "failed",
            "reason": f"refine_sysml.py exited with code {proc.returncode}",
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "run_dir": str(run_dir) if run_dir else None,
            "loop_start_utc": iso_utc(loop_start_utc),
            "loop_end_utc": iso_utc(loop_end_utc),
            "loop_duration_seconds": loop_duration_seconds,
        }

    if not run_log_path or not run_log_path.exists():
        return {
            "model_id": model_id,
            "status": "failed",
            "reason": "could not locate run_log.json from refine_sysml output",
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "run_dir": str(run_dir) if run_dir else None,
            "loop_start_utc": iso_utc(loop_start_utc),
            "loop_end_utc": iso_utc(loop_end_utc),
            "loop_duration_seconds": loop_duration_seconds,
        }

    try:
        run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "model_id": model_id,
            "status": "failed",
            "reason": f"invalid run_log.json: {exc}",
            "run_log_path": str(run_log_path),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "loop_start_utc": iso_utc(loop_start_utc),
            "loop_end_utc": iso_utc(loop_end_utc),
            "loop_duration_seconds": loop_duration_seconds,
        }

    if not isinstance(run_log, list) or not run_log:
        return {
            "model_id": model_id,
            "status": "failed",
            "reason": "run_log.json is empty",
            "run_log_path": str(run_log_path),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "loop_start_utc": iso_utc(loop_start_utc),
            "loop_end_utc": iso_utc(loop_end_utc),
            "loop_duration_seconds": loop_duration_seconds,
        }

    last_step = run_log[-1]
    final_candidate = Path(str(last_step.get("sysml_path", "")))
    if not final_candidate.exists():
        return {
            "model_id": model_id,
            "status": "failed",
            "reason": f"final sysml path missing: {final_candidate}",
            "run_log_path": str(run_log_path),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "loop_start_utc": iso_utc(loop_start_utc),
            "loop_end_utc": iso_utc(loop_end_utc),
            "loop_duration_seconds": loop_duration_seconds,
        }

    shutil.copy2(final_candidate, final_sysml_path)

    archived_run_dir = None
    if run_dir and run_dir.exists():
        archive_root = case_dir / "refine_runs"
        ensure_dir(archive_root)
        archived_run_dir = archive_root / run_dir.name
        if archived_run_dir.exists():
            shutil.rmtree(archived_run_dir)
        shutil.copytree(run_dir, archived_run_dir)

    any_iteration_success = any(bool(step.get("success", False)) for step in run_log)

    iteration_timings: List[Dict[str, object]] = []
    for step in run_log:
        token_obj = step.get("tokens_used_this_iter") or {}
        if not isinstance(token_obj, dict):
            token_obj = {}
        iteration_timings.append(
            {
                "iteration": int(step.get("iteration", 0)),
                "iteration_start": step.get("iteration_start"),
                "iteration_end": step.get("iteration_end"),
                "iteration_duration_seconds": step.get("iteration_duration_seconds"),
                "success": bool(step.get("success", False)),
                "return_code": step.get("return_code"),
                "tokens_used_this_iter_total": token_obj.get("total_tokens"),
                "tokens_used_total": step.get("tokens_used_total"),
            }
        )

    if not any_iteration_success:
        return {
            "model_id": model_id,
            "status": "failed",
            "reason": "no iteration passed syside validation",
            "prompt_path": str(prompt_path),
            "generated_path": str(final_sysml_path),
            "groundtruth_path": str(groundtruth_path) if groundtruth_path else None,
            "run_dir": str(run_dir) if run_dir else None,
            "archived_run_dir": str(archived_run_dir) if archived_run_dir else None,
            "run_log_path": str(run_log_path),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "iterations_completed": len(run_log),
            "final_iteration_success": False,
            "tokens_used_total": int(last_step.get("tokens_used_total", 0)),
            "loop_start_utc": iso_utc(loop_start_utc),
            "loop_end_utc": iso_utc(loop_end_utc),
            "loop_duration_seconds": loop_duration_seconds,
            "iteration_timings": iteration_timings,
        }

    model_manifest = {
        "model_id": model_id,
        "status": "ok",
        "prompt_path": str(prompt_path),
        "generated_path": str(final_sysml_path),
        "groundtruth_path": str(groundtruth_path) if groundtruth_path else None,
        "run_dir": str(run_dir) if run_dir else None,
        "archived_run_dir": str(archived_run_dir) if archived_run_dir else None,
        "run_log_path": str(run_log_path),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "iterations_completed": len(run_log),
        "final_iteration_success": bool(last_step.get("success", False)),
        "tokens_used_total": int(last_step.get("tokens_used_total", 0)),
        "loop_start_utc": iso_utc(loop_start_utc),
        "loop_end_utc": iso_utc(loop_end_utc),
        "loop_duration_seconds": loop_duration_seconds,
        "iteration_timings": iteration_timings,
    }
    (case_dir / f"{model_id}_refine_manifest.json").write_text(
        json.dumps(model_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return model_manifest


def run_refine_for_id_with_retries(
    args: argparse.Namespace,
    model_id: int,
    base_env: Dict[str, str],
) -> Dict[str, object]:
    last_result: Optional[Dict[str, object]] = None
    for attempt in range(1, args.id_retries + 2):
        result = run_refine_for_id(args, model_id, base_env)
        result["attempt"] = attempt
        if result.get("status") != "failed":
            result["attempts_used"] = attempt
            return result
        last_result = result
        print(
            f"[model {model_id}] attempt {attempt}/{args.id_retries + 1} failed: "
            f"{result.get('reason')}"
        )
    assert last_result is not None
    last_result["attempts_used"] = args.id_retries + 1
    return last_result


def write_timing_csvs(
    session_output_dir: Path,
    session_id: str,
    results: Sequence[Dict[str, object]],
) -> None:
    loop_csv = session_output_dir / f"_refine_designbench_loop_timings_{session_id}.csv"
    with loop_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "batch_index",
                "status",
                "loop_start_utc",
                "loop_end_utc",
                "loop_duration_seconds",
                "iterations_completed",
                "tokens_used_total",
                "final_iteration_success",
                "reason",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "model_id": r.get("model_id"),
                    "batch_index": r.get("batch_index"),
                    "status": r.get("status"),
                    "loop_start_utc": r.get("loop_start_utc"),
                    "loop_end_utc": r.get("loop_end_utc"),
                    "loop_duration_seconds": r.get("loop_duration_seconds"),
                    "iterations_completed": r.get("iterations_completed"),
                    "tokens_used_total": r.get("tokens_used_total"),
                    "final_iteration_success": r.get("final_iteration_success"),
                    "reason": r.get("reason"),
                }
            )

    iter_csv = session_output_dir / f"_refine_designbench_iteration_timings_{session_id}.csv"
    with iter_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "batch_index",
                "iteration",
                "iteration_start",
                "iteration_end",
                "iteration_duration_seconds",
                "success",
                "return_code",
                "tokens_used_this_iter_total",
                "tokens_used_total",
            ],
        )
        writer.writeheader()
        for r in results:
            for step in (r.get("iteration_timings") or []):
                writer.writerow(
                    {
                        "model_id": r.get("model_id"),
                        "batch_index": r.get("batch_index"),
                        "iteration": step.get("iteration"),
                        "iteration_start": step.get("iteration_start"),
                        "iteration_end": step.get("iteration_end"),
                        "iteration_duration_seconds": step.get("iteration_duration_seconds"),
                        "success": step.get("success"),
                        "return_code": step.get("return_code"),
                        "tokens_used_this_iter_total": step.get("tokens_used_this_iter_total"),
                        "tokens_used_total": step.get("tokens_used_total"),
                    }
                )


def write_session_manifest(
    session_output_dir: Path,
    session_id: str,
    args: argparse.Namespace,
    selected_ids: Sequence[int],
    results: Sequence[Dict[str, object]],
) -> Path:
    summary = {
        "session_id": session_id,
        "timestamp_utc": iso_utc(utc_now()),
        "provider": args.provider,
        "model": args.model,
        "batch_size": args.batch_size,
        "parallelism": args.parallelism,
        "id_retries": args.id_retries,
        "start_id": args.start_id,
        "end_id": args.end_id,
        "skip": args.skip,
        "max_iters": args.max_iters,
        "max_total_tokens": args.max_total_tokens,
        "temperature": args.temperature,
        "syside_validate_with": args.syside_validate_with,
        "api_max_retries": args.api_max_retries,
        "api_retry_backoff_seconds": args.api_retry_backoff_seconds,
        "api_retry_max_backoff_seconds": args.api_retry_max_backoff_seconds,
        "api_timeout_seconds": args.api_timeout_seconds,
        "anthropic_max_output_tokens": args.anthropic_max_output_tokens,
        "deepseek_base_url": args.deepseek_base_url,
        "mistral_base_url": args.mistral_base_url,
        "prompts_root": str(args.prompts_root),
        "output_root": str(args.output_root),
        "refine_runs_root": str(args.refine_runs_root),
        "total_selected": len(selected_ids),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "results": list(results),
    }
    manifest_path = session_output_dir / f"_refine_designbench_session_{session_id}.json"
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_timing_csvs(session_output_dir, session_id, results)
    return manifest_path


def main() -> None:
    args = parse_args()
    if args.provider == "anthropic" and args.model == DEFAULT_OPENAI_MODEL:
        args.model = DEFAULT_ANTHROPIC_MODEL
        print(
            f"[config] provider=anthropic and model not set; defaulting to {args.model}"
        )
    if args.provider == "deepseek_reasoner" and args.model == DEFAULT_OPENAI_MODEL:
        args.model = DEFAULT_DEEPSEEK_REASONER_MODEL
        print(
            f"[config] provider=deepseek_reasoner and model not set; defaulting to {args.model}"
        )
    if args.provider == "mistral_large" and args.model == DEFAULT_OPENAI_MODEL:
        args.model = DEFAULT_MISTRAL_LARGE_MODEL
        print(
            f"[config] provider=mistral_large and model not set; defaulting to {args.model}"
        )
    args.prompts_root = (SCRIPT_DIR / args.prompts_root).resolve()
    args.samples_root = (SCRIPT_DIR / args.samples_root).resolve()
    args.output_root = (SCRIPT_DIR / args.output_root).resolve()
    args.refine_runs_root = (SCRIPT_DIR / args.refine_runs_root).resolve()
    args.refine_script = (SCRIPT_DIR / args.refine_script).resolve()
    args.venv = (SCRIPT_DIR / args.venv).resolve()
    if args.example is not None:
        args.example = (SCRIPT_DIR / args.example).resolve()
    args.env_file = (SCRIPT_DIR / args.env_file).resolve()

    if not args.prompts_root.exists():
        raise SystemExit(f"Prompts root does not exist: {args.prompts_root}")
    if not args.refine_script.exists():
        raise SystemExit(f"refine_sysml.py not found: {args.refine_script}")
    if not args.venv.exists():
        raise SystemExit(f"venv path not found: {args.venv}")

    all_ids = discover_prompt_ids(args.prompts_root)
    selected_ids = select_ids(all_ids, args.start_id, args.end_id, args.skip)
    if not selected_ids:
        raise SystemExit("No prompt IDs matched selection.")

    ensure_dir(args.output_root)
    ensure_dir(args.refine_runs_root)

    base_env = os.environ.copy()
    load_env_file(args.env_file, base_env)

    session_id = utc_now().strftime("%Y%m%d-%H%M%S")
    session_output_dir = args.output_root / "_refine_sessions" / session_id
    ensure_dir(session_output_dir)
    results: List[Dict[str, object]] = []
    total = len(selected_ids)
    batches = [selected_ids[i : i + args.batch_size] for i in range(0, total, args.batch_size)]

    print(f"[start] selected {total} IDs from {selected_ids[0]} to {selected_ids[-1]}")
    print(
        f"[start] running in {len(batches)} batch(es) with batch size {args.batch_size} "
        f"and parallelism {args.parallelism}"
    )
    for batch_index, batch_ids in enumerate(batches, start=1):
        print(
            f"[batch {batch_index}/{len(batches)}] ids {batch_ids[0]}..{batch_ids[-1]} "
            f"(count={len(batch_ids)})"
        )
        stop_requested = False
        if args.parallelism == 1 or len(batch_ids) == 1:
            for index_in_batch, model_id in enumerate(batch_ids, start=1):
                print(
                    f"[batch {batch_index}/{len(batches)}] "
                    f"{index_in_batch}/{len(batch_ids)} model {model_id}"
                )
                result = run_refine_for_id_with_retries(args, model_id, base_env)
                result["batch_index"] = batch_index
                results.append(result)
                manifest_path = write_session_manifest(
                    session_output_dir, session_id, args, selected_ids, results
                )
                print(
                    f"[progress] completed {len(results)}/{total}; "
                    f"manifest updated: {manifest_path}"
                )
                print(f"[model {model_id}] {result.get('status')}")
                if result.get("status") == "failed" and args.stop_on_error:
                    stop_requested = True
                    break
        else:
            max_workers = min(args.parallelism, len(batch_ids))
            print(f"[batch {batch_index}/{len(batches)}] parallel workers={max_workers}")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_model = {
                    executor.submit(
                        run_refine_for_id_with_retries, args, model_id, base_env
                    ): model_id
                    for model_id in batch_ids
                }
                completed_in_batch = 0
                for future in as_completed(future_to_model):
                    model_id = future_to_model[future]
                    completed_in_batch += 1
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "model_id": model_id,
                            "status": "failed",
                            "reason": f"runner exception: {exc}",
                        }
                    result["batch_index"] = batch_index
                    results.append(result)
                    manifest_path = write_session_manifest(
                        session_output_dir, session_id, args, selected_ids, results
                    )
                    print(
                        f"[progress] completed {len(results)}/{total}; "
                        f"manifest updated: {manifest_path}"
                    )
                    print(
                        f"[batch {batch_index}/{len(batches)}] "
                        f"{completed_in_batch}/{len(batch_ids)} model {model_id} "
                        f"-> {result.get('status')}"
                    )
                    if result.get("status") == "failed" and args.stop_on_error:
                        stop_requested = True
            if stop_requested and args.stop_on_error:
                manifest_path = write_session_manifest(
                    session_output_dir, session_id, args, selected_ids, results
                )
                raise SystemExit(
                    f"Stopped on error in batch {batch_index}. Session manifest: {manifest_path}"
                )
        manifest_path = write_session_manifest(
            session_output_dir, session_id, args, selected_ids, results
        )
        print(f"[batch {batch_index}/{len(batches)}] manifest updated: {manifest_path}")
        if stop_requested and args.stop_on_error:
            raise SystemExit(
                f"Stopped on error in batch {batch_index}. Session manifest: {manifest_path}"
            )

    manifest_path = write_session_manifest(
        session_output_dir, session_id, args, selected_ids, results
    )
    ok = sum(1 for r in results if r.get("status") == "ok")
    failed = sum(1 for r in results if r.get("status") == "failed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    print(f"[done] ok={ok} failed={failed} skipped={skipped}")
    print(f"[done] session manifest: {manifest_path}")


if __name__ == "__main__":
    main()
