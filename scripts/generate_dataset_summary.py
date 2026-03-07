#!/usr/bin/env python3
"""Generate trajectory-level dataset summary from persisted refine artifacts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ERROR_FAMILY_RE = re.compile(r"\berror \(([^)]+)\):")

DEFAULT_MODEL_ROOTS = [
    "Generated_from_Prompts_API_LOOP_OPENAI",
    "Generated_from_Prompts_API_LOOP_ANTHROPIC",
    "Generated_from_Prompts_API_LOOP_DEEPSEEK_REASONER",
    "Generated_from_Prompts_API_LOOP_MISTRAL_LARGE",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize released trajectory-level SysML artifacts.")
    p.add_argument(
        "--api-loop-root",
        type=Path,
        default=Path("api_loop"),
        help="Path to api_loop root (default: ./api_loop)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("paper/results/data/dataset_summary.json"),
        help="Output JSON path (default: paper/results/data/dataset_summary.json)",
    )
    p.add_argument(
        "--prompt-min",
        type=int,
        default=1,
        help="Minimum prompt id inclusive (default: 1)",
    )
    p.add_argument(
        "--prompt-max",
        type=int,
        default=151,
        help="Maximum prompt id inclusive (default: 151)",
    )
    return p.parse_args()


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def resolve_run_log_path(case_dir: Path, manifest: Dict[str, Any]) -> Optional[Path]:
    raw = manifest.get("run_log_path")
    candidates: List[Path] = []

    if isinstance(raw, str) and raw.strip():
        candidates.append(Path(raw))

    for k in ("selected_run_dir", "selected_run_path"):
        v = manifest.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(Path(v) / "run_log.json")

    for cand in candidates:
        if cand.is_absolute() and cand.exists():
            return cand
        rel = (case_dir / cand).resolve()
        if rel.exists():
            return rel

    refine_runs_dir = case_dir / "refine_runs"
    if refine_runs_dir.exists():
        logs = sorted(refine_runs_dir.glob("*/run_log.json"), key=lambda x: x.parent.name)
        if logs:
            return logs[-1]

    return None


def parse_error_count(*texts: str) -> int:
    merged = "\n".join(t or "" for t in texts)
    merged = ANSI_RE.sub("", merged)
    return len(ERROR_FAMILY_RE.findall(merged))


def mean_or_none(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def median_or_none(values: List[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def min_or_none(values: List[float]) -> Optional[float]:
    return float(min(values)) if values else None


def max_or_none(values: List[float]) -> Optional[float]:
    return float(max(values)) if values else None


def model_name_from_root(root: Path) -> str:
    n = root.name
    if n.endswith("_OPENAI"):
        return "OpenAI Codex 5.2"
    if n.endswith("_ANTHROPIC"):
        return "Anthropic Sonnet 4.6"
    if n.endswith("_DEEPSEEK_REASONER"):
        return "DeepSeek Reasoner"
    if n.endswith("_MISTRAL_LARGE"):
        return "Mistral Large"
    return n


def main() -> None:
    args = parse_args()
    api_root = args.api_loop_root.resolve()
    model_roots = [api_root / d for d in DEFAULT_MODEL_ROOTS]

    missing_models = [str(m) for m in model_roots if not m.exists()]
    if missing_models:
        raise SystemExit(f"Missing model roots: {missing_models}")

    prompt_ids = list(range(args.prompt_min, args.prompt_max + 1))

    # Aggregate counters
    trajectories_total = 0
    prompt_count_unique = len(prompt_ids)
    backend_count = len(model_roots)

    initial_candidate_artifacts_stored = 0
    final_accepted_artifacts_stored = 0

    artifacts_total = 0
    artifacts_stored = 0
    artifacts_positive = 0
    artifacts_negative = 0

    initial_error_cases_gt0 = 0
    initial_error_cases_eq0 = 0
    initial_failing_cases = 0

    errors_total_all_artifacts = 0
    errors_total_failing_artifacts = 0
    initial_error_total = 0

    iterations_per_case: List[int] = []
    iterations_to_acceptance: List[int] = []
    failing_error_counts: List[int] = []
    all_error_counts: List[int] = []

    iter_runtime_present = 0
    iter_token_present = 0
    traj_loop_runtime_present = 0
    traj_loop_token_present = 0

    invalid_cases: List[Dict[str, Any]] = []

    for model_root in model_roots:
        for prompt_id in prompt_ids:
            case_dir = model_root / str(prompt_id)
            manifest_path = case_dir / f"{prompt_id}_refine_manifest.json"
            manifest = read_json(manifest_path)
            if not isinstance(manifest, dict):
                invalid_cases.append({"model_root": str(model_root), "prompt_id": prompt_id, "reason": "missing_or_invalid_manifest"})
                continue

            run_log_path = resolve_run_log_path(case_dir, manifest)
            if run_log_path is None or not run_log_path.exists():
                invalid_cases.append({"model_root": str(model_root), "prompt_id": prompt_id, "reason": "missing_run_log"})
                continue

            run_log = read_json(run_log_path)
            if not isinstance(run_log, list) or not run_log:
                invalid_cases.append({"model_root": str(model_root), "prompt_id": prompt_id, "reason": "empty_or_invalid_run_log"})
                continue

            trajectories_total += 1
            if manifest.get("loop_duration_seconds") is not None:
                traj_loop_runtime_present += 1
            if manifest.get("tokens_used_total") is not None:
                traj_loop_token_present += 1

            iterations_per_case.append(len(run_log))

            # initial candidate (k=0) is iteration_01 artifact
            initial_path = run_log_path.parent / "iteration_01.sysml"
            if initial_path.exists():
                initial_candidate_artifacts_stored += 1

            final_path = case_dir / f"{prompt_id}.sysml"
            if final_path.exists():
                final_accepted_artifacts_stored += 1

            first_step: Optional[Dict[str, Any]] = None
            success_iters: List[int] = []

            for step in run_log:
                if not isinstance(step, dict):
                    continue

                it = step.get("iteration")
                try:
                    it_i = int(it)
                except Exception:
                    it_i = None

                if it_i == 1 and first_step is None:
                    first_step = step

                if step.get("iteration_duration_seconds") is not None:
                    iter_runtime_present += 1

                tok = step.get("tokens_used_this_iter")
                if isinstance(tok, dict) and any(tok.get(k) is not None for k in ("input_tokens", "output_tokens", "total_tokens")):
                    iter_token_present += 1

                artifacts_total += 1
                iter_path = run_log_path.parent / f"iteration_{int(step.get('iteration', 0)):02d}.sysml"
                if iter_path.exists():
                    artifacts_stored += 1

                ec = parse_error_count(step.get("compiler_stdout") or "", step.get("compiler_stderr") or "")
                all_error_counts.append(ec)
                errors_total_all_artifacts += ec

                if bool(step.get("success", False)):
                    artifacts_positive += 1
                    if it_i is not None:
                        success_iters.append(it_i)
                else:
                    artifacts_negative += 1
                    failing_error_counts.append(ec)
                    errors_total_failing_artifacts += ec

            if success_iters:
                iterations_to_acceptance.append(min(success_iters))

            if isinstance(first_step, dict):
                first_ec = parse_error_count(first_step.get("compiler_stdout") or "", first_step.get("compiler_stderr") or "")
                initial_error_total += first_ec
                if first_ec > 0:
                    initial_error_cases_gt0 += 1
                else:
                    initial_error_cases_eq0 += 1
                if not bool(first_step.get("success", False)):
                    initial_failing_cases += 1

    summary: Dict[str, Any] = {
        "artifact_source": {
            "api_loop_root": str(api_root),
            "model_roots": [str(p) for p in model_roots],
            "prompt_id_min": args.prompt_min,
            "prompt_id_max": args.prompt_max,
        },
        "release_scope": {
            "prompts": prompt_count_unique,
            "model_backends": backend_count,
            "prompt_model_cases": trajectories_total,
        },
        "storage_presence": {
            "initial_candidate_artifacts_stored": initial_candidate_artifacts_stored,
            "final_accepted_artifacts_stored": final_accepted_artifacts_stored,
            "iteration_artifacts_total": artifacts_total,
            "iteration_artifacts_stored": artifacts_stored,
            "iteration_runtime_metadata_present": iter_runtime_present,
            "iteration_token_metadata_present": iter_token_present,
            "trajectory_runtime_metadata_present": traj_loop_runtime_present,
            "trajectory_token_metadata_present": traj_loop_token_present,
        },
        "artifact_level_outcomes": {
            "positive_artifacts": artifacts_positive,
            "negative_artifacts": artifacts_negative,
            "positive_to_negative_ratio": (artifacts_positive / artifacts_negative) if artifacts_negative else None,
            "repair_transitions_total": artifacts_total - trajectories_total,
        },
        "validator_error_burden": {
            "total_validator_errors_all_artifacts": errors_total_all_artifacts,
            "total_validator_errors_failing_artifacts": errors_total_failing_artifacts,
            "failing_artifact_error_min": min_or_none([float(v) for v in failing_error_counts]),
            "failing_artifact_error_mean": mean_or_none([float(v) for v in failing_error_counts]),
            "failing_artifact_error_median": median_or_none([float(v) for v in failing_error_counts]),
            "failing_artifact_error_max": max_or_none([float(v) for v in failing_error_counts]),
            "max_errors_single_artifact": max_or_none([float(v) for v in all_error_counts]),
        },
        "trajectory_depth": {
            "max_iterations_observed": max_or_none([float(v) for v in iterations_per_case]),
            "mean_iterations_per_trajectory": mean_or_none([float(v) for v in iterations_per_case]),
            "median_iterations_per_trajectory": median_or_none([float(v) for v in iterations_per_case]),
            "mean_iterations_to_acceptance": mean_or_none([float(v) for v in iterations_to_acceptance]),
            "median_iterations_to_acceptance": median_or_none([float(v) for v in iterations_to_acceptance]),
        },
        "initial_candidate_checks": {
            "cases_with_initial_error_count_gt0": initial_error_cases_gt0,
            "cases_with_initial_error_count_eq0": initial_error_cases_eq0,
            "initial_failing_cases": initial_failing_cases,
            "initial_error_total": initial_error_total,
            "mean_initial_error_count_per_failing_case": (initial_error_total / initial_failing_cases) if initial_failing_cases else None,
        },
        "invalid_cases": invalid_cases,
        "invalid_case_count": len(invalid_cases),
        "notes": [
            "Error counts are parsed from validator diagnostics using lines that match 'error (<family>):'.",
            "Initial candidate corresponds to k=0 and is stored as iteration_01.sysml.",
            "Positive artifacts are run-log steps with success=true; negative artifacts have success=false.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"[ok] wrote {args.output}")
    print("\nDataset summary")
    print("---------------")
    print(f"Prompt-model cases:          {summary['release_scope']['prompt_model_cases']}")
    print(f"Total iteration artifacts:   {summary['storage_presence']['iteration_artifacts_total']}")
    print(f"Positive artifacts:          {summary['artifact_level_outcomes']['positive_artifacts']}")
    print(f"Negative artifacts:          {summary['artifact_level_outcomes']['negative_artifacts']}")
    print(f"Total validator errors:      {summary['validator_error_burden']['total_validator_errors_all_artifacts']}")
    print(f"Mean errors/failing artifact:{summary['validator_error_burden']['failing_artifact_error_mean']:.3f}")
    print(f"Median errors/failing art.:  {summary['validator_error_burden']['failing_artifact_error_median']:.3f}")
    print(f"Max errors in one artifact:  {int(summary['validator_error_burden']['max_errors_single_artifact'] or 0)}")
    print(f"Max iterations observed:     {int(summary['trajectory_depth']['max_iterations_observed'] or 0)}")


if __name__ == "__main__":
    main()
