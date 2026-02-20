#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Set

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


@dataclass(frozen=True)
class CheckEntry:
    model_id: int
    generated_path: Path
    source_path: Path
    source_kind: str
    manifest_status: str


def detect_default_output_root() -> Path:
    candidates = [
        REPO_ROOT / "ai_agent" / "Generated_from_Prompts_AI_AGENT",
        REPO_ROOT / "api_loop" / "Generated_from_Prompts_API_LOOP",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Second-pass verifier: run syside checks against generated SysML files. "
            "Supports API-loop manifests and direct per-ID SysML directories."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=detect_default_output_root(),
        help="Root directory containing per-ID outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "manifest", "direct"),
        default="auto",
        help=(
            "Entry discovery mode. `manifest` reads *_refine_manifest.json, "
            "`direct` scans per-ID directories, `auto` picks manifest if available."
        ),
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=None,
        help=(
            "Optional virtual environment root used to resolve syside. "
            "If omitted, uses current interpreter with python -m syside."
        ),
    )
    parser.add_argument(
        "--validate-with",
        choices=("check", "format"),
        default="check",
        help="syside subcommand to run for verification (default: check).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Timeout per syside invocation.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Number of concurrent workers (default: 4).",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default="",
        help='Optional ID filter, e.g. "1,2,5-10".',
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help=(
            "Where to write the verification JSON report. "
            "Default: <output-root>/second_pass_check_summary.json"
        ),
    )
    return parser.parse_args()


def parse_id_filter(raw: str) -> Optional[Set[int]]:
    raw = raw.strip()
    if not raw:
        return None
    selected: Set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            parts = chunk.split("-", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid ID range: {chunk}")
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            if end < start:
                raise ValueError(f"Invalid ID range (end < start): {chunk}")
            for item in range(start, end + 1):
                selected.add(item)
        else:
            selected.add(int(chunk))
    return selected


def resolve_python_executable(venv_root: Optional[Path]) -> Path:
    if venv_root is None:
        return Path(sys.executable).resolve()
    python_path = (venv_root.resolve() / "bin" / "python")
    if not python_path.exists():
        raise FileNotFoundError(
            f"Could not find python at {python_path}. "
            "Pass --venv pointing to a valid virtual environment."
        )
    return python_path


def resolve_syside_command(python_path: Path, venv_root: Optional[Path]) -> List[str]:
    if venv_root is not None:
        syside_cli = venv_root.resolve() / "bin" / "syside"
        if syside_cli.exists():
            return [str(syside_cli)]
    return [str(python_path), "-m", "syside"]


def find_generated_file(model_dir: Path, model_id: int) -> Optional[Path]:
    candidates = [
        model_dir / f"{model_id}.sysml",
        model_dir / f"{model_id:03d}.sysml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in sorted(model_dir.glob("*.sysml")):
        name = candidate.name.lower()
        if "groundtruth" in name:
            continue
        if name.startswith("iteration_"):
            continue
        return candidate
    return None


def load_entries_from_manifests(output_root: Path, selected_ids: Optional[Set[int]]) -> List[CheckEntry]:
    manifest_paths = sorted(output_root.glob("*/*_refine_manifest.json"))
    entries: List[CheckEntry] = []
    for manifest_path in manifest_paths:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        model_id = payload.get("model_id")
        if model_id is None:
            parent_name = manifest_path.parent.name
            if not parent_name.isdigit():
                continue
            model_id = int(parent_name)
        else:
            model_id = int(model_id)

        if selected_ids is not None and model_id not in selected_ids:
            continue

        generated_path_raw = payload.get("generated_path")
        if not generated_path_raw:
            continue

        generated_path = Path(str(generated_path_raw))
        if not generated_path.is_absolute():
            generated_path = (output_root / generated_path).resolve()

        entries.append(
            CheckEntry(
                model_id=model_id,
                generated_path=generated_path,
                source_path=manifest_path,
                source_kind="manifest",
                manifest_status=str(payload.get("status", "")),
            )
        )

    entries.sort(key=lambda item: item.model_id)
    return entries


def load_entries_direct(output_root: Path, selected_ids: Optional[Set[int]]) -> List[CheckEntry]:
    entries: List[CheckEntry] = []
    for child in sorted(output_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or not child.name.isdigit():
            continue
        model_id = int(child.name)
        if selected_ids is not None and model_id not in selected_ids:
            continue
        generated_path = find_generated_file(child, model_id)
        if generated_path is None:
            continue
        entries.append(
            CheckEntry(
                model_id=model_id,
                generated_path=generated_path,
                source_path=generated_path,
                source_kind="direct",
                manifest_status="",
            )
        )

    entries.sort(key=lambda item: item.model_id)
    return entries


def resolve_entries(
    output_root: Path,
    selected_ids: Optional[Set[int]],
    mode: str,
) -> tuple[list[CheckEntry], str]:
    manifest_entries = load_entries_from_manifests(output_root, selected_ids)

    if mode == "manifest":
        return manifest_entries, "manifest"

    if mode == "direct":
        return load_entries_direct(output_root, selected_ids), "direct"

    if manifest_entries:
        return manifest_entries, "manifest"
    return load_entries_direct(output_root, selected_ids), "direct"


def run_single_check(
    entry: CheckEntry,
    syside_cmd_prefix: Sequence[str],
    validate_with: str,
    timeout_seconds: int,
) -> Dict[str, object]:
    t0 = perf_counter()
    result: Dict[str, object] = {
        "id": entry.model_id,
        "source_kind": entry.source_kind,
        "source_path": str(entry.source_path),
        "manifest_status": entry.manifest_status,
        "generated_path": str(entry.generated_path),
        "exists": entry.generated_path.exists(),
        "validate_with": validate_with,
        "return_code": None,
        "passed": False,
        "duration_seconds": 0.0,
        "stdout": "",
        "stderr": "",
    }
    if not entry.generated_path.exists():
        result["stderr"] = "Generated SysML file does not exist."
        result["duration_seconds"] = perf_counter() - t0
        return result

    target_name = entry.generated_path.name
    subcmd = ["check", target_name] if validate_with == "check" else ["format", target_name]
    cmd = list(syside_cmd_prefix) + subcmd
    try:
        proc = subprocess.run(
            cmd,
            cwd=entry.generated_path.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        result["return_code"] = proc.returncode
        result["stdout"] = proc.stdout.strip()
        result["stderr"] = proc.stderr.strip()
        result["passed"] = proc.returncode == 0
    except subprocess.TimeoutExpired as exc:
        result["return_code"] = 124
        result["stdout"] = (exc.stdout or "").strip()
        result["stderr"] = ((exc.stderr or "").strip() + f"\n[timeout] exceeded {timeout_seconds}s").strip()

    result["duration_seconds"] = perf_counter() - t0
    return result


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"--output-root does not exist: {output_root}")

    selected_ids = parse_id_filter(args.ids)
    python_path = resolve_python_executable(args.venv)
    syside_prefix = resolve_syside_command(python_path, args.venv)
    summary_json_path = (
        args.summary_json.resolve()
        if args.summary_json is not None
        else (output_root / "second_pass_check_summary.json").resolve()
    )

    entries, resolved_mode = resolve_entries(output_root, selected_ids, args.mode)
    if not entries:
        raise RuntimeError(
            f"No entries found under {output_root} for mode={args.mode} and filter={args.ids!r}."
        )

    print(
        f"[verify] loaded {len(entries)} entries from {output_root} | "
        f"workers={args.parallelism} | mode={resolved_mode}"
    )

    checks: List[Dict[str, object]] = []
    by_id: Dict[int, Dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.parallelism)) as executor:
        future_to_id = {
            executor.submit(
                run_single_check,
                entry,
                syside_prefix,
                args.validate_with,
                args.timeout_seconds,
            ): entry.model_id
            for entry in entries
        }
        completed = 0
        total = len(entries)
        for fut in as_completed(future_to_id):
            completed += 1
            result = fut.result()
            by_id[int(result["id"])] = result
            status = "PASS" if result.get("passed") else "FAIL"
            print(
                f"[{completed}/{total}] id={result['id']} {status} "
                f"rc={result['return_code']} dt={result['duration_seconds']:.2f}s"
            )

    for entry in entries:
        checks.append(by_id[entry.model_id])

    pass_count = sum(1 for row in checks if row.get("passed"))
    fail_count = len(checks) - pass_count
    missing_count = sum(1 for row in checks if not row.get("exists"))
    timeout_count = sum(1 for row in checks if row.get("return_code") == 124)
    status_histogram: Dict[str, int] = {}
    for row in checks:
        key = str(row.get("manifest_status", "")).lower()
        status_histogram[key] = status_histogram.get(key, 0) + 1

    summary = {
        "generated_at_utc": utc_now_iso(),
        "output_root": str(output_root),
        "resolved_mode": resolved_mode,
        "venv": str(args.venv.resolve()) if args.venv else None,
        "python_executable": str(python_path),
        "syside_command_prefix": syside_prefix,
        "validate_with": args.validate_with,
        "timeout_seconds": args.timeout_seconds,
        "parallelism": args.parallelism,
        "id_filter": sorted(selected_ids) if selected_ids is not None else None,
        "total_checked": len(checks),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "missing_file_count": missing_count,
        "timeout_count": timeout_count,
        "manifest_status_histogram": status_histogram,
        "failed_ids": [row["id"] for row in checks if not row.get("passed")],
        "checks": checks,
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[verify] summary written to {summary_json_path}")
    print(
        f"[verify] total={len(checks)} pass={pass_count} fail={fail_count} "
        f"missing={missing_count} timeout={timeout_count}"
    )

    if fail_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
