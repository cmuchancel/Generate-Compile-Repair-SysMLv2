#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_PROVIDERS = [
    "Generated_from_Prompts_API_LOOP_OPENAI",
    "Generated_from_Prompts_API_LOOP_ANTHROPIC",
    "Generated_from_Prompts_API_LOOP_DEEPSEEK_REASONER",
    "Generated_from_Prompts_API_LOOP_MISTRAL_LARGE",
]


@dataclass
class WorkItem:
    provider: str
    prompt_id: int
    manifest_path: str
    run_dir: str
    first_iteration_path: str
    resolve_error: str = ""


@dataclass
class Row:
    provider: str
    prompt_id: int
    manifest_path: str
    run_dir: str
    first_iteration_path: str
    antlr_ok: bool | None
    syside_ok: bool | None
    paired_label: str
    antlr_exit_code: int | None
    syside_exit_code: int | None
    antlr_output: str
    syside_output: str
    resolve_error: str


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def _compact(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    return t[:max_chars]


def _to_path(raw: str | None, repo_root: Path) -> Path | None:
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_log_record(log_json: Any) -> dict[str, Any] | None:
    if isinstance(log_json, list):
        for rec in log_json:
            try:
                if int(rec.get("iteration", -1)) == 1:
                    return rec
            except Exception:
                continue
        if log_json:
            first = log_json[0]
            if isinstance(first, dict):
                return first
    elif isinstance(log_json, dict):
        entries = log_json.get("iterations")
        if isinstance(entries, list):
            for rec in entries:
                try:
                    if int(rec.get("iteration", -1)) == 1:
                        return rec
                except Exception:
                    continue
            if entries and isinstance(entries[0], dict):
                return entries[0]
    return None


def _candidate_first_iter_paths(manifest: dict[str, Any], repo_root: Path) -> tuple[str, list[Path]]:
    run_dir = _to_path(manifest.get("run_dir"), repo_root)
    archived = _to_path(manifest.get("archived_run_dir"), repo_root)
    run_log = _to_path(manifest.get("run_log_path"), repo_root)

    candidates: list[Path] = []
    if run_dir:
        candidates.append(run_dir / "iteration_01.sysml")
    if archived:
        candidates.append(archived / "iteration_01.sysml")

    if run_log and run_log.exists():
        try:
            log_json = _load_json(run_log)
            rec = _first_log_record(log_json)
            if rec and rec.get("sysml_path"):
                p = _to_path(rec["sysml_path"], repo_root)
                if p:
                    candidates.append(p)
                    # If old path was preserved in run_log, map filename onto run_dir as fallback.
                    if run_dir:
                        candidates.append(run_dir / p.name)
        except Exception:
            pass

    # De-duplicate while preserving order.
    unique: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return str(run_dir) if run_dir else "", unique


def resolve_first_iteration_files(api_loop_dir: Path, providers: list[str], repo_root: Path) -> list[WorkItem]:
    items: list[WorkItem] = []

    for provider in providers:
        prov_root = api_loop_dir / provider
        if not prov_root.exists():
            continue

        prompt_dirs = [p for p in prov_root.iterdir() if p.is_dir() and p.name.isdigit()]
        for prompt_dir in sorted(prompt_dirs, key=lambda p: int(p.name)):
            prompt_id = int(prompt_dir.name)
            manifest_path = prompt_dir / f"{prompt_id}_refine_manifest.json"
            if not manifest_path.exists():
                items.append(
                    WorkItem(
                        provider=provider,
                        prompt_id=prompt_id,
                        manifest_path=str(manifest_path),
                        run_dir="",
                        first_iteration_path="",
                        resolve_error="missing_manifest",
                    )
                )
                continue

            try:
                manifest = _load_json(manifest_path)
            except Exception as exc:
                items.append(
                    WorkItem(
                        provider=provider,
                        prompt_id=prompt_id,
                        manifest_path=str(manifest_path),
                        run_dir="",
                        first_iteration_path="",
                        resolve_error=f"manifest_parse_error: {exc}",
                    )
                )
                continue

            run_dir_str, candidates = _candidate_first_iter_paths(manifest, repo_root)
            existing = next((p for p in candidates if p.exists()), None)
            if existing is None:
                candidate_preview = "; ".join(str(p) for p in candidates[:4]) if candidates else "none"
                items.append(
                    WorkItem(
                        provider=provider,
                        prompt_id=prompt_id,
                        manifest_path=str(manifest_path),
                        run_dir=run_dir_str,
                        first_iteration_path="",
                        resolve_error=f"missing_iteration_01_sysml (candidates: {candidate_preview})",
                    )
                )
                continue

            items.append(
                WorkItem(
                    provider=provider,
                    prompt_id=prompt_id,
                    manifest_path=str(manifest_path),
                    run_dir=run_dir_str,
                    first_iteration_path=str(existing),
                    resolve_error="",
                )
            )

    return items


def audit_one(item: WorkItem, antlr_cmd: list[str], syside_cmd: list[str], max_output_chars: int) -> Row:
    if item.resolve_error:
        return Row(
            provider=item.provider,
            prompt_id=item.prompt_id,
            manifest_path=item.manifest_path,
            run_dir=item.run_dir,
            first_iteration_path=item.first_iteration_path,
            antlr_ok=None,
            syside_ok=None,
            paired_label="unresolved_input",
            antlr_exit_code=None,
            syside_exit_code=None,
            antlr_output="",
            syside_output="",
            resolve_error=item.resolve_error,
        )

    first_iter_path = Path(item.first_iteration_path)
    antlr = _run(antlr_cmd + [str(first_iter_path)])
    syside = _run(syside_cmd + [str(first_iter_path)])

    antlr_ok = antlr.returncode == 0
    syside_ok = syside.returncode == 0

    if antlr_ok and syside_ok:
        paired = "both_pass"
    elif antlr_ok and not syside_ok:
        paired = "antlr_only_pass"
    elif not antlr_ok and syside_ok:
        paired = "syside_only_pass"
    else:
        paired = "both_fail"

    antlr_output = _compact((antlr.stdout or "") + ("\n" + antlr.stderr if antlr.stderr else ""), max_output_chars)
    syside_output = _compact((syside.stdout or "") + ("\n" + syside.stderr if syside.stderr else ""), max_output_chars)

    return Row(
        provider=item.provider,
        prompt_id=item.prompt_id,
        manifest_path=item.manifest_path,
        run_dir=item.run_dir,
        first_iteration_path=item.first_iteration_path,
        antlr_ok=antlr_ok,
        syside_ok=syside_ok,
        paired_label=paired,
        antlr_exit_code=antlr.returncode,
        syside_exit_code=syside.returncode,
        antlr_output=antlr_output,
        syside_output=syside_output,
        resolve_error="",
    )


def summarize(rows: list[Row], providers: list[str]) -> dict[str, dict[str, int]]:
    def _summarize_subset(subset: list[Row]) -> dict[str, int]:
        evaluated = [r for r in subset if not r.resolve_error]
        return {
            "total_rows": len(subset),
            "evaluated_rows": len(evaluated),
            "skipped_rows": len(subset) - len(evaluated),
            "antlr_pass": sum(1 for r in evaluated if r.antlr_ok is True),
            "syside_pass": sum(1 for r in evaluated if r.syside_ok is True),
            "both_pass": sum(1 for r in evaluated if r.paired_label == "both_pass"),
            "antlr_only_pass": sum(1 for r in evaluated if r.paired_label == "antlr_only_pass"),
            "syside_only_pass": sum(1 for r in evaluated if r.paired_label == "syside_only_pass"),
            "both_fail": sum(1 for r in evaluated if r.paired_label == "both_fail"),
        }

    summary: dict[str, dict[str, int]] = {}
    for provider in providers:
        summary[provider] = _summarize_subset([r for r in rows if r.provider == provider])

    summary["ALL"] = _summarize_subset(rows)
    return summary


def write_outputs(rows: list[Row], summary: dict[str, dict[str, int]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "first_iteration_paired_audit.json"
    csv_path = out_dir / "first_iteration_paired_audit.csv"
    mismatch_csv_path = out_dir / "first_iteration_paired_mismatches.csv"
    summary_csv_path = out_dir / "first_iteration_paired_summary.csv"

    json_path.write_text(
        json.dumps({"summary": summary, "rows": [asdict(r) for r in rows]}, indent=2),
        encoding="utf-8",
    )

    fieldnames = [
        "provider",
        "prompt_id",
        "manifest_path",
        "run_dir",
        "first_iteration_path",
        "antlr_ok",
        "syside_ok",
        "paired_label",
        "antlr_exit_code",
        "syside_exit_code",
        "resolve_error",
        "antlr_output",
        "syside_output",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    mismatch_rows = [r for r in rows if r.paired_label in {"antlr_only_pass", "syside_only_pass"}]
    with mismatch_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in mismatch_rows:
            writer.writerow(asdict(row))

    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "provider",
                "total_rows",
                "evaluated_rows",
                "skipped_rows",
                "antlr_pass",
                "syside_pass",
                "both_pass",
                "antlr_only_pass",
                "syside_only_pass",
                "both_fail",
            ],
        )
        writer.writeheader()
        for provider, stats in summary.items():
            writer.writerow({"provider": provider, **stats})


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Audit first-iteration SysML outputs (iteration_01.sysml) against ANTLR and SysIDE, "
            "with paired pass/fail labels."
        )
    )
    ap.add_argument("--api-loop-dir", type=Path, default=Path("api_loop"))
    ap.add_argument("--providers", nargs="*", default=DEFAULT_PROVIDERS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/antlr_vs_syside/results"))
    ap.add_argument(
        "--python-bin",
        type=Path,
        default=Path("./.venv/bin/python"),
        help="Python executable used to invoke antlr_check.py and syside_check.py",
    )
    ap.add_argument("--max-output-chars", type=int, default=1200)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    antlr_cmd = [str(args.python_bin), "experiments/antlr_vs_syside/antlr_check.py"]
    syside_cmd = [str(args.python_bin), "experiments/antlr_vs_syside/syside_check.py"]

    items = resolve_first_iteration_files(args.api_loop_dir, args.providers, repo_root)
    if not items:
        print("No prompt folders discovered for the requested providers.")
        return 2

    rows: list[Row] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(audit_one, item, antlr_cmd, syside_cmd, args.max_output_chars)
            for item in items
        ]
        for idx, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if idx % 100 == 0 or idx == len(futures):
                print(f"progress {idx}/{len(futures)}")

    rows.sort(key=lambda r: (r.provider, r.prompt_id))
    summary = summarize(rows, args.providers)
    write_outputs(rows, summary, args.out_dir)

    print("=== First-Iteration Paired Summary ===")
    for provider in args.providers + ["ALL"]:
        if provider not in summary:
            continue
        s = summary[provider]
        print(
            f"{provider}: total={s['total_rows']} evaluated={s['evaluated_rows']} skipped={s['skipped_rows']} "
            f"antlr_pass={s['antlr_pass']} syside_pass={s['syside_pass']} "
            f"both_pass={s['both_pass']} antlr_only={s['antlr_only_pass']} "
            f"syside_only={s['syside_only_pass']} both_fail={s['both_fail']}"
        )

    print(f"Wrote {args.out_dir / 'first_iteration_paired_audit.json'}")
    print(f"Wrote {args.out_dir / 'first_iteration_paired_audit.csv'}")
    print(f"Wrote {args.out_dir / 'first_iteration_paired_mismatches.csv'}")
    print(f"Wrote {args.out_dir / 'first_iteration_paired_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
