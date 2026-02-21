#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

DEFAULT_PROVIDERS = [
    "Generated_from_Prompts_API_LOOP_OPENAI",
    "Generated_from_Prompts_API_LOOP_ANTHROPIC",
    "Generated_from_Prompts_API_LOOP_DEEPSEEK_REASONER",
    "Generated_from_Prompts_API_LOOP_MISTRAL_LARGE",
]


@dataclass
class Row:
    provider: str
    prompt_id: int
    file: str
    antlr_ok: bool
    syside_ok: bool
    both_ok: bool
    antlr_output: str
    syside_output: str


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def _compact(text: str, max_chars: int = 1200) -> str:
    t = (text or "").strip()
    return t[:max_chars]


def discover_files(api_loop_dir: Path, providers: list[str]) -> list[tuple[str, int, Path]]:
    items: list[tuple[str, int, Path]] = []
    for prov in providers:
        root = api_loop_dir / prov
        if not root.exists():
            continue
        ids = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
        for d in sorted(ids, key=lambda p: int(p.name)):
            f = d / f"{d.name}.sysml"
            if f.exists():
                items.append((prov, int(d.name), f))
    return items


def audit_one(item: tuple[str, int, Path], antlr_cmd: list[str], syside_cmd: list[str]) -> Row:
    provider, prompt_id, file_path = item
    antlr = _run(antlr_cmd + [str(file_path)])
    syside = _run(syside_cmd + [str(file_path)])

    antlr_out = _compact((antlr.stdout or "") + ("\n" + antlr.stderr if antlr.stderr else ""))
    syside_out = _compact((syside.stdout or "") + ("\n" + syside.stderr if syside.stderr else ""))

    antlr_ok = antlr.returncode == 0
    syside_ok = syside.returncode == 0

    return Row(
        provider=provider,
        prompt_id=prompt_id,
        file=str(file_path),
        antlr_ok=antlr_ok,
        syside_ok=syside_ok,
        both_ok=antlr_ok and syside_ok,
        antlr_output=antlr_out,
        syside_output=syside_out,
    )


def summarize(rows: list[Row], providers: list[str]) -> dict:
    summary: dict[str, dict[str, int]] = {}
    for p in providers:
        p_rows = [r for r in rows if r.provider == p]
        summary[p] = {
            "total": len(p_rows),
            "antlr_pass": sum(1 for r in p_rows if r.antlr_ok),
            "syside_pass": sum(1 for r in p_rows if r.syside_ok),
            "both_pass": sum(1 for r in p_rows if r.both_ok),
            "any_fail": sum(1 for r in p_rows if not r.both_ok),
        }

    summary["ALL"] = {
        "total": len(rows),
        "antlr_pass": sum(1 for r in rows if r.antlr_ok),
        "syside_pass": sum(1 for r in rows if r.syside_ok),
        "both_pass": sum(1 for r in rows if r.both_ok),
        "any_fail": sum(1 for r in rows if not r.both_ok),
    }
    return summary


def write_outputs(rows: list[Row], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "generated_sysml_audit.json"
    csv_path = out_dir / "generated_sysml_audit.csv"
    fail_csv_path = out_dir / "generated_sysml_audit_failures.csv"

    json_path.write_text(
        json.dumps({"summary": summary, "rows": [asdict(r) for r in rows]}, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "provider",
                "prompt_id",
                "file",
                "antlr_ok",
                "syside_ok",
                "both_ok",
                "antlr_output",
                "syside_output",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    fail_rows = [r for r in rows if not r.both_ok]
    with fail_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "provider",
                "prompt_id",
                "file",
                "antlr_ok",
                "syside_ok",
                "both_ok",
                "antlr_output",
                "syside_output",
            ],
        )
        w.writeheader()
        for r in fail_rows:
            w.writerow(asdict(r))


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit generated SysML files with ANTLR + SysIDE checks.")
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
    args = ap.parse_args()

    antlr_cmd = [str(args.python_bin), "experiments/antlr_vs_syside/antlr_check.py"]
    syside_cmd = [str(args.python_bin), "experiments/antlr_vs_syside/syside_check.py"]

    items = discover_files(args.api_loop_dir, args.providers)
    if not items:
        print("No generated .sysml files found for requested providers")
        return 2

    rows: list[Row] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(audit_one, item, antlr_cmd, syside_cmd) for item in items]
        for idx, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if idx % 100 == 0:
                print(f"progress {idx}/{len(futures)}")

    rows.sort(key=lambda r: (r.provider, r.prompt_id))
    summary = summarize(rows, args.providers)
    write_outputs(rows, summary, args.out_dir)

    print("=== Summary ===")
    for p in args.providers + ["ALL"]:
        if p not in summary:
            continue
        s = summary[p]
        print(
            f"{p}: total={s['total']} antlr_pass={s['antlr_pass']} "
            f"syside_pass={s['syside_pass']} both_pass={s['both_pass']} any_fail={s['any_fail']}"
        )

    print(f"Wrote {args.out_dir / 'generated_sysml_audit.json'}")
    print(f"Wrote {args.out_dir / 'generated_sysml_audit.csv'}")
    print(f"Wrote {args.out_dir / 'generated_sysml_audit_failures.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
