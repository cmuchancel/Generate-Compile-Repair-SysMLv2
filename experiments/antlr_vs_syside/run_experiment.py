#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
EXAMPLES_DIR = SCRIPT_DIR / "examples" / "mismatch_10_distinct"
ANTLR_CHECK = SCRIPT_DIR / "antlr_check.py"
SYSIDE_CHECK = SCRIPT_DIR / "syside_check.py"


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def pick_python_bin() -> str:
    # Prefer explicit override, then repo .venv, then current interpreter.
    env_bin = os.environ.get("PYTHON_BIN")
    if env_bin:
        return env_bin
    venv_bin = SCRIPT_DIR.parent.parent / ".venv" / "bin" / "python"
    if venv_bin.exists():
        return str(venv_bin)
    return sys.executable


def compact(text: str, max_lines: int = 40) -> str:
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    cleaned = ansi_escape.sub("", text)
    lines = [ln.rstrip() for ln in cleaned.splitlines() if ln.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + ["... [truncated]"])


def run_one(example: Path) -> dict[str, str | bool]:
    python_bin = pick_python_bin()
    antlr_cp = run_cmd([python_bin, str(ANTLR_CHECK), str(example)])
    syside_cp = run_cmd([python_bin, str(SYSIDE_CHECK), str(example)])

    antlr_text = (antlr_cp.stdout or "") + (("\n" + antlr_cp.stderr) if antlr_cp.stderr else "")
    syside_text = (syside_cp.stdout or "") + (("\n" + syside_cp.stderr) if syside_cp.stderr else "")

    parse_ok = antlr_cp.returncode == 0
    compile_ok = syside_cp.returncode == 0

    return {
        "example": example.name,
        "parse_ok": parse_ok,
        "compile_ok": compile_ok,
        "antlr_returncode": str(antlr_cp.returncode),
        "syside_returncode": str(syside_cp.returncode),
        "antlr_errors": "" if parse_ok else compact(antlr_text),
        "syside_errors": "" if compile_ok else compact(syside_text),
    }


def write_csv(rows: list[dict[str, str | bool]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "example",
        "parse_ok",
        "compile_ok",
        "antlr_returncode",
        "syside_returncode",
        "antlr_errors",
        "syside_errors",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_summary(rows: list[dict[str, str | bool]], out_md: Path) -> None:
    total = len(rows)
    parse_pass = sum(1 for r in rows if r["parse_ok"])
    compile_pass = sum(1 for r in rows if r["compile_ok"])
    mismatches = [r for r in rows if r["parse_ok"] and not r["compile_ok"]]

    lines: list[str] = []
    lines.append("# ANTLR vs SysIDE Summary")
    lines.append("")
    lines.append(f"- Total examples: {total}")
    lines.append(f"- ANTLR parse pass: {parse_pass}")
    lines.append(f"- SysIDE compile pass: {compile_pass}")
    lines.append(f"- Mismatch count (parse PASS, compile FAIL): {len(mismatches)}")
    lines.append("")

    if mismatches:
        lines.append("## Mismatch Examples")
        for m in mismatches:
            lines.append(f"- `{m['example']}`")
        lines.append("")

        lines.append("## Highlighted Diagnostics")
        for m in mismatches[:2]:
            lines.append(f"### {m['example']}")
            lines.append("```text")
            lines.append(str(m["syside_errors"]).strip() or "(no diagnostics captured)")
            lines.append("```")
            lines.append("")
    else:
        lines.append("## No mismatches found")
        lines.append("The current set did not produce parse-pass/compile-fail cases.")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ANTLR-vs-SysIDE mismatch experiment")
    ap.add_argument("--examples-dir", type=Path, default=EXAMPLES_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    examples = sorted(args.examples_dir.glob("*.sysml"))
    if not examples:
        print(
            f"No examples found under {args.examples_dir}. "
            "Populate this folder with curated test files first.",
            file=sys.stderr,
        )
        return 2

    rows = [run_one(ex) for ex in examples]

    out_csv = args.results_dir / "results.csv"
    out_md = args.results_dir / "summary.md"
    write_csv(rows, out_csv)
    write_summary(rows, out_md)

    mismatches = [r for r in rows if r["parse_ok"] and not r["compile_ok"]]
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    print(f"Mismatch count (parse PASS, compile FAIL): {len(mismatches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
