#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def _is_working(cmd: list[str]) -> bool:
    try:
        cp = subprocess.run(cmd + ["--help"], capture_output=True, text=True, timeout=15)
        return cp.returncode == 0
    except Exception:
        return False


def detect_syside_command() -> list[str]:
    if shutil.which("syside") and _is_working(["syside", "check"]):
        return ["syside", "check"]

    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists() and _is_working([str(venv_py), "-m", "syside", "check"]):
        return [str(venv_py), "-m", "syside", "check"]

    if _is_working([sys.executable, "-m", "syside", "check"]):
        return [sys.executable, "-m", "syside", "check"]

    py3 = shutil.which("python3")
    if py3 and _is_working([py3, "-m", "syside", "check"]):
        return [py3, "-m", "syside", "check"]

    raise RuntimeError(
        "Could not locate a working SysIDE checker. Tried: syside check, "
        "<repo>/.venv/bin/python -m syside check, sys.executable -m syside check."
    )


def run_syside_check(path: Path) -> subprocess.CompletedProcess[str]:
    cmd = detect_syside_command() + [str(path)]
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run SysIDE compile check for one SysML file")
    ap.add_argument("sysml_file", type=Path, help="Path to .sysml file")
    args = ap.parse_args()

    if not args.sysml_file.exists():
        print(f"ERROR: file not found: {args.sysml_file}", file=sys.stderr)
        return 2

    try:
        cp = run_syside_check(args.sysml_file)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if cp.stdout:
        print(cp.stdout.rstrip())
    if cp.stderr:
        print(cp.stderr.rstrip(), file=sys.stderr)

    if cp.returncode == 0:
        print(f"SYSIDE_COMPILE_PASS {args.sysml_file}")
    else:
        print(f"SYSIDE_COMPILE_FAIL {args.sysml_file}")

    return cp.returncode


if __name__ == "__main__":
    raise SystemExit(main())
