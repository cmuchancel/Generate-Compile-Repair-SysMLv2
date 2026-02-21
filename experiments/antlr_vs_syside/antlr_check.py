#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HAMR_CLASSES_DIR = SCRIPT_DIR / "generated" / "hamr_java_classes"
ANTLR_JAR = SCRIPT_DIR / "tools" / "antlr-4.13.2-complete.jar"


def _check_with_hamr_java(path: Path) -> tuple[bool, list[str]]:
    if not HAMR_CLASSES_DIR.exists() or not ANTLR_JAR.exists():
        raise RuntimeError(
            "HAMR Java parser artifacts missing. Run: bash experiments/antlr_vs_syside/setup.sh"
        )

    java_bin = os.environ.get("JAVA_BIN", "java")
    cp = f"{HAMR_CLASSES_DIR}:{ANTLR_JAR}"
    proc = subprocess.run(
        [java_bin, "-cp", cp, "ParseSysML", str(path)],
        capture_output=True,
        text=True,
    )

    out_lines = [ln.rstrip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    err_lines = [ln.rstrip() for ln in (proc.stderr or "").splitlines() if ln.strip()]
    lines = out_lines + err_lines

    ok = proc.returncode == 0
    # Drop marker line; keep concrete diagnostics only
    lines = [ln for ln in lines if not ln.startswith("ANTLR_PARSE_PASS") and not ln.startswith("ANTLR_PARSE_FAIL")]
    return ok, lines


def check_parse(path: Path) -> tuple[str, bool, list[str]]:
    ok, errors = _check_with_hamr_java(path)
    return "hamr_full", ok, errors


def main() -> int:
    ap = argparse.ArgumentParser(description="ANTLR grammar-level parse check")
    ap.add_argument("sysml_file", type=Path, help="Path to .sysml file")
    args = ap.parse_args()

    if not args.sysml_file.exists():
        print(f"ERROR: file not found: {args.sysml_file}", file=sys.stderr)
        return 2

    try:
        label, ok, errors = check_parse(args.sysml_file)
    except Exception as exc:
        print("ERROR: HAMR ANTLR parser backend unavailable.", file=sys.stderr)
        print(f"Backend load failure: {exc}", file=sys.stderr)
        return 2

    if ok:
        print(f"ANTLR_PARSE_PASS[{label}] {args.sysml_file}")
        return 0

    print(f"ANTLR_PARSE_FAIL[{label}] {args.sysml_file}")
    for err in errors:
        print(err)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
