#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from antlr_check import check_parse

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET_DIR = SCRIPT_DIR / "examples" / "mismatch_10_distinct"


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify all generated SysML files pass ANTLR parsing.")
    ap.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    ap.add_argument("--glob", default="*.sysml")
    args = ap.parse_args()

    if not args.target_dir.exists():
        print(f"ERROR: target directory not found: {args.target_dir}")
        return 2

    files = sorted(args.target_dir.glob(args.glob))
    if not files:
        print(f"ERROR: no files matching {args.glob} under {args.target_dir}")
        return 2

    failures: list[tuple[Path, str, list[str]]] = []
    for path in files:
        backend, ok, errors = check_parse(path)
        if ok:
            print(f"PASS[{backend}] {path}")
        else:
            print(f"FAIL[{backend}] {path}")
            for err in errors:
                print(f"  {err}")
            failures.append((path, backend, errors))

    print(f"Checked {len(files)} files")
    print(f"ANTLR pass count: {len(files) - len(failures)}")
    print(f"ANTLR fail count: {len(failures)}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
