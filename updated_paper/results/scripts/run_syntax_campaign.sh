#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
DATA_DIR="$REPO_ROOT/paper/results/data"
FIG_DIR="$REPO_ROOT/paper/results/figures"
TABLE_DIR="$REPO_ROOT/paper/results/tables"

mkdir -p "$DATA_DIR" "$FIG_DIR" "$TABLE_DIR"

echo "[campaign] extracting syntax metrics"
"$PYTHON_BIN" "$SCRIPT_DIR/extract_syntax_metrics.py" --repo-root "$REPO_ROOT"

echo "[campaign] computing summary stats"
"$PYTHON_BIN" "$SCRIPT_DIR/compute_syntax_stats.py" --output-data-dir "$DATA_DIR"

echo "[campaign] generating tables"
"$PYTHON_BIN" "$SCRIPT_DIR/make_syntax_tables.py" --data-dir "$DATA_DIR" --tables-dir "$TABLE_DIR"

echo "[campaign] generating figures"
"$PYTHON_BIN" "$SCRIPT_DIR/make_syntax_figures.py" --data-dir "$DATA_DIR" --figures-dir "$FIG_DIR"

# Determinism check on stats outputs (excluding campaign_manifest.json timestamp metadata)
KEY_FILES=(
  "$DATA_DIR/prompt_level_syntax_metrics.csv"
  "$DATA_DIR/iteration_level_syntax_metrics.csv"
  "$DATA_DIR/model_level_syntax_summary.csv"
  "$DATA_DIR/error_taxonomy_summary.csv"
  "$DATA_DIR/stat_tests.json"
)

hash_bundle() {
  "$PYTHON_BIN" - "${KEY_FILES[@]}" <<'PY'
import hashlib
import pathlib
import sys

files = sys.argv[1:]
bundle = hashlib.sha256()
for file_path in files:
    data = pathlib.Path(file_path).read_bytes()
    bundle.update(hashlib.sha256(data).hexdigest().encode("ascii"))
    bundle.update(b"\n")

print(bundle.hexdigest())
PY
}

HASH1="$(hash_bundle)"

# Re-run extract+compute only, then compare hashes
"$PYTHON_BIN" "$SCRIPT_DIR/extract_syntax_metrics.py" --repo-root "$REPO_ROOT" >/dev/null
"$PYTHON_BIN" "$SCRIPT_DIR/compute_syntax_stats.py" --output-data-dir "$DATA_DIR" >/dev/null

HASH2="$(hash_bundle)"
if [[ "$HASH1" == "$HASH2" ]]; then
  echo "[determinism] PASS: key stats outputs are hash-stable"
else
  echo "[determinism] FAIL: key stats outputs changed between runs"
  echo "hash1=$HASH1"
  echo "hash2=$HASH2"
  exit 1
fi

echo "[campaign] final summary"
"$PYTHON_BIN" - "$REPO_ROOT" <<'PY'
import json
from pathlib import Path
import sys
import pandas as pd

repo = Path(sys.argv[1]).resolve()
data = repo / "paper" / "results" / "data"
stat = json.loads((data / "stat_tests.json").read_text(encoding="utf-8"))
overall = stat["overall"]

m = pd.read_csv(data / "model_level_syntax_summary.csv")
pm = m[~((m["provider"] == "ALL") & (m["model"] == "ALL"))].copy()

print(f"Total prompts analyzed per model:")
for _, r in pm.iterrows():
    print(f"  - {r['provider']}/{r['model']}: {int(r['total_prompts'])}")

print(f"First-shot pass %: {overall['first_shot_pass_rate_pct']:.2f}")
print(f"Eventual pass %: {overall['eventual_pass_rate_pct']:.2f}")
print(f"Improvement % (absolute gain pp): {overall['absolute_gain_pp']:.2f}")
print(f"Count unresolved prompts: {int(overall['unresolved_count'])}")
print("Paths:")
print("  methods: paper/methods/methods_syntactic_only.tex")
print("  results: paper/results/results_syntactic_only.tex")
print("  figures: paper/results/figures")
print("  tables:  paper/results/tables")
print("  scripts: paper/results/scripts")
PY
