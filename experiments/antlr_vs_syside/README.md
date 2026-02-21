# ANTLR vs SysIDE (SysMLv2): Grammar Parsability vs Toolchain Compilability

This experiment demonstrates that **grammar-level parsability** does not guarantee **SysIDE compilability**.

- **Parse check**: ANTLR parser (grammar-level only)
- **Compile check**: SysIDE checker (`syside check` equivalent)

## What this proves
A file can:
1. Parse successfully under an ANTLR grammar, yet
2. Fail SysIDE compiler checks due to unresolved references or other static/semantic constraints.

## Directory Layout
- `setup.sh`: setup and parser generation
- `antlr_check.py`: runs grammar-level parse check (exit 0 on parse success)
- `syside_check.py`: runs SysIDE compile check (exit 0 on compile success)
- `verify_generated_antlr_pass.py`: verifies generated files all pass ANTLR parsing
- `run_experiment.py`: executes all checks and writes report artifacts
- `examples/mismatch_10_distinct/`: canonical example set used in the experiment
- `results/results.csv`: per-file parse/compile outcomes + diagnostics
- `results/summary.md`: concise summary and highlighted mismatch diagnostics
- `docs/`: extensive theory + per-example failure documentation

## Grammar Sources + Pins
This experiment now uses the HAMR ANTLR SysML parser as the primary grammar source:

- HAMR parser repo: `https://github.com/sireum/hamr-sysml-parser`
- Pinned commit: `d7c87942ca9f84de611415c8cca0c5916cc9ccae`
- Primary grammar used by `antlr_check.py`:
  - `src/org/sireum/hamr/sysml/parser/SysMLv2.g4`
  - Entry rule: `entryRuleRootNamespace`
  - Execution backend: compiled Java parser (`ParseSysML`) for strict grammar parsing
  - No locally-authored parser grammar is used in the runtime check path.

For traceability to OMG release content, `setup.sh` also pins:
- `https://github.com/Systems-Modeling/SysML-v2-Release`
- Commit: `b48c37f3bc5702bc4dfce9ce2b7e454720c7c2fb`

## Prerequisites
- Python 3.10+
- Java 17+ (required to generate ANTLR parser artifacts)
- Internet access (for cloning grammar repo and downloading ANTLR jar)
- SysIDE available via one of:
  - `syside check`
  - `<repo>/.venv/bin/python -m syside check`
  - `python -m syside check`

## Setup
From repo root:

```bash
bash experiments/antlr_vs_syside/setup.sh
```

This will:
1. Clone/pin `experiments/antlr_vs_syside/third_party/SysML-v2-Release` and `experiments/antlr_vs_syside/third_party/hamr-sysml-parser`.
2. Download ANTLR jar into `experiments/antlr_vs_syside/tools/`.
3. Compile full HAMR Java parser artifacts into:
   - `experiments/antlr_vs_syside/generated/hamr_java_classes/`

## Run
From repo root:

```bash
python experiments/antlr_vs_syside/run_experiment.py

# Verify all curated examples pass ANTLR
python experiments/antlr_vs_syside/verify_generated_antlr_pass.py \
  --target-dir experiments/antlr_vs_syside/examples/mismatch_10_distinct
```

Outputs:
- `experiments/antlr_vs_syside/results/results.csv`
- `experiments/antlr_vs_syside/results/summary.md`
- `experiments/antlr_vs_syside/docs/FAILURE_CATALOG.generated.md`
- `experiments/antlr_vs_syside/docs/ERROR_FAMILY_BREAKDOWN.generated.md`
- `experiments/antlr_vs_syside/docs/FAILURE_MATRIX.generated.md`

## Documentation (Deep)
Start with:
- `experiments/antlr_vs_syside/docs/README.md`

Key docs:
- `experiments/antlr_vs_syside/docs/ANTLR_VS_SYSIDE_THEORY.md`
- `experiments/antlr_vs_syside/docs/WHY_EACH_EXAMPLE_FAILS.md`
- `experiments/antlr_vs_syside/docs/CONTEXT_BOUNDARY_CHECKLIST.md`
- `experiments/antlr_vs_syside/docs/KNOWN_DISCREPANCIES.md`
- `experiments/antlr_vs_syside/docs/FAILURE_CATALOG.generated.md`
- `experiments/antlr_vs_syside/docs/ERROR_FAMILY_BREAKDOWN.generated.md`
- `experiments/antlr_vs_syside/docs/FAILURE_MATRIX.generated.md`

## Parse vs Compile Definitions
- **Parse PASS** (`parse_ok=True`): ANTLR grammar accepted file syntax.
- **Compile PASS** (`compile_ok=True`): SysIDE accepted file under `check` (no compile errors).

Mismatch of interest:
- `parse_ok=True` and `compile_ok=False`.

## Why this demonstrates the boundary
ANTLR answers: "Does this text match the grammar language?"

SysIDE answers: "Does this parsed model satisfy semantic/static constraints needed for compilability?"

In this experiment, all 10 files satisfy the first but fail the second, which establishes that grammar parsability is necessary but not sufficient for SysIDE compilability.

## Parser provenance statement
This experiment's ANTLR pass/fail results come from the third-party HAMR parser implementation pinned in:
- `experiments/antlr_vs_syside/third_party/hamr-sysml-parser`

No local custom parser grammar is used to produce reported parse outcomes.
