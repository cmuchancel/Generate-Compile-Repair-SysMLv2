# Documentation Index

This folder contains deep documentation for the ANTLR-vs-SysIDE experiment.

Parser provenance:
- ANTLR parse outcomes in this experiment are produced by the third-party HAMR parser pinned under `experiments/antlr_vs_syside/third_party/hamr-sysml-parser`.
- No locally-authored parser grammar is used in the runtime parse check path.

## Core docs
- `ANTLR_VS_SYSIDE_THEORY.md`
  - Formal boundary between grammar parsing and compilability checks.
  - Explains why ANTLR pass does not imply SysIDE pass.
- `WHY_EACH_EXAMPLE_FAILS.md`
  - Human-readable explanation for all 10 distinct mismatch files.
- `CONTEXT_BOUNDARY_CHECKLIST.md`
  - Checklist-style map of what syntax can prove vs what semantic checks require.
- `KNOWN_DISCREPANCIES.md`
  - Documents known parser/tooling mismatch cases (ANTLR fail while SysIDE passes).

## Generated from results
- `FAILURE_CATALOG.generated.md`
  - Per-file machine-backed failure catalog from `results/results.csv`.
  - Includes source context and captured SysIDE diagnostic snippet.
- `ERROR_FAMILY_BREAKDOWN.generated.md`
  - Aggregate family counts and interpretation.
- `FAILURE_MATRIX.generated.md`
  - Compact matrix of all 10 counterexamples with family, location, and semantic category.

## Refresh workflow
These generated docs are checked-in snapshots based on `results/results.csv`.
If results change, refresh this folder manually alongside the updated results.
