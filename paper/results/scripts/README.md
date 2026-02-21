# Syntax Campaign Scripts

This folder reproduces the syntax-only campaign outputs from raw API-loop artifacts.

## One-command run

```bash
bash paper/results/scripts/run_syntax_campaign.sh
```

This performs:
1. `extract_syntax_metrics.py`
2. `compute_syntax_stats.py`
3. `make_syntax_tables.py`
4. `make_syntax_figures.py`
5. deterministic hash check for key stats outputs

## Manual run

```bash
./.venv/bin/python paper/results/scripts/extract_syntax_metrics.py
./.venv/bin/python paper/results/scripts/compute_syntax_stats.py
./.venv/bin/python paper/results/scripts/make_syntax_tables.py
./.venv/bin/python paper/results/scripts/make_syntax_figures.py
```

## Inputs

- `api_loop/Generated_from_Prompts_API_LOOP_*/*/*_refine_manifest.json`
- associated `run_log.json` / `run_meta.json` under archived `refine_runs` folders

## Outputs

- `paper/results/data/prompt_level_syntax_metrics.csv`
- `paper/results/data/iteration_level_syntax_metrics.csv`
- `paper/results/data/model_level_syntax_summary.csv`
- `paper/results/data/error_taxonomy_summary.csv`
- `paper/results/data/stat_tests.json`
- `paper/results/data/campaign_manifest.json`
- `paper/results/tables/*.tex`
- `paper/results/figures/*` + `paper/results/figures/FIGURE_CATALOG.md`

## Dependencies

Install with:

```bash
./.venv/bin/python -m pip install -r paper/results/scripts/requirements.txt
```
