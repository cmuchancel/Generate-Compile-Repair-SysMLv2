# Benchmarking Compiler-in-the-Loop

This repository is now organized into two workflow folders:

- `ai_agent/`: assets and outputs for the agent-driven generation workflow.
- `api_loop/`: API compiler-in-the-loop scripts, prompts, and outputs.
- `sysmbench_original_upstream/`: original SysMBench/DesignBench source repository snapshot (upstream, not our authored work).
- `evaluation_scripts/`: shared evaluation/analysis scripts for both workflows.

## Quick Map

### `ai_agent/`
- `GenerationPrompt.txt`
- `context_examples/`
- `generation_prompts/`
- `Generated_from_Prompts_AI_AGENT/`
- `analysis/`
- `timings.csv`
- `formal-25-09-05.pdf`

### `api_loop/`
- `refine_sysml.py`
- `run_refine_sysml_designbench.py`
- `nl_prompts/`
- `Generated_from_Prompts_API_LOOP_OPENAI/`
- `runs/`

### `sysmbench_original_upstream/`
- Upstream SysMBench/DesignBench repository contents
- Includes `dataset/`, `src/`, prompts, and original artifacts
- Treated as external source material, not new benchmark code authored in this repo

### `evaluation_scripts/`
- `run_sysml_gpt41_eval.py`
- `summarize_gpt41_scores.py`
- `get_domain_metrics.py`
- `get_grammar_metrics.py`
- `get_difficult_metrics.py`
- `verify_final_sysml_checks.py`
- `backfill_refine_timings.py`
- `Evaluation_Prompts/`

See `ai_agent/README.md`, `api_loop/README.md`, `evaluation_scripts/README.md`, and `sysmbench_original_upstream/UPSTREAM_NOTICE.md` for details.
