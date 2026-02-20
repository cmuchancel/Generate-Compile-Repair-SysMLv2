Evaluation Scripts

This folder contains evaluation and analysis scripts that work for both output layouts:

- AI-agent outputs: `ai_agent/Generated_from_Prompts_AI_AGENT`
- API-loop outputs: `api_loop/Generated_from_Prompts_API_LOOP`
- Dataset defaults: `sysmbench_original_upstream/dataset/sysml/dataset.json`

Scripts

- `run_sysml_gpt41_eval.py`: precision/recall judge runner.
- `summarize_gpt41_scores.py`: aggregate precision/recall summaries.
- `get_domain_metrics.py`: domain-bucket metrics from score JSONs.
- `get_grammar_metrics.py`: grammar-bucket metrics from score JSONs.
- `get_difficult_metrics.py`: difficulty-bucket metrics from score JSONs.
- `verify_final_sysml_checks.py`: `syside` validation for final generated SysML.
- `backfill_refine_timings.py`: backfill timing CSVs from refine logs/manifests.

By default, the grouped metric scripts write outputs into:
- `<generated-root-parent>/analysis/domain_result.json`
- `<generated-root-parent>/analysis/grammar_result.json`
- `<generated-root-parent>/analysis/difficult_result.json`

Examples

- Evaluate AI-agent outputs:
  `python evaluation_scripts/run_sysml_gpt41_eval.py --generated-root ai_agent/Generated_from_Prompts_AI_AGENT --reference-root ai_agent/Generated_from_Prompts_AI_AGENT`

- Evaluate API-loop outputs:
  `python evaluation_scripts/run_sysml_gpt41_eval.py --generated-root api_loop/Generated_from_Prompts_API_LOOP --reference-root api_loop/Generated_from_Prompts_API_LOOP`

- Verify SysML checks (works for both):
  `python evaluation_scripts/verify_final_sysml_checks.py --output-root ai_agent/Generated_from_Prompts_AI_AGENT --venv .venv`
  `python evaluation_scripts/verify_final_sysml_checks.py --output-root api_loop/Generated_from_Prompts_API_LOOP --venv .venv`
