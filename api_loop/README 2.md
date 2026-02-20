API Loop Assets

- `refine_sysml.py`: single-prompt compile-in-the-loop generator.
- `run_refine_sysml_designbench.py`: batch runner over SysMBench prompts.
- `nl_prompts/`: local NL prompt set used by the API loop.
- `Generated_from_Prompts_API_LOOP/`: generated outputs, manifests, and archived refine runs.
- `runs/`: raw run-artifact root for API-loop executions.

Provider support:

- `refine_sysml.py` accepts `--provider openai|anthropic|deepseek_reasoner|mistral_large`.
- `run_refine_sysml_designbench.py` forwards provider settings with the same loop behavior.
- Set credentials in `.env` (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`).
- If `--provider anthropic` is set and `--model` is omitted, the default model becomes `claude-sonnet-4-6`.
- If `--provider deepseek_reasoner` is set and `--model` is omitted, the default model becomes `deepseek-reasoner`.
- If `--provider mistral_large` is set and `--model` is omitted, the default model becomes `mistral-large-latest`.

External dependency path expected by defaults:

- `../sysmbench_original_upstream/dataset/sysml/samples/` (ground-truth sources)

This folder groups the API compiler-in-the-loop workflow assets.
