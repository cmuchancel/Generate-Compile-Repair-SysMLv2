# Benchmarking Compiler-in-the-Loop

This repository captures the workflow we used to benchmark SysML v2 generation quality when driving **Codex GPT‑5.1 in agent mode** against a curated dataset of natural-language prompts and reference designs.

- `context_examples/` – 47 grammar exemplars, each with `nl.txt` + `design.sysml`, that the agent had to load before every batch.
- `generation_prompts/` – directories named after prompt IDs (e.g., `149/`) that contain only an `nl.txt`.
- `Generated_from_Prompts/` – the resulting SysML packages plus ground-truth models and evaluation artifacts.
- `Evaluation_Prompts/` – the precision/recall instructions consumed by the evaluator.
- `run_sysml_gpt41_eval.py`, `summarize_gpt41_scores.py`, and the new `get_*_metrics.py` scripts – tooling for evaluation and reporting.

## Prompting the agent

1. Start inside the repo root and activate the Python environment (`source .venv/bin/activate`).
2. Launch Codex GPT‑5.1 in agent mode and provide `GenerationPrompt.txt` as the governing instructions. The key expectations were:
   - Load every pair in `context_examples/` before modeling a batch.
   - Pull NL-only prompts from `generation_prompts/<ID>/nl.txt`.
   - Model **five prompts per batch**, ensuring each SysML file covers all NL facts and passes `python -m syside check`.
   - Record timings in `timings.csv` and commit per-prompt whenever requested.
3. The agent stored each generated package in `Generated_from_Prompts/<ID>/<ID>.sysml` and left the original NL alongside it for traceability.

## Evaluating generated models

After a batch completed, we ran GPT‑4.1 to judge structural precision/recall:

```bash
OPENAI_API_KEY=... python run_sysml_gpt41_eval.py \
  --generated-root Generated_from_Prompts \
  --reference-root Generated_from_Prompts \
  --precision-prompt Evaluation_Prompts/sysm-eval-p.txt \
  --recall-prompt Evaluation_Prompts/sysm-eval-r.txt \
  --start-id 1 --end-id 151 --skip <omissions>
```

This script renders the evaluation prompts, calls GPT‑4.1, and saves JSON responses (e.g., `107_precision_gpt41.json`). We summarize ranges with:

```bash
python summarize_gpt41_scores.py --root Generated_from_Prompts --start-id 1 --end-id 151
```

## Computing grouped metrics

The `get_domain_metrics.py`, `get_grammar_metrics.py`, and `get_difficult_metrics.py` scripts adapt the original DesignBench analytics to our file layout. Each script:

1. Reads `DesignBench/dataset/sysml/dataset.json` to obtain the canonical NL/design pairs, domain labels, grammar tags, and difficulty buckets (via ground-truth line counts).
2. Aligns each `Generated_from_Prompts/<ID>` directory with the dataset entry by matching the normalized NL text.
3. Recomputes sentence BLEU, ROUGE-L, and BERTScore for the generated SysML, and parses the GPT‑4.1 precision/recall scores from the evaluator outputs.
4. Groups results (by domain, grammar, or difficulty) and writes summaries such as `domain_result.json`.

Example usage:

```bash
python get_domain_metrics.py \
  --dataset DesignBench/dataset/sysml/dataset.json \
  --generated-root Generated_from_Prompts \
  --output domain_result.json

python get_grammar_metrics.py --output grammar_result.json
python get_difficult_metrics.py --output difficult_result.json
```

These reports guided qualitative reviews and highlighted where the agent underperformed across domains, grammar constructs, or complexity tiers.
