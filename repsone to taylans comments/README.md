# Response to Taylan's Comments: Subcategory Analysis Pack

This folder contains label-conditioned analysis across all 4 API-loop model roots:

- `openai`
- `anthropic`
- `deepseek_reasoner`
- `mistral_large`

Subcategory labels used:

- `domain` (from `dataset.json`)
- `grammar` (used as the key grammar label, from `dataset.json`)
- `difficulty` bucket (computed from reference SysML design line count, buckets `1..5`)

## Regenerate

```bash
./.venv/bin/python "repsone to taylans comments/scripts/make_subcategory_analysis.py"
```

## Error Repair Insights (from iteration readouts)

Generates figures and data for which compiler error families persist longest and require the most repair effort.

```bash
./.venv/bin/python "repsone to taylans comments/scripts/make_error_repair_insights.py"
```

Identity-level persistence check (same exact diagnostic vs same-family churn):

```bash
./.venv/bin/python "repsone to taylans comments/scripts/analyze_error_identity_persistence.py"
```

Decision-oriented priority plots (what to fix first, with conclusions):

```bash
./.venv/bin/python "repsone to taylans comments/scripts/make_actionable_error_priority_plots.py"
```

User-focused pack (common errors, quickest repairs, and hard-repair behavior):

```bash
./.venv/bin/python "repsone to taylans comments/scripts/make_common_fast_hard_repair_figures.py"
```

Outputs:

- `error_repair_insights/data/*.csv`
- `error_repair_insights/figures/*.png`
- `error_repair_insights/writeup.md`
- `error_repair_insights/identity_persistence_writeup.md`
- `error_repair_insights/actionable_priority_conclusions.md`
- `error_repair_insights/common_fast_hard_takeaways.md`

## Outputs

- `data/merged_prompt_metrics_with_labels.csv`
- `data/summary_domain_by_model.csv`
- `data/summary_grammar_by_model.csv`
- `data/summary_difficulty_by_model.csv`
- `figures/*.png` (13 figures)
- `short_writeup.md` (paper-ready short narrative + key numbers)

## Note

Eventual compile pass is near-ceiling in this campaign due iterative repair loops, so first-shot pass, iterations-to-success, and first-iteration error burden are more discriminative than eventual-pass alone.
