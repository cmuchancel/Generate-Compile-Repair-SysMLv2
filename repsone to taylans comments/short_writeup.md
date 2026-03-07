# Response To Taylan's Comments: Subcategory Analysis Draft

This note adds a label-conditioned analysis using SysMBench metadata:
- Domain label (`domain`)
- Key grammar label (`grammar`)
- Difficulty bucket (computed from reference SysML length, buckets 1-5)

Data source for the 4-model comparison:
- `paper/results/data/prompt_level_syntax_metrics.csv` (151 prompts x 4 models = 604 rows)

## Suggested Results Text (Short)

Across all four LLM pipelines, first-shot behavior and repair burden vary systematically by domain, key grammar label, and difficulty.
Because eventual compile pass is near-ceiling under iterative repair, first-shot pass, iterations-to-success, and first-iteration error burden are the most informative differentiators.
Domain-conditioned and grammar-conditioned results are heterogeneous: some labels compile on first shot consistently, while others require materially more repair iterations.
Difficulty buckets (proxied by reference SysML line-count complexity) also show non-uniform first-shot pass and error burden.
These findings support reporting subcategory-stratified performance rather than only aggregate eventual compile rates.

## Quick Numerical Highlights (Macro-averaged across models)

Top domains by first-shot pass rate:
- Embedded device: first-shot 75.0% | mean iters-to-success 1.25 | mean first-iter errors 0.25
- Water resource transportation: first-shot 75.0% | mean iters-to-success 1.25 | mean first-iter errors 2.50
- Confidentiality and security: first-shot 62.5% | mean iters-to-success 1.88 | mean first-iter errors 3.75

Lowest domains by first-shot pass rate:
- Medical Health: first-shot 25.0% | mean iters-to-success 2.25 | mean first-iter errors 2.50
- Systems Engineering: first-shot 25.0% | mean iters-to-success 2.62 | mean first-iter errors 7.25
- Fault diagnosis: first-shot 25.0% | mean iters-to-success 2.00 | mean first-iter errors 4.33

Top grammar labels by first-shot pass rate:
- Item: first-shot 100.0% | mean iters-to-success 1.00 | mean first-iter errors 0.00
- Individual and Snapshot: first-shot 100.0% | mean iters-to-success 1.00 | mean first-iter errors 0.00
- Conditional Succession: first-shot 87.5% | mean iters-to-success 1.12 | mean first-iter errors 0.50

Lowest grammar labels by first-shot pass rate:
- Analysis and Trade: first-shot 31.2% | mean iters-to-success 2.06 | mean first-iter errors 1.94
- Metadata: first-shot 25.0% | mean iters-to-success 2.38 | mean first-iter errors 5.75
- Transition: first-shot 25.0% | mean iters-to-success 2.17 | mean first-iter errors 11.00

Difficulty trend (macro-averaged across models):
- Difficulty 1: first-shot 50.8% | mean iterations-to-success 1.77 | mean first-iter errors 3.18
- Difficulty 2: first-shot 52.7% | mean iterations-to-success 1.67 | mean first-iter errors 3.37
- Difficulty 3: first-shot 41.7% | mean iterations-to-success 1.94 | mean first-iter errors 9.38
- Difficulty 4: first-shot 50.0% | mean iterations-to-success 1.67 | mean first-iter errors 1.58
- Difficulty 5: first-shot 58.3% | mean iterations-to-success 1.58 | mean first-iter errors 2.33

## Figure/Data Map

- `figures/fig01_domain_eventual_pass_heatmap.png`
- `figures/fig02_domain_first_shot_pass_heatmap.png`
- `figures/fig03_domain_iterations_to_success_heatmap.png` (combo: heatmap + all-model average bar chart with ±SE)
- `figures/fig04_grammar_topK_eventual_pass_heatmap.png`
- `figures/fig05_grammar_topK_first_shot_pass_heatmap.png`
- `figures/fig06_grammar_topK_iterations_to_success_heatmap.png` (combo: heatmap + all-model average bar chart)
- `figures/fig07_difficulty_eventual_pass_grouped_bar.png`
- `figures/fig08_difficulty_first_shot_pass_grouped_bar.png`
- `figures/fig09_difficulty_mean_iterations_to_success_grouped_bar.png` (combo: grouped bars + all-model trendline with $R^2$)
- `figures/fig10_label_sample_size_panels.png`
- `figures/fig11_domain_first_iteration_error_heatmap.png`
- `figures/fig12_grammar_topK_first_iteration_error_heatmap.png`
- `figures/fig13_difficulty_first_iteration_error_grouped_bar.png`
- `data/merged_prompt_metrics_with_labels.csv`
- `data/summary_domain_by_model.csv`
- `data/summary_grammar_by_model.csv`
- `data/summary_difficulty_by_model.csv`
