# Actionable Conclusions: Which Error Types Matter Most

## What To Care About

- Prioritize error types by **total extra iterations** and **total repair time** (not raw count alone).
- Favor high-repeatability families first: repeated exact diagnostics are best candidates for deterministic fixes.
- Separate rare-but-expensive families from frequent families; each needs different intervention.

## High-Value Findings

- `parsing-error` alone accounts for **52.1%** of extra iterations and **50.4%** of repair time.
- `parsing-error` + `reference-error` together account for **80.2%** of extra iterations and **78.4%** of repair time.
- Top-3 families reach **85.4%** of extra-iteration burden; top-5 reach **90.3%**.

Most repeatable families (persistent episodes >=3):
- `port-definition-owned-usages-not-composite`: exact recurrence 100.0%, persistent episodes 6, burden share 5.2%.
- `feature-chaining-feature-not-one`: exact recurrence 100.0%, persistent episodes 3, burden share 2.8%.
- `reference-error`: exact recurrence 91.4%, persistent episodes 35, burden share 28.2%.
- `parsing-error`: exact recurrence 78.9%, persistent episodes 71, burden share 52.1%.

Slow-to-repair families (episodes >=3):
- `connector-related-features`: mean repair time 204.6s, mean additional iterations 2.00.
- `feature-chaining-feature-not-one`: mean repair time 84.1s, mean additional iterations 1.27.
- `invocation-expression-instantiated-type`: mean repair time 81.4s, mean additional iterations 1.09.
- `quantity-operator-expression`: mean repair time 63.5s, mean additional iterations 1.00.
- `parsing-error`: mean repair time 55.7s, mean additional iterations 1.43.

## Suggested Intervention Priority

1. Build hard-coded repair rules and prompt checks for parsing/reference diagnostics first (highest ROI).
2. Add targeted remediation templates for feature/connector families with high repeatability but lower frequency.
3. Add specialized handling for rare high-latency families to cut wall-clock tail.

## Figure Map

- `figures/figE11_error_burden_pareto_iterations_time.png`
- `figures/figE12_error_priority_matrix_bubble.png`
- `figures/figE13_model_burden_concentration_stacked.png`
- `figures/figE14_top10_burden_vs_exact_recurrence.png`

## Data Map

- `data/error_priority_summary.csv`
