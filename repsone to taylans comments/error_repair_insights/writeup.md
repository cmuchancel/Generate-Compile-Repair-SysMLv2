# Error Repair Insights (From Iteration Readouts)

This analysis classifies actual compiler error families from iteration logs and quantifies repair effort.

## Core Findings

- Extracted **502 error-family repair episodes**; resolved episodes: **502 (100.0%)**.
- Eventual convergence is 100% in this campaign, so repair burden is captured by additional iterations and elapsed seconds to repair.

Highest-volume error families:
- `parsing-error`: 1666 instances across 247 episodes; mean additional iterations-to-repair 1.43
- `reference-error`: 1224 instances across 145 episodes; mean additional iterations-to-repair 1.32
- `port-definition-owned-usages-not-composite`: 97 instances across 29 episodes; mean additional iterations-to-repair 1.21
- `type-error`: 52 instances across 12 episodes; mean additional iterations-to-repair 1.17
- `feature-chaining-feature-not-one`: 35 instances across 15 episodes; mean additional iterations-to-repair 1.27

Hardest families by additional iterations-to-repair (episodes >= 5):
- `parsing-error`: mean 1.43, median 1.00, P(requires >=2 extra iters) 28.7%
- `reference-error`: mean 1.32, median 1.00, P(requires >=2 extra iters) 24.1%
- `feature-chaining-feature-not-one`: mean 1.27, median 1.00, P(requires >=2 extra iters) 20.0%
- `parameter-membership-owning-type`: mean 1.22, median 1.00, P(requires >=2 extra iters) 22.2%
- `port-definition-owned-usages-not-composite`: mean 1.21, median 1.00, P(requires >=2 extra iters) 20.7%

Longest-persisting families (tail burden, episodes >= 5):
- `parsing-error`: P90 additional iterations 2.40, max additional iterations 7
- `reference-error`: P90 additional iterations 2.00, max additional iterations 4
- `feature-chaining-feature-not-one`: P90 additional iterations 2.00, max additional iterations 3
- `port-definition-owned-usages-not-composite`: P90 additional iterations 2.00, max additional iterations 2
- `parameter-membership-owning-type`: P90 additional iterations 2.00, max additional iterations 2

Longest families by elapsed seconds-to-repair (episodes >= 5):
- `feature-chaining-feature-not-one`: mean 84.1s, median 38.4s
- `invocation-expression-instantiated-type`: mean 81.4s, median 14.8s
- `parsing-error`: mean 55.7s, median 32.4s
- `reference-error`: mean 52.8s, median 28.3s
- `feature-reference-expression-referent-is-feature`: mean 49.6s, median 29.0s

## Figure Map

- `figures/figE01_error_family_pareto_volume.png`
- `figures/figE02_error_family_hardest_by_iterations.png`
- `figures/figE03_error_family_time_to_repair_boxplot.png`
- `figures/figE04_error_family_model_heatmap_mean_iterations.png`
- `figures/figE05_error_family_effort_map.png`
- `figures/figE06_error_family_persistence_curves_top5.png`
- `figures/figE07_error_family_longest_persisting_tail.png`

## Data Map

- `data/error_family_iteration_expanded.csv`
- `data/error_family_repair_episodes.csv`
- `data/error_family_summary.csv`
- `data/error_family_model_summary.csv`
