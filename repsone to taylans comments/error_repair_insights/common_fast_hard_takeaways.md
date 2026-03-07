# Common vs Quick vs Hard Repair: What To Take Away

This section answers three direct questions:
1. Which errors are common?
2. Which errors are repaired quickest?
3. If not repaired immediately, what does repair behavior look like?

## 1) Common Errors

- `parsing-error`: 1144 single-shot instances, 1666 total instances, across 126 prompts.
- `reference-error`: 824 single-shot instances, 1224 total instances, across 92 prompts.
- `port-definition-owned-usages-not-composite`: 82 single-shot instances, 97 total instances, across 29 prompts.
- `type-error`: 26 single-shot instances, 52 total instances, across 10 prompts.
- `feature-chaining-feature-not-one`: 28 single-shot instances, 35 total instances, across 13 prompts.

## 2) Quickest Repairs (episodes >= 5)

- `parameter-membership-owning-type`: median 12.4s, repaired in <=1 iteration 78% of episodes.
- `type-error`: median 12.8s, repaired in <=1 iteration 83% of episodes.
- `invocation-expression-instantiated-type`: median 14.8s, repaired in <=1 iteration 91% of episodes.
- `reference-error`: median 28.3s, repaired in <=1 iteration 76% of episodes.
- `feature-reference-expression-referent-is-feature`: median 29.0s, repaired in <=1 iteration 88% of episodes.

## 3) Hard/Not-Immediate Repairs

- Unrepaired by campaign end: **0/502 episodes**.
- Because unresolved-by-end is zero here, hard repair is best viewed as multi-iteration persistence.
- Among persistent episodes: same exact diagnostic recurs in **84.8%**, template recurrence in **8.0%**, family-only churn in **7.2%**.

Slowest repairs by median time (episodes >= 5):
- `feature-chaining-feature-not-one`: median 38.4s, mean iterations 1.27.
- `port-definition-owned-usages-not-composite`: median 33.4s, mean iterations 1.21.
- `parsing-error`: median 32.4s, mean iterations 1.43.
- `feature-reference-expression-referent-is-feature`: median 29.0s, mean iterations 1.12.
- `reference-error`: median 28.3s, mean iterations 1.32.

## Figure Map

- `figures/figE15_common_errors_volume_and_coverage.png`
- `figures/figE16_repair_speed_fast_vs_slow.png`
- `figures/figE17_repair_completion_curves_top_common.png`
- `figures/figE18_unrepaired_status_and_persistence_behavior.png`
- `figures/figE19_hard_repair_rate_by_family.png`

## Data Map

- `data/error_common_fast_hard_summary.csv`
