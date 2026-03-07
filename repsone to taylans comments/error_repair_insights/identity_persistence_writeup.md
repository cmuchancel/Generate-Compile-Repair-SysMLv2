# Identity Persistence Analysis (Same Error vs Same Type)

Question: when an error family persists across iterations, is it the exact same diagnostic recurring, or only the same family label with changing diagnostics?

## Core Result

- Persistent family episodes analyzed (`iterations_with_family >= 2`): **125**.
- **Same exact diagnostic recurs:** 84.8% of persistent episodes.
- **Same template recurs (details changed):** 8.0% of persistent episodes.
- **Family-only churn (no recurring exact/template diagnostic):** 7.2% of persistent episodes.

Interpretation: persistent burden is mostly repeated concrete diagnostics, not just category-level churn.

Top families by persistent episodes (for detailed breakdown):
- `parsing-error` (n=71): exact 78.9%, template 12.7%, churn 8.5%.
- `reference-error` (n=35): exact 91.4%, template 2.9%, churn 5.7%.
- `port-definition-owned-usages-not-composite` (n=6): exact 100.0%, template 0.0%, churn 0.0%.
- `feature-chaining-feature-not-one` (n=3): exact 100.0%, template 0.0%, churn 0.0%.
- `connector-related-features` (n=2): exact 100.0%, template 0.0%, churn 0.0%.
- `parameter-membership-owning-type` (n=2): exact 100.0%, template 0.0%, churn 0.0%.

## Figure Map

- `figures/figE08_identity_persistence_overall.png`
- `figures/figE09_identity_persistence_by_family_top12.png`
- `figures/figE10_identity_persistence_by_model.png`

## Data Map

- `data/error_diagnostic_instances.csv`
- `data/error_identity_episodes.csv`
- `data/error_identity_summary_overall.csv`
- `data/error_identity_summary_by_family.csv`
- `data/error_identity_summary_by_model.csv`
