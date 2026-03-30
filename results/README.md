# Results Directory

Model notebooks should write their persisted outputs here so later comparison notebooks can read consistent artifacts without depending on notebook cell state.

Recommended structure:

- `results/logistic_regression_baseline/`
- `results/random_forest/`
- `results/hist_gradient_boosting/`
- `results/extra_trees/`

Each model-specific folder should contain:

- `run_metadata.csv`
- `overview.csv`
- `split_summary.csv`
- `validation_metrics.csv`
- `validation_threshold_curve.csv`
- `test_metrics.csv`
- `test_risk_scores.csv.gz`
- `top_risk_rows.csv`
- `feature_importance.csv` when the model supports it
- saved evaluation plots as `.png`
