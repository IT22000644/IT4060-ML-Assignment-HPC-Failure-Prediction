# Processed Data

This folder stores final model-ready datasets created after feature engineering.

Expected exports:

- `node_hour_features_multi_horizon.csv.gz`

These files are intended for:

- train/test splitting
- model training
- evaluation

Unlike `data/interim/`, these tables should already contain engineered features and labels.

The multi-horizon dataset includes:

- engineered node-hour workload features
- lag and rolling features
- `next_failure_time`
- `hours_to_next_failure`
- `label_next_1h`
- `label_next_6h`
- `label_next_12h`
- `label_next_24h`
