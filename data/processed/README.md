# Processed Data

This folder stores final model-ready datasets created after feature engineering.

Expected exports:

- `node_hour_features_6h.csv.gz`

These files are intended for:

- train/test splitting
- model training
- evaluation

Unlike `data/interim/`, these tables should already contain engineered features and labels.
