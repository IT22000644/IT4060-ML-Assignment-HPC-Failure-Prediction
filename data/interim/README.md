# Interim Data

This folder stores cleaned and reshaped datasets created from the raw LANL files.

Expected exports from the notebooks:

- `failure_system20_nodes_0_255.csv`
- `usage_jobs_clean.csv.gz`
- `usage_node_events.csv.gz`
- `failure_system20_clean.csv`
- `usage_node_events_clean.csv.gz`

Purpose:

- keep expensive parsing steps out of later notebooks
- make data cleaning reproducible
- separate raw source files from cleaned analysis-ready tables

The raw files in `data/raw/` should remain unchanged.
