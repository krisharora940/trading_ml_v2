# ML V2

Research workspace for MNQ breakout/retest backtests, model retraining, live-paper engines, and trade export tooling.

## What This Repo Contains

- Deterministic backtest engines for MNQ breakout logic.
- Retraining scripts for `pwin` / entry-quality models.
- HOD/LOD experiment branches under `experiments/`.
- Live paper-trading engines used by `stream_test.py` in the sibling project.
- Export utilities for brokerage-style generic CSV output.
- Saved outputs, experiment snapshots, and changelogs.

This is a research repo, not a packaged library. The code assumes local data/model paths and uses experiment copies rather than a single stable API.

## Current Working Areas

### Core engines
- `bnr_deterministic_engine.py`
- `bnr_deterministic_engine_13feat.py`
- `bnr_deterministic_engine_13feat_merged.py`
- `bnr_deterministic_engine_13feat_baseline_reset.py`

### Retraining
- `retrain_pwin_from_backtest.py`
- `retrain_pwin_13features.py`
- `retrain_pwin_13features_merged.py`
- `retrain_pwin_13features_timefeatures.py`

### HOD/LOD experiments
Main experiment folder:
- `experiments/hodlod_entry_features_retrained/`

Important files in that branch include:
- `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R.py`
- `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_executionflip_20plus.py`
- `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_executionflip_20plus_min20only.py`
- `experiments/hodlod_entry_features_retrained/run_oos_monthly_hodlod_retrained.py`

### Live engines
- `bnr_live_engine_pwin.py`
- `bnr_live_engine_pwin13.py`
- `bnr_live_engine_pwin13_retrained.py`
- `bnr_live_engine_pwin13_hodlod_merged.py`
- `bnr_live_engine_pwin13_hodlod_merged_executionflip_20plus.py`

## Directory Map

- `experiments/`: strategy variants and OOS wrappers
- `outputs/`: backtest results, generic exports, comparison runs
- `saved_versions/`: rollback snapshots of important engines
- `saved_training_sets/`: frozen feature/training artifacts
- `data/`: locally fetched supporting data, including Databento parquet files
- `baselines/`: baseline engine copies

## Model Artifacts

Model files are local `.joblib` artifacts in repo root, for example:
- `entry_model_pwin_13features_retrained.joblib`
- `entry_model_pwin_13features_retrained_hodlod_entry.joblib`
- `entry_model_pwin_13features_retrained_hodlod_entry_merged_live.joblib`

Feature matrices used to train them are also stored in repo root as `.csv`.

## Common Workflows

### 1. Run a backtest experiment
Typical pattern:
- create a new copy under `experiments/...`
- make the change there
- run the script directly, or import `run_engine(...)` from it
- write the result to `outputs/`
- log the change in `CHANGELOG.md` and `LOCAL_CHANGELOG.md`

### 2. Run OOS monthly wrapper
Example:
```bash
python3 experiments/hodlod_entry_features_retrained/run_oos_monthly_hodlod_retrained.py
```

### 3. Export brokerage-style generic CSV
Example utility:
```bash
python3 export_to_generic.py
```

Current generic exports in `outputs/` are typically derived from grouped trade CSVs and expanded to entry/exit rows.

### 4. Fetch missing 2026 Jan-Feb 30s data from Databento
Script:
```bash
python3 fetch_mnq_databento_janfeb_2026.py
```

Environment:
- requires `DATABENTO_API_KEY`

Output:
- `data/mnq_30s_2026-01-01_to_2026-02-28.parquet`

## Execution / Backtest Conventions

A few conventions matter across this repo:
- Many experiment files are intentionally copy-on-write. Do not assume two similarly named files behave identically.
- Recent work favors reversible edits: save a snapshot in `saved_versions/` before touching an important branch.
- `outputs/` contains both valid baselines and abandoned experiments. Read the filename carefully.
- In execution-flip branches, `outcome` may reflect the virtual/original side while `pnl` reflects actual flipped execution. When evaluating those files, actual win rate should be computed from `pnl > 0`.

## Live / Stream Notes

The live paper engines here are consumed by the separate stream runner in:
- `/Users/radhikaarora/Documents/New Project/stream_test.py`

Recent live-related behavior:
- daily backtest-ready market data logging was added there
- Engine I is backed by:
  - `bnr_live_engine_pwin13_hodlod_merged_executionflip_20plus.py`

This repo stores the engine code; the stream runner itself lives outside the repo.

## Changelogs

Use these first when trying to understand recent work:
- `CHANGELOG.md`
- `LOCAL_CHANGELOG.md`

They contain:
- major experiment changes
- rollback notes
- output filenames
- data-source notes

## Recommended Working Style

For new strategy work in this repo:
1. Copy the target experiment file instead of editing the baseline directly.
2. Write outputs to `outputs/` with a descriptive filename.
3. If the change is nontrivial, save a rollback copy in `saved_versions/`.
4. Record the change in both changelogs.
5. Keep data-source assumptions explicit, especially for 2026 `30s` inputs.

## Quick Start

From repo root:
```bash
python3 -m py_compile experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_executionflip_20plus_min20only.py
```

Then run an experiment script or import its `run_engine(...)` function in a one-off Python runner.

## Caveats

- Hardcoded local paths are common.
- There is no single canonical config file for data/model locations.
- Some outputs are diagnostic or provisional rather than production-trustworthy.
- The repo mixes historical experiments, active branches, and export artifacts in the same tree.
