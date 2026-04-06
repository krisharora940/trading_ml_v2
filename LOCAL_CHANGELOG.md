# Local Changelog (Manual)

All entries are local-only for easy rollback. No training joblibs were overwritten.

## 2026-03-27
- Created baseline reset script (non-retrained model) and organized in `baselines/`.
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/baselines/bnr_deterministic_engine_13feat_baseline_reset.py`
  - Uses `entry_model_pwin_13features.joblib` (non-retrained).
  - Outputs:
    - `output_bnr_det_2025_13feat_pwin50_baseline_reset.csv`
- Updated baseline script to use quarterly 2025 1m/30s data files:
  - 1m: `mnq_1m_2025_q1.csv` .. `mnq_1m_2025_q4.csv`
  - 30s: `mnq_30s_2025_q1.csv` .. `mnq_30s_2025_q4.csv`
- Ran baseline reset script (2025 only). Output saved at:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/output_bnr_det_2025_13feat_pwin50_baseline_reset.csv`

## 2026-03-26
- Created copies for time-feature experiment:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/bnr_deterministic_engine_13feat_timefeatures.py`
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/retrain_pwin_13features_timefeatures.py`
- Added time-based features in timefeatures experiment only:
  - `min_since_pivot`, `min_since_930`
- Trained and saved timefeature model (new file, no overwrite):
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pwin_13features_retrained_timefeatures.joblib`
  - Feature matrix: `/Users/radhikaarora/Documents/Trading ML/ML V2/ml_features_13feat_retrain_timefeatures.csv`
- Ran timefeatures backtest outputs:
  - `output_bnr_det_2025_13feat_pwin50_timefeatures.csv`
  - `output_bnr_det_2026_13feat_pwin50_timefeatures.csv`
  - `output_bnr_det_mar2026_13feat_pwin50_timefeatures.csv`
- Created merged retrain experiment (new files only):
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/retrain_pwin_13features_merged.py`
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pwin_13features_retrained_merged.joblib`
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/ml_features_13feat_retrain_merged.csv`
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/bnr_deterministic_engine_13feat_merged.py`
  - Outputs:
    - `output_bnr_det_2025_13feat_pwin50_merged.csv`
    - `output_bnr_det_2026_13feat_pwin50_merged.csv`
    - `output_bnr_det_mar2026_13feat_pwin50_merged.csv`
- Created experiments workspace for HOD/LOD entry feature work:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features/`
  - Copied scripts into it (no changes yet):
    - `retrain_pwin_13features_hodlod.py`
    - `bnr_deterministic_engine_13feat_hodlod.py`


## 2026-03-27 (hodlod entry, no-gates v2)
- Removed retrace/zone invalidation gates in `experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained.py` (no max retrace reset, no far-side zone invalidation).
- Ran HOD/LOD entry retrained model sweep (pwin >= 0.40/0.50/0.60/0.70) with no-gates in 2025, 2026 Jan-Feb, 2026 Mar.
- Outputs:
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_13feat_pwin40_retrained_hodlod_entry_nogates_v2.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_13feat_pwin50_retrained_hodlod_entry_nogates_v2.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_13feat_pwin60_retrained_hodlod_entry_nogates_v2.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_13feat_pwin70_retrained_hodlod_entry_nogates_v2.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2026_janfeb_13feat_pwin40_retrained_hodlod_entry_nogates_v2.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2026_janfeb_13feat_pwin50_retrained_hodlod_entry_nogates_v2.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2026_janfeb_13feat_pwin60_retrained_hodlod_entry_nogates_v2.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2026_janfeb_13feat_pwin70_retrained_hodlod_entry_nogates_v2.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2026_mar_13feat_pwin40_retrained_hodlod_entry_nogates_v2.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2026_mar_13feat_pwin50_retrained_hodlod_entry_nogates_v2.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2026_mar_13feat_pwin60_retrained_hodlod_entry_nogates_v2.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2026_mar_13feat_pwin70_retrained_hodlod_entry_nogates_v2.csv

## 2026-03-28
- Patched `experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained.py`
  to honor `PWIN_MODEL_PATH` so monthly OOS wrapper runs can actually swap models.
- Patched `experiments/hodlod_entry_features_retrained/retrain_pwin_13features_hodlod_retrained.py`
  to accept `--base-features-csv` and merge an existing saved feature matrix before training.
- Patched `experiments/hodlod_entry_features_retrained/run_oos_monthly_hodlod_retrained.py`
  to accept:
  - `--backtest-csv`
  - `--base-features-csv`
  - `--tag`
- Existing saved training snapshot preserved and reused:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_training_sets/snapshot_2026-03-27/ml_features_13feat_retrain_hodlod_entry_retrainedset.csv`
- January 2025 exclusionary OOS comparison runs created in isolated folders:
  - Merged snapshot + latest baseline run:
    - model: `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_training_sets/oos_models_compare_testjan/entry_model_pwin_13features_retrained_hodlod_entry_retrainedset_merged_snapshot_plus_commit4d779f0_testjan_excl_2025-01.joblib`
    - features: `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_training_sets/oos_models_compare_testjan/ml_features_13feat_retrain_hodlod_entry_retrainedset_merged_snapshot_plus_commit4d779f0_testjan_excl_2025-01.csv`
    - output: `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_testjan/output_bnr_det_2025_01_13feat_pwin40_hodlod_merged_snapshot_plus_commit4d779f0_testjan_oos.csv`
  - Latest baseline run only:
    - model: `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_training_sets/oos_models_compare_testjan/entry_model_pwin_13features_retrained_hodlod_entry_retrainedset_recent_run_only_testjan_excl_2025-01.joblib`
    - features: `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_training_sets/oos_models_compare_testjan/ml_features_13feat_retrain_hodlod_entry_retrainedset_recent_run_only_testjan_excl_2025-01.csv`
    - output: `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_testjan/output_bnr_det_2025_01_13feat_pwin40_hodlod_recent_run_only_testjan_oos.csv`
- Built static live-paper merged HOD/LOD training artifacts for stream integration:
  - model: `/Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pwin_13features_retrained_hodlod_entry_merged_live.joblib`
  - features: `/Users/radhikaarora/Documents/Trading ML/ML V2/ml_features_13feat_retrain_hodlod_entry_merged_live.csv`
- Added live engine file:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/bnr_live_engine_pwin13_hodlod_merged.py`
- Integrated it into:
  - `/Users/radhikaarora/Documents/New Project/stream_test.py`
- Live engine behavior:
  - merged HOD/LOD pwin model
  - threshold `0.40`
  - no slippage
  - no deterministic retrace / prior-candle gate (matches current no-gates backtest entry filter)
  - `MAX_RISK_DOLLARS=500`, `MAX_CONTRACTS=250`
- Added copy-only experiment engine:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained_sizegate20.py`
- Purpose:
  - block oversized trades with weak pwin in 2026 analysis
- Rule:
  - `contracts >= 20` requires `pwin_score >= 0.50`
  - contracts below 20 still use the normal engine threshold passed into `run_engine(...)`
- This was implemented as a separate copy to keep reversion trivial.
- Extended the copy-only size-gated engine rules:
  - `contracts >= 20` requires `pwin_score >= 0.50`
  - `contracts > 35` requires `pwin_score >= 0.60`
- Implemented as a cascading minimum-pwin rule in:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained_sizegate20.py`
- Promoted the merged-training-set HOD/LOD OOS engine to the active baseline for new edits.
- Saved baseline snapshot before size gating:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/merged_oos_baseline_pre_sizegates_2026-03-28/bnr_deterministic_engine_13feat_hodlod_retrained.py`
- Edited active baseline engine in place:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained.py`
- Active size-gate rules now:
  - `contracts >= 20` requires `pwin_score >= 0.50`
  - `contracts > 35` requires `pwin_score >= 0.60`
- The copy-only experiment file still exists, but the main engine is now the authoritative baseline for this rule set.
- Switched active merged baseline engine from hard size-gate skips to phantom trades.
- Phantom behavior:
  - oversized low-pwin trades still run through the engine and affect state/pathing
  - realized trade rows get `phantom_trade=True`, `raw_pnl=<actual>`, `pnl=0.0`
- Direct 2026 no-wrapper phantom run completed with fixed merged model.
- Important validation note:
  - grouped scale-out leg quantities can undercount the original entry contracts
  - use entry sizing logic (`risk`, risk budget) rather than summed leg quantities when checking whether a trade should have been size-gated.

- Fixed scale-out carry-forward bookkeeping in both the active phantom engine and the saved non-phantom comparator.
- Bug: after scale1/scale2, the engine was carrying only `next_q` forward as the active trade instead of the full remaining quantity.
- Result: recorded leg quantities and later forced-close rows could understate the real remaining position.
- Fix: carry `remaining_qty = sum(q for q, _ in scale_out_plan[scale_out_stage:])` and set the remaining trade `contracts` / `risk_dollars` from that value.
- Reran direct 2026 no-wrapper tests after the fix.
- Important note for future debugging:
  - when size-gated phantom trades appear inconsistent, first check whether the comparison is against the original baseline or against a hard-skip comparator, because hard-skip changes the later trade path.

- Added a new scale-out experiment copy based on the clean no-size-gates + scaleout-fix baseline:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained_scaleout75_30to79_4R_80plus_5R20R.py`
- Scale-out rules in this copy:
  - `contracts >= 80`: `75% @ 1.2R`, `10% @ 5R`, `15% @ 20R`
  - `30 <= contracts < 80`: `75% @ 1.2R`, `25% @ 4R`
  - `contracts < 30`: preserved prior small-trade behavior
- Direct 2026 no-wrapper outputs for this experiment:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_origlogic_scaleoutfix_scaleout75_2026/output_bnr_det_2026-01_origlogic_scaleout75.csv`
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_origlogic_scaleoutfix_scaleout75_2026/output_bnr_det_2026-02_origlogic_scaleout75.csv`
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_origlogic_scaleoutfix_scaleout75_2026/output_bnr_det_2026-03_origlogic_scaleout75.csv`

- Added a follow-up scale-out experiment copy for 2026 only:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained_30to79_4R_80plus_50at12_50at10R.py`
- Rules in this copy:
  - `contracts >= 80`: `50% @ 1.2R`, `50% @ 10R`
  - `30 <= contracts < 80`: `75% @ 1.2R`, `25% @ 4R`
  - `contracts < 30`: preserved prior small-trade behavior
- Direct 2026 outputs:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_origlogic_scaleout50_10R_2026/output_bnr_det_2026-01_origlogic_scaleout50_10R.csv`
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_origlogic_scaleout50_10R_2026/output_bnr_det_2026-02_origlogic_scaleout50_10R.csv`
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_origlogic_scaleout50_10R_2026/output_bnr_det_2026-03_origlogic_scaleout50_10R.csv`

- Added a third 80+ scale-out experiment copy for 2026 only:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained_30to79_4R_80plus_75at12_10at5_15at10.py`
- Rules in this copy:
  - `contracts >= 80`: `75% @ 1.2R`, `10% @ 5R`, `15% @ 10R`
  - `30 <= contracts < 80`: `75% @ 1.2R`, `25% @ 4R`
  - `contracts < 30`: preserved prior small-trade behavior
- Direct 2026 outputs:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_origlogic_scaleout75_5R10R_2026/output_bnr_det_2026-01_origlogic_scaleout75_5R10R.csv`
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_origlogic_scaleout75_5R10R_2026/output_bnr_det_2026-02_origlogic_scaleout75_5R10R.csv`
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_origlogic_scaleout75_5R10R_2026/output_bnr_det_2026-03_origlogic_scaleout75_5R10R.csv`

- Added a 2026-only experiment copy for large-trade de-risking:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained_gt35_50at4R_50at65R_hard3000.py`
- Rules in this copy:
  - `contracts > 35`: remove scale1/scale2 and use `50% @ 4R`, `50% @ 6.5R`
  - add per-trade hard wick stop at `$3,000` max loss
  - other trades keep the repaired baseline scale-out rules
- Direct 2026 outputs:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_gt35_4R65R_hard3000_2026/output_bnr_det_2026-01_gt35_4R65R_hard3000.csv`
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_gt35_4R65R_hard3000_2026/output_bnr_det_2026-02_gt35_4R65R_hard3000.csv`
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_gt35_4R65R_hard3000_2026/output_bnr_det_2026-03_gt35_4R65R_hard3000.csv`

- Patched the non-retrained 13-feature no-reset engine with the same scale-out carry-forward fix.
  - Engine: `/Users/radhikaarora/Documents/Trading ML/ML V2/bnr_deterministic_engine_13feat_noreset.py`
  - Snapshot before fix: `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/bnr_deterministic_engine_13feat_noreset_pre_scaleoutfix_2026-03-28.py`
- Ran 2025 only on quarterly 1m/30s market data.
  - Output: `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_13feat_pwin50_nonretrained_noreset_scaleoutfix.csv`

- 2026-03-28: Patched baseline reset-enabled non-retrained 13-feature engine for scale-out carry-forward. Snapshot: /Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/bnr_deterministic_engine_13feat_baseline_reset_pre_scaleoutfix_2026-03-28.py. Output: /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_13feat_pwin50_baseline_reset_scaleoutfix.csv.

- 2026-03-28: Non-retrained HOD/LOD baseline rebuild. Feature CSV: /Users/radhikaarora/Documents/Trading ML/ML V2/ml_features_13features_hodlod_baseline.csv. Model: /Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pwin_13features_hodlod_baseline.joblib. Engine snapshot: /Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/bnr_deterministic_engine_13feat_hodlod_pre_baseline_hodlod_scaleoutfix_2026-03-28.py. Output: /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_15feat_baseline_hodlod_scaleoutfix.csv.

- 2026-03-28: Baseline HOD/LOD no-gates variant. Removed deterministic retrace/color gates, kept PWIN=0.50. Snapshot: /Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/bnr_deterministic_engine_13feat_hodlod_pre_nogates_pwin40_2026-03-28.py. Output: /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_15feat_baseline_hodlod_scaleoutfix_nogates_pwin50.csv.

- 2026-03-28: 2025 month-excluded OOS run completed for corrected 15-feature baseline no-gates model at PWIN=0.50. Wrapper: /Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features/run_oos_monthly_hodlod_baseline_nogates.py. Outputs: /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_monthly_hodlod_baseline_nogates. Models/features: /Users/radhikaarora/Documents/Trading ML/ML V2/saved_training_sets/oos_models_hodlod_baseline_nogates.

- 2026-03-28: New corrected OOS-derived 15-feature no-gates retrain artifacts created. Combined source: /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_15feat_baseline_hodlod_scaleoutfix_nogates_pwin50_ooswrapper_combined.csv. Features: /Users/radhikaarora/Documents/Trading ML/ML V2/ml_features_15feat_hodlod_baseline_nogates_oos2025_retrained.csv. Model: /Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pwin_15features_hodlod_baseline_nogates_oos2025_retrained.joblib.

- 2026-03-29: Created new experiment copy from saved merged baseline snapshot `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/merged_oos_baseline_pre_sizegates_scaleoutfix_only_2026-03-28.py` at `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained_gt35_4_6p5_10_20_be_after_6p5.py`. Set default `PWIN_THRESH_DEFAULT = 0.40`. Added custom `>35` contract scale-out ladder: `25% @ 4R`, `25% @ 6.5R`, `25% @ 10R`, `25% @ 20R`. After the `6.5R` leg, remaining size now moves stop to break-even and exits on wick touch at entry price. Saved snapshot left untouched for easy rollback.

- 2026-03-29: Saved `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained_sizegate20.py` to `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/bnr_deterministic_engine_13feat_hodlod_retrained_sizegate20_pre_remove_sizegates_2026-03-29.py`, then removed the 20+ and >35 size-based pwin tightening so the file now uses only the base `pwin_thresh` gate.

- 2026-03-29: Promoted the verified correct merged 2026 baseline to a clearly named working copy: `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/correct_baseline_w_correcting_scaleout_edits.py`. This copy comes directly from `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/merged_oos_baseline_pre_sizegates_scaleoutfix_only_2026-03-28.py` and matches the archived correct 2026 rerun when used with `/Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pwin_13features_retrained_hodlod_entry_merged_live.joblib` at `pwin=0.40`.

- 2026-03-29: Updated `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/correct_baseline_w_correcting_scaleout_edits.py` to use the merged-live model by default and added the requested `35+` contract exit ladder: `25% @ 4R`, `25% @ 6.5R`, then stop to break-even on wick for the remaining size, then `25% @ 10R`, `25% @ 20R`.

- 2026-03-29: Created `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R.py` as a copy of the verified correct baseline, modified so all trades exit 100% at `1.2R`.

- 2026-03-29: Updated `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R.py` so prices are tick-rounded to valid `0.25` increments and the minimum profit target distance is now `max(1.2R, 2.5 pts)`.

- 2026-03-29: Fixed `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R.py` so stored/exported `pnl` is in true MNQ dollars (`points * contracts * 2`) instead of half-sized point-contract units.

- 2026-03-29: Created `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_tickround_only.py` to isolate tick rounding only, with true dollar PnL storage but without the 2.5-point minimum target floor.

- Packaged the current full-exit 1.2R strategy as a new baseline folder.
  - Folder: `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/fixed_high_win_rate_baseline`
  - Assumptions: 1-tick slippage, tick-rounded entry/stop/target/exit, minimum target distance = max(1.2R, 2.5 points), true-dollar exported PnL.
  - Included raw backtest CSVs for 2024-2026, engine copy, and generic exports with no commissions/fees.

- Exploratory target-timestamp patch on the full-exit 1.2R strategy.
  - Edited: `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R.py`
  - Snapshot for reversal: `saved_versions/correct_baseline_full_exit_at_1p2R_pre_targettimefix_2026-03-29.py`
  - Change 1: target-hit helper now refuses bars whose timestamp is before the trade activation time.
  - Change 2: target-hit helper is constrained to the current 1-minute bar window instead of scanning earlier minutes.
  - Test rerun output: `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_targettimefix.csv`
  - Result: this was kept exploratory only; the November 5 / November 6 impossible target examples still survived, so this is not yet accepted as the new baseline.
  - To reverse: copy `saved_versions/correct_baseline_full_exit_at_1p2R_pre_targettimefix_2026-03-29.py` back over `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R.py`.

- Applied the minimal exit-only fix for the full-exit 1.2R strategy after debugging impossible green exits.
  - Edited: `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R.py`
  - Snapshot for reversal: `saved_versions/correct_baseline_full_exit_at_1p2R_pre_targettimefix_2026-03-29.py`
  - Root cause: the trade-management block was running on 30-second events while reusing stale 1-minute OHLC values from the prior 1-minute event.
  - Minimal fix: gated the trade-management block to `ev["kind"] == "1m"` so exits are evaluated only on 1-minute events, matching the code comment and preserving entry logic.
  - Clean rerun output: `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_1mexitfix.csv`
  - To reverse: copy `saved_versions/correct_baseline_full_exit_at_1p2R_pre_targettimefix_2026-03-29.py` back over `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R.py`.

- Added a separate size-compression experiment on top of the corrected 1m-exitfix full-exit 1.2R strategy.
  - New experiment file: `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_sizecompress.py`
  - Compression function: monotone power-law compression anchored at 20 contracts with beta=0.75.
  - Examples: 20->20, 25->23, 30->27, 40->33, 50->39, 60->45, 80->56, 100->66, 150->90, 250->132.
  - 2025 rerun output: `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_1mexitfix_sizecompress.csv`
  - This experiment leaves entries unchanged and only compresses position size after raw contract sizing is computed.

- Added a follow-on experiment widening stop and target on top of the corrected 1m-exitfix size-compressed strategy.
  - New experiment file: `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_sizecompress_widerisk.py`
  - Uses the same monotone size compression, then widens stop distance and target distance by sqrt(raw_contracts / adjusted_contracts).
  - Entries remain unchanged; this only alters size plus exit geometry after raw sizing is computed.
  - 2025 rerun output: `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_1mexitfix_sizecompress_widerisk.csv`
- Added `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_sizecompress_widerisk_fullratio.py` as a separate experiment from the 1m-exitfix branch. It uses more aggressive size compression (`SIZE_COMPRESS_BETA = 0.65`) and widens stop/target geometry by the full raw-to-adjusted size ratio instead of the earlier sqrt damped factor. 2025 output copied to `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_1mexitfix_sizecompress_widerisk_fullratio.csv`.
- Added two more follow-on experiments from the corrected 1m-exitfix branch to keep compressing 20+ trades while widening stop/target geometry by the full raw-to-adjusted size ratio.
  - `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_sizecompress_widerisk_fullratio_beta060.py` uses `SIZE_COMPRESS_BETA = 0.60`.
  - `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_sizecompress_widerisk_fullratio_beta055.py` uses `SIZE_COMPRESS_BETA = 0.55`.
  - 2025 outputs copied to:
    - `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_1mexitfix_sizecompress_widerisk_fullratio_beta060.csv`
    - `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_1mexitfix_sizecompress_widerisk_fullratio_beta055.csv`
- Added three more follow-on experiments from the corrected 1m-exitfix branch using full-ratio stop/target widening with stronger monotone size compression.
  - `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_sizecompress_widerisk_fullratio_beta0050.py` uses `SIZE_COMPRESS_BETA = 0.50`.
  - `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_sizecompress_widerisk_fullratio_beta0045.py` uses `SIZE_COMPRESS_BETA = 0.45`.
  - `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_sizecompress_widerisk_fullratio_beta0040.py` uses `SIZE_COMPRESS_BETA = 0.40`.
  - 2025 outputs copied to:
    - `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_1mexitfix_sizecompress_widerisk_fullratio_beta050.csv`
    - `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_1mexitfix_sizecompress_widerisk_fullratio_beta045.csv`
    - `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_1mexitfix_sizecompress_widerisk_fullratio_beta040.csv`
- Added two more follow-on experiments from the corrected 1m-exitfix branch using full-ratio stop/target widening with even stronger monotone size compression.
  - `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_sizecompress_widerisk_fullratio_beta00035.py` uses `SIZE_COMPRESS_BETA = 0.35`.
  - `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_sizecompress_widerisk_fullratio_beta00030.py` uses `SIZE_COMPRESS_BETA = 0.30`.
  - 2025 outputs copied to:
    - `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_1mexitfix_sizecompress_widerisk_fullratio_beta035.csv`
    - `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_1mexitfix_sizecompress_widerisk_fullratio_beta030.csv`
- Added `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_sizecompress_widerisk_fullratio_beta00030_target135_20plus.py` as a separate experiment from the `beta00030` branch.
  - Keeps `SIZE_COMPRESS_BETA = 0.30` and full-ratio stop/target widening.
  - Raises the full-exit target only for originally `20+`-contract trades from `1.2R` to `1.35R`.
  - 2025 output copied to `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_1mexitfix_sizecompress_widerisk_fullratio_beta030_target135_20plus.csv`.
- Added `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_flip_side_20plus.py` as a separate experiment from the corrected `1mexitfix` branch.
  - Keeps `0-19` contract trades unchanged.
  - For trades whose original raw size would have been `20+`, it flips the executed side at entry while keeping the same entry timing, same raw risk magnitude, and the same 1.2R full-exit framework.
  - 2025 output copied to `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_flip_side_20plus.csv`.
- Saved rollback snapshot `saved_versions/correct_baseline_full_exit_at_1p2R_pre_targetpriorityfix_2026-03-29.py` before changing exit precedence in the original corrected 1m-exitfix baseline.
  - Edited `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R.py` so within each 1m bar the hard target wick is checked before the close-based stop, matching the intended intrabar ordering.
  - This is an exit-only change; entries and sizing are unchanged.
  - Reverse by copying `saved_versions/correct_baseline_full_exit_at_1p2R_pre_targetpriorityfix_2026-03-29.py` back over `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R.py`.
  - 2025 rerun output copied to `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_1mexitfix_targetpriority.csv`.
- Added `experiments/hodlod_entry_features_retrained/reconstruct_backtest_flip_side_bucket.py` to encode the exact fixed-ledger reconstruction workflow requested by the user.
  - This is not a live path-regenerating rerun. It preserves the original backtest's timestamps, prices, sizes, and execution count, and reconstructs a new backtest ledger by flipping only the side / PnL / outcome for a chosen contract bucket.
  - Verified on the original corrected 2025 `1mexitfix` baseline for `100+` contracts:
    - input: `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_1mexitfix.csv`
    - output: `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_reconstructed_flip_side_100plus.csv`
    - assertions passed: trade count unchanged and bucket count unchanged.
- Added `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_executionflip_20plus.py`, preserving original engine path but flipping execution side for `20+` contract trades. Wrote `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_executionflip_20plus.csv`.
- Exported generic CSV for the `20+` execution-flip 2025 run: `outputs/trades_generic_export_2025_correct_baseline_full_exit_at_1p2R_executionflip_20plus.csv` (2090 rows, no commissions/fees).
- Ran the `20+` execution-flip backtest for 2024 and 2026. For 2026, stitched the newly fetched Jan-Feb 30s parquet `data/mnq_30s_2026-01-01_to_2026-02-28.parquet` with Jan-Feb 1m and March 1m/30s inputs.
- Exported generic CSVs for 2024 and 2026 `20+` execution-flip runs. Separate commission totals at `$0.25/contract/execution`: 2024 `$18,878.00`, 2026 `$2,590.50`.
- Added `bnr_live_engine_pwin13_hodlod_merged_executionflip_20plus.py` and wired it into `/Users/radhikaarora/Documents/New Project/stream_test.py` as a separate paper engine (`output/bnr_live_pwin13_hodlod_merged_execflip20`).
- Renamed the live combined-trades engine tag for the new execflip20 engine to `bnr_pwin13hm_execflip20`.
- Aligned the `execflip20` live engine with the backtest exit mechanics (full exit, min 2.5 pts target, target-before-stop, no slippage) and removed `bnr_top2` from `stream_test.py`.
- Engine I now remains active through `12:01 PM ET` instead of `12:00 PM ET`.

- 2026-03-31: Fixed stale post-win state reset across all BNR live engines by calling `_reset_after_close()` after final target exits in `bnr_live_engine_pwin.py`, `bnr_live_engine_pwin13.py`, `bnr_live_engine_pwin13_retrained.py`, `bnr_live_engine_pwin13_hodlod_merged.py`, `bnr_live_engine_pwin13_hodlod_merged_executionflip_20plus.py`, and `bnr_live_engine_top2.py`. This prevents engines from getting stuck after winners and missing fresh retest setups.

- 2026-03-31: Patched `/Users/radhikaarora/Documents/New Project/stream_test.py` to append backtest-format daily market-data files to `output/live_market_data/` for both 1m and 30s bars. Logging resumes into the same same-day files after restart and skips duplicate timestamps already present. 1m startup backfill is also logged once per restart.

- 2026-03-31: Added `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_executionflip_20plus_min20only.py`, which blocks entries below 20 contracts and otherwise keeps the latest execution-flip-20plus logic unchanged. Ran 2025 and wrote `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_executionflip_20plus_min20only.csv`.
