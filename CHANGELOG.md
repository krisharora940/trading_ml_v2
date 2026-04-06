# Changelog

All notable changes made by Codex are recorded here.

## 2026-03-27
- Reverted entry execution to **next 30s candle open** after a signal (no same‑bar fills).
- Added pending-entry state to avoid lookahead and preserve signal-time fields (pivot/flem/retrace/strength/pwin).

### Notes
- Market data root: `/Users/radhikaarora/Documents/New Project/output/market/quarterly` (use this folder for all 1m/30s quarterly data).

## 2026-03-27 (cont.)
- Reran 2026 Jan–Feb with next‑open execution (no lookahead).
- Exported generic CSV for 2026 Jan–Feb only.
  - Output: `/Users/radhikaarora/Documents/Trading ML/ML V2/trades_generic_export_2026_janfeb_pwin40_cap250_slip1tick_tickvalid_cap100scale_openfill.csv`
- Reran 2026 March with correct 1m/30s sources and next‑open execution.
- Exported generic CSV for 2026 March only.
  - Output: `/Users/radhikaarora/Documents/Trading ML/ML V2/trades_generic_export_2026_mar_pwin40_cap250_slip1tick_tickvalid_cap100scale_openfill.csv`
- Restored baseline execution semantics (no pending next-open fills; entry at 30s close). Removed slippage/tick-valid exit skips and reverted scale-out plan to >=13 vs <13.
- Reran 2025 with baseline execution (output_bnr_det_2025_13feat_pwin40_retrained_hodlod_entry_nogates_baseline_exec.csv).
- Reverted `experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained.py` to commit `4d779f0`.

### Revert Checklist
- To restore engine to latest commit baseline:
  - `git checkout 4d779f0 -- experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained.py`
- To verify current engine diff vs commit:
  - `git diff 4d779f0 -- experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained.py`
- Reran 2025/2026 with engine reverted to commit 4d779f0. Outputs:
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_13feat_pwin40_retrained_hodlod_entry_nogates_commit4d779f0.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2026_janfeb_13feat_pwin40_retrained_hodlod_entry_nogates_commit4d779f0.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2026_mar_13feat_pwin40_retrained_hodlod_entry_nogates_commit4d779f0.csv

### Notes (continued)
- Saved baseline engine copy: `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/commit4d779f0_baseline/bnr_deterministic_engine_13feat_hodlod_retrained.py`
- For clean OOS runs on the HOD/LOD retrained engine, use the exclusionary monthly wrapper:
  `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/run_oos_monthly_hodlod_retrained.py`
- Default assumption going forward: if the request is for proper OOS / no same-month training leakage, run through that wrapper instead of the raw engine file.
- Added multi-trade support with 5-minute entry cooldown (unlimited concurrent trades, any direction). Implemented per-trade exit management via open_trades. Saved baseline copy remains unchanged.
- Reran 2025/2026 with multi-trade 5m cooldown outputs:
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_13feat_pwin40_retrained_hodlod_entry_nogates_multi5m.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2026_janfeb_13feat_pwin40_retrained_hodlod_entry_nogates_multi5m.csv
  - /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2026_mar_13feat_pwin40_retrained_hodlod_entry_nogates_multi5m.csv
- Undo: reverted engine back to commit 4d779f0 (removed multi-trade 5m cooldown changes).

## 2026-03-28
- Re-implemented multi-trade support with 5-minute entry cooldown (unlimited concurrent trades, any direction) in:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained.py`
- Entry logic now uses per-trade `open_trades` list and per-trade scale-out state.
- Direction flips and end-of-day cleanup close **all** open trades.
- Saved baseline engine copy remains unchanged:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/commit4d779f0_baseline/bnr_deterministic_engine_13feat_hodlod_retrained.py`
- Cooldown now applies **only while at least one trade is open**; cleared when all trades close.
- Added anti-spam overlap guard: overlapping entries now require a **new pivot timestamp** since the last entry (prevents auto re-entry every 5 minutes on the same unchanged setup).
- Undo: reverted the entire 2026-03-28 additional-entry / overlapping-entry branch by restoring the engine from the saved baseline copy above.
- Built a static live-paper merged HOD/LOD joblib for stream use:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pwin_13features_retrained_hodlod_entry_merged_live.joblib`
  - Feature matrix snapshot:
    `/Users/radhikaarora/Documents/Trading ML/ML V2/ml_features_13feat_retrain_hodlod_entry_merged_live.csv`
- Added new live engine:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/bnr_live_engine_pwin13_hodlod_merged.py`
  - Config: merged HOD/LOD model, `pwin >= 0.40`, no slippage, `MAX_RISK_DOLLARS=500`, `MAX_CONTRACTS=250`.
- Integrated the new engine into stream runner:
  - `/Users/radhikaarora/Documents/New Project/stream_test.py`
  - Stream label: `Engine H  [BNR HOD/LOD merged, thresh=0.40]`
- Notes:
  - For live stream / paper model use the static merged live joblib above, not the month-excluded OOS wrapper models.
  - Quarterly market data root remains:
    `/Users/radhikaarora/Documents/New Project/output/market/quarterly`
- Created a copy-only engine variant with size-based pwin gating:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained_sizegate20.py`
- New rule in that copy only:
  - if computed `contracts >= 20`, require `pwin_score >= 0.50`
  - otherwise keep the base engine threshold behavior unchanged
- Baseline engine remains untouched:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained.py`
- Updated the copy-only size-gated engine:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained_sizegate20.py`
- Size-based pwin rules now are:
  - `contracts >= 20` => require `pwin_score >= 0.50`
  - `contracts > 35` => require `pwin_score >= 0.60`
- Rebased the working baseline onto the merged-training-set OOS backtest engine.
- Saved pre-size-gates snapshot of the merged baseline engine:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/merged_oos_baseline_pre_sizegates_2026-03-28/bnr_deterministic_engine_13feat_hodlod_retrained.py`
- Updated the main merged-baseline engine in place:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features_retrained/bnr_deterministic_engine_13feat_hodlod_retrained.py`
- New size-based pwin rules on the merged baseline engine:
  - `contracts >= 20` => require `pwin_score >= 0.50`
  - `contracts > 35` => require `pwin_score >= 0.60`
- Notes:
  - This now affects future merged OOS wrapper runs that use the main HOD/LOD retrained engine.
  - The old pre-edit merged baseline is preserved at the saved snapshot path above for easy rollback.
- Replaced hard size-based skipping on the active merged baseline engine with phantom-trade handling.
- Saved pre-phantom snapshot:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/merged_oos_baseline_pre_phantom_sizegates_2026-03-28.py`
- Active baseline engine now preserves pathing and zeroes PnL for size-gated trades instead of skipping them.
- Ran direct 2026 no-wrapper forward test using fixed merged model:
  - model: `/Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pwin_13features_retrained_hodlod_entry_merged_live.joblib`
  - outputs:
    - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_merged_phantom_2026/output_bnr_det_2026_01_13feat_pwin40_hodlod_merged_phantom_2026.csv`
    - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_merged_phantom_2026/output_bnr_det_2026_02_13feat_pwin40_hodlod_merged_phantom_2026.csv`
    - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_merged_phantom_2026/output_bnr_det_2026_03_13feat_pwin40_hodlod_merged_phantom_2026.csv`
- Root-cause note:
  - leg-level `contracts` in the output do not always equal original entry size, so validating size gates by summing scale-out legs can understate the true entry contracts.

- Found and fixed a scale-out carry-forward bookkeeping bug in the merged baseline engines.
- Saved pre-fix snapshots:
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/merged_oos_baseline_pre_scaleout_fix_2026-03-28.py`
  - `/Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/merged_oos_baseline_pre_phantom_sizegates_scaleoutfix_2026-03-28.py`
- Fix details:
  - after an intermediate scale-out, the active remaining trade now carries the full remaining quantity from the rest of the scale-out plan, not just the next leg quantity.
  - this also updates `risk_dollars` on the carried trade to match the full remaining quantity.
- New direct 2026 no-wrapper reruns after the scale-out fix:
  - non-phantom comparator:
    - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_nophantom_2026_scaleoutfix/output_bnr_det_2026-01_nophantom_scaleoutfix.csv`
    - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_nophantom_2026_scaleoutfix/output_bnr_det_2026-02_nophantom_scaleoutfix.csv`
    - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_nophantom_2026_scaleoutfix/output_bnr_det_2026-03_nophantom_scaleoutfix.csv`
  - phantom engine:
    - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_phantom_2026_scaleoutfix/output_bnr_det_2026-01_phantom_scaleoutfix.csv`
    - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_phantom_2026_scaleoutfix/output_bnr_det_2026-02_phantom_scaleoutfix.csv`
    - `/Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_compare_phantom_2026_scaleoutfix/output_bnr_det_2026-03_phantom_scaleoutfix.csv`
- Key debugging note:
  - the earlier February phantom discrepancy centered on `2026-02-10 10:59 short`.
  - after the scale-out fix, the phantom engine preserves that original 10:59 trade with full remaining size, while the hard-gated comparator skips it and later takes a different 11:00 short.
  - this means the clean comparison is phantom vs original baseline, not phantom vs hard-skip comparator.

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

- 2026-03-28: Patched /Users/radhikaarora/Documents/Trading ML/ML V2/bnr_deterministic_engine_13feat_baseline_reset.py with the same scale-out carry-forward fix (carry full remaining quantity after partial exits, not just next leg). Saved pre-fix snapshot at /Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/bnr_deterministic_engine_13feat_baseline_reset_pre_scaleoutfix_2026-03-28.py. Ran corrected 2025 non-retrained reset-enabled 13-feature model and wrote /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_13feat_pwin50_baseline_reset_scaleoutfix.csv.

- 2026-03-28: Built a non-retrained 15-feature HOD/LOD baseline model from /Users/radhikaarora/Documents/Trading ML/ML V2/ml_features_13features.csv by augmenting HOD/LOD features from 2025 quarterly 1m data. Saved features to /Users/radhikaarora/Documents/Trading ML/ML V2/ml_features_13features_hodlod_baseline.csv and model to /Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pwin_13features_hodlod_baseline.joblib. Patched /Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features/bnr_deterministic_engine_13feat_hodlod.py to use that baseline model and fixed scale-out carry-forward. Saved pre-fix snapshot at /Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/bnr_deterministic_engine_13feat_hodlod_pre_baseline_hodlod_scaleoutfix_2026-03-28.py. Ran corrected 2025 baseline HOD/LOD engine and wrote /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_15feat_baseline_hodlod_scaleoutfix.csv.

- 2026-03-28: From /Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features/bnr_deterministic_engine_13feat_hodlod.py removed deterministic entry gates (min retrace, max retrace, prior 1m candle color) while keeping PWIN threshold at 0.50. Saved pre-change snapshot at /Users/radhikaarora/Documents/Trading ML/ML V2/saved_versions/bnr_deterministic_engine_13feat_hodlod_pre_nogates_pwin40_2026-03-28.py. Ran corrected 2025 baseline HOD/LOD no-gates variant and wrote /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_15feat_baseline_hodlod_scaleoutfix_nogates_pwin50.csv.

- 2026-03-28: Added month-excluded OOS tooling for the corrected 15-feature baseline no-gates model. Engine now honors PWIN_MODEL_PATH and accepts pwin_thresh. New scripts: /Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features/retrain_pwin_15features_hodlod_baseline_nogates.py and /Users/radhikaarora/Documents/Trading ML/ML V2/experiments/hodlod_entry_features/run_oos_monthly_hodlod_baseline_nogates.py. Ran 2025 exclusionary sweep at PWIN=0.50; outputs in /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/oos_monthly_hodlod_baseline_nogates and models in /Users/radhikaarora/Documents/Trading ML/ML V2/saved_training_sets/oos_models_hodlod_baseline_nogates.

- 2026-03-28: Combined the 2025 month-excluded OOS outputs for the corrected 15-feature no-gates baseline into /Users/radhikaarora/Documents/Trading ML/ML V2/outputs/output_bnr_det_2025_15feat_baseline_hodlod_scaleoutfix_nogates_pwin50_ooswrapper_combined.csv and retrained a fresh model from that corrected OOS-derived set. Saved feature matrix to /Users/radhikaarora/Documents/Trading ML/ML V2/ml_features_15feat_hodlod_baseline_nogates_oos2025_retrained.csv and model to /Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pwin_15features_hodlod_baseline_nogates_oos2025_retrained.joblib.

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
- Added `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_executionflip_20plus.py`, a backtest-layer execution-side flip variant that preserves original path/timestamps/prices while flipping booked side/PnL for `20+` contract trades only. Output: `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_executionflip_20plus.csv`
- Exported generic CSV for `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_executionflip_20plus.csv` to `outputs/trades_generic_export_2025_correct_baseline_full_exit_at_1p2R_executionflip_20plus.csv` with no commissions/fees.
- Ran `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_executionflip_20plus.py` on 2024 quarterly market data and on stitched 2026 data using `data/mnq_30s_2026-01-01_to_2026-02-28.parquet` + `/Users/radhikaarora/Documents/New Project/Input Data/market/mnq_1m_jan_feb_2026.csv` + March `1m/30s` files. Outputs: `outputs/output_bnr_det_2024_correct_baseline_full_exit_at_1p2R_executionflip_20plus.csv`, `outputs/output_bnr_det_2026_correct_baseline_full_exit_at_1p2R_executionflip_20plus.csv`.
- Exported generic CSVs for the `20+` execution-flip backtests: `outputs/trades_generic_export_2024_correct_baseline_full_exit_at_1p2R_executionflip_20plus.csv` and `outputs/trades_generic_export_2026_correct_baseline_full_exit_at_1p2R_executionflip_20plus.csv` with no commissions/fees. Calculated commissions separately at `$0.25` per contract per execution.
- Added live engine `bnr_live_engine_pwin13_hodlod_merged_executionflip_20plus.py` implementing execution-side flip for `20+` contract trades while preserving the original virtual direction/path logic. Patched `/Users/radhikaarora/Documents/New Project/stream_test.py` to instantiate it as a separate live stream engine (`Engine I`) and pass it through the stream loop.
- Updated the combined-trades label for the new live execution-flip engine to `bnr_pwin13hm_execflip20` so its rows are easier to isolate in merged live CSV outputs.
- Updated `bnr_live_engine_pwin13_hodlod_merged_executionflip_20plus.py` to match the backtest exit mechanics more closely: no scale-outs, fixed full exit at `max(1.2R, 2.5 pts)`, target priority before close-based stop, and no live slippage. Removed `bnr_top2` engine wiring from `/Users/radhikaarora/Documents/New Project/stream_test.py`.
- Changed Engine I (`bnr_live_engine_pwin13_hodlod_merged_executionflip_20plus.py`) session close/force-close from `12:00 PM ET` to `12:01 PM ET`.

- 2026-03-31: Fixed stale post-win state reset across all BNR live engines by calling `_reset_after_close()` after final target exits in `bnr_live_engine_pwin.py`, `bnr_live_engine_pwin13.py`, `bnr_live_engine_pwin13_retrained.py`, `bnr_live_engine_pwin13_hodlod_merged.py`, `bnr_live_engine_pwin13_hodlod_merged_executionflip_20plus.py`, and `bnr_live_engine_top2.py`. This prevents engines from getting stuck after winners and missing fresh retest setups.

- 2026-03-31: Patched `/Users/radhikaarora/Documents/New Project/stream_test.py` to append backtest-format daily market-data files to `output/live_market_data/` for both 1m and 30s bars. Logging resumes into the same same-day files after restart and skips duplicate timestamps already present. 1m startup backfill is also logged once per restart.

- 2026-03-31: Added `experiments/hodlod_entry_features_retrained/correct_baseline_full_exit_at_1p2R_executionflip_20plus_min20only.py`, which blocks entries below 20 contracts and otherwise keeps the latest execution-flip-20plus logic unchanged. Ran 2025 and wrote `outputs/output_bnr_det_2025_correct_baseline_full_exit_at_1p2R_executionflip_20plus_min20only.csv`.
