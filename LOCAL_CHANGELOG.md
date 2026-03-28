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
