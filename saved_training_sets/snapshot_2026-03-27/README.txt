Snapshot: 2026-03-27

Context:
- This snapshot captures the current ML training dataset + trained model used by the
  13-feature HOD/LOD retrained backtest (effectively 15 features including HOD/LOD).
- This run used the “no-gates” version (no retrace min/max gates, no prior 1m color gate)
  for the BNR deterministic engine with ML entry scoring.

Files:
- ml_features_13feat_retrain_hodlod_entry_retrainedset.csv
  Training feature matrix used to train p_win model (retrained set only).

- entry_model_pwin_13features_retrained_hodlod_entry_retrainedset.joblib
  Serialized trained model (joblib). This is the exact classifier used for inference
  when running the HOD/LOD retrained backtests.

Notes:
- The joblib is the trained ML model weights + preprocessing pipeline.
  It lets you run backtests without retraining and preserves exact historical behavior.
- Keep this snapshot to reproduce the results for the 2025–2026 runs discussed
  around March 27, 2026.
