#!/usr/bin/env python3
"""
retrain_timing_model.py
-----------------------
Trains a timing model on ml_entry_dataset_30s_retrace_from_labels.csv:
  - One row per 30s bar during the retrace window
  - label=1: this bar is the correct entry bar (trade won at this timing)
  - label=0: pre-entry bars or windows with no winning entry

Handles heavy class imbalance (~7% positive rate) via scale_pos_weight.
Saves to entry_model.joblib (dict with 'model' and 'features') so the
engine picks it up automatically.

Also outputs an updated threshold range suitable for this model.
"""

import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

BASE = "/Users/radhikaarora/Documents/Trading ML/ML V2"
DATASET = os.path.join(BASE, "ml_entry_dataset_30s_retrace_from_labels.csv")
OUT_MODEL = os.path.join(BASE, "entry_model.joblib")  # engine reads this

ML_FEATURES = [
    "retrace", "pivot_flem_dist", "time_since_pivot_sec",
    "body_last", "body_sum", "body_mean",
    "in_dir_ratio", "max_in_dir_run", "bars_since_pivot",
    "zone_over_range", "pivot_over_range",
]

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATASET)
print(f"  {len(df)} rows  |  label dist: {df['label'].value_counts().to_dict()}")
print(f"  Positive rate: {df['label'].mean():.3f}")

X = df[ML_FEATURES].fillna(0.0).values
y = df['label'].astype(int).values

n_pos = y.sum()
n_neg = (y == 0).sum()
spw   = n_neg / n_pos  # scale_pos_weight for XGBoost
print(f"  scale_pos_weight = {spw:.1f}")

# ── Grid search ────────────────────────────────────────────────────────────────
print("\nRunning GridSearchCV (this may take ~2 min) …")
param_grid = {
    "max_depth":        [2, 3, 4],
    "n_estimators":     [50, 100, 200],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "reg_alpha":        [0, 0.5],
    "reg_lambda":       [1.0, 5.0],
    "min_child_weight": [5, 10],
    "gamma":            [0, 0.2],
}

base_clf = XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=spw,
    eval_metric="auc",
    verbosity=0,
    random_state=42,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    base_clf, param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=0,
)
grid.fit(X, y)

best = grid.best_estimator_
print(f"  Best params: {grid.best_params_}")
print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")

# ── Calibrate probabilities ────────────────────────────────────────────────────
# Calibration helps the threshold be more meaningful
print("\nCalibrating probabilities (isotonic) …")
calibrated = CalibratedClassifierCV(best, method='isotonic', cv=5)
calibrated.fit(X, y)

# Evaluate calibrated model
auc_scores = cross_val_score(calibrated, X, y, cv=cv, scoring='roc_auc')
print(f"  Calibrated CV ROC-AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

# Score all rows for threshold analysis
probs = calibrated.predict_proba(X)[:, 1]
print(f"\n  Probability distribution:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"    p{p:2d}: {np.percentile(probs, p):.4f}")

# Win bars vs non-win bars
win_probs  = probs[y == 1]
loss_probs = probs[y == 0]
print(f"\n  label=1 bars: mean={win_probs.mean():.4f}  median={np.median(win_probs):.4f}  "
      f"min={win_probs.min():.4f}  max={win_probs.max():.4f}")
print(f"  label=0 bars: mean={loss_probs.mean():.4f}  median={np.median(loss_probs):.4f}  "
      f"min={loss_probs.min():.4f}  max={loss_probs.max():.4f}")

# ── Threshold analysis on this model ──────────────────────────────────────────
print("\n── Threshold sweep (simulate first qualifying bar per window) ──")
df['prob'] = probs

# Group by window (open_date, direction)
if 'current_time' in df.columns:
    df = df.sort_values(['open_date', 'direction', 'current_time'])

windows = list(df.groupby(['open_date', 'direction']))
total_windows = len(windows)

thresholds = np.round(np.arange(0.05, 0.50, 0.05), 2)
print(f"  {'thresh':>7}  {'trades':>7}  {'WR':>6}  {'coverage':>9}")
print(f"  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*9}")

best_thresh = 0.10
best_composite = 0.0

for thresh in thresholds:
    wins = losses = 0
    for (od, d), grp in windows:
        q = grp[grp['prob'] >= thresh]
        if q.empty:
            continue
        first = q.iloc[0]
        if first['label'] == 1:
            wins += 1
        else:
            losses += 1
    n = wins + losses
    wr = wins / n if n > 0 else 0.0
    cov = n / total_windows
    comp = wr * (n ** 0.5)
    print(f"  {thresh:>7.2f}  {n:>7d}  {wr:>6.1%}  {cov:>9.1%}")
    if comp > best_composite:
        best_composite = comp
        best_thresh = thresh

print(f"\n  Suggested ENTRY_MODEL_THRESH = {best_thresh:.2f}")

# ── Save ───────────────────────────────────────────────────────────────────────
print(f"\nSaving model to {OUT_MODEL} …")
bundle = {
    'model':    calibrated,
    'features': ML_FEATURES,
    'model_type': 'timing',  # trained on 30s bars, not entry-level
    'threshold': best_thresh,
}
joblib.dump(bundle, OUT_MODEL)
print("Done.")

# Print feature importances from the underlying XGBoost model
try:
    imp = best.feature_importances_
    feat_imp = sorted(zip(ML_FEATURES, imp), key=lambda x: -x[1])
    print("\nFeature importances (gain):")
    for feat, score in feat_imp:
        print(f"  {feat:<25} {score:.4f}")
except Exception:
    pass
