#!/usr/bin/env python3
"""
train_models.py
---------------
Trains two XGBoost models from ml_features_combined.csv:

  p_valid  →  entry_model_pvalid.joblib   (label = Setup Valid yes/no)
  p_win    →  entry_model_pwin.joblib      (label = Net P&L > 0)

Uses GridSearchCV to tune regularization params that help with small datasets.
Reports 5-fold stratified CV accuracy + ROC-AUC for each model.
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────

FEATURES_CSV = "/Users/radhikaarora/Documents/Trading ML/ML V2/ml_features_combined.csv"
MODEL_PVALID = "/Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pvalid.joblib"
MODEL_PWIN   = "/Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pwin.joblib"
MODEL_ENTRY  = "/Users/radhikaarora/Documents/Trading ML/ML V2/entry_model.joblib"  # used by engine

ML_FEATURES = [
    "retrace", "pivot_flem_dist", "time_since_pivot_sec",
    "body_last", "body_sum", "body_mean",
    "in_dir_ratio", "max_in_dir_run", "bars_since_pivot",
    "zone_over_range", "pivot_over_range",
]

# Hyperparameter grid — focused on regularization for small datasets
PARAM_GRID = {
    "max_depth":        [2, 3, 4],
    "n_estimators":     [50, 100, 200],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.6, 0.8],
    "colsample_bytree": [0.6, 0.8],
    "reg_alpha":        [0, 0.5, 1.0],      # L1
    "reg_lambda":       [1.0, 5.0, 10.0],   # L2
    "min_child_weight": [3, 5, 10],          # min samples per leaf
    "gamma":            [0, 0.1, 0.5],       # min loss reduction to split
}

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading feature matrix …")
df = pd.read_csv(FEATURES_CSV)
print(f"  {len(df)} rows, {df.shape[1]} columns")

before = len(df)
df = df.dropna(subset=ML_FEATURES)
if len(df) != before:
    print(f"  Dropped {before - len(df)} rows with NaN features  ({len(df)} remain)")

X = df[ML_FEATURES].values


# ── Train helper ──────────────────────────────────────────────────────────────

def train_and_evaluate(X, y, label_name: str, model_path: str):
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    print(f"\n{'='*60}")
    print(f"  Model: {label_name}")
    print(f"  Samples: {len(y)}  |  0={n_neg}  1={n_pos}  "
          f"scale_pos_weight={scale_pos_weight:.2f}")
    print(f"{'='*60}")

    base_clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,   # handles class imbalance
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("  Running GridSearchCV (this may take a moment) …")
    grid = GridSearchCV(
        base_clf,
        PARAM_GRID,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,          # refit best estimator on full data
        verbose=0,
    )
    grid.fit(X, y)

    best = grid.best_estimator_
    print(f"\n  Best params:")
    for k, v in grid.best_params_.items():
        print(f"    {k:20s} = {v}")
    print(f"\n  Best CV ROC-AUC: {grid.best_score_:.4f}")

    # Full CV report on best model (accuracy + AUC)
    cv_full = cross_validate(
        best, X, y, cv=cv,
        scoring=["accuracy", "roc_auc"],
        return_train_score=True,
    )
    print(f"\n  5-Fold CV (best model):")
    print(f"    Accuracy  train={cv_full['train_accuracy'].mean():.3f}±{cv_full['train_accuracy'].std():.3f}"
          f"  |  val={cv_full['test_accuracy'].mean():.3f}±{cv_full['test_accuracy'].std():.3f}")
    print(f"    ROC-AUC   train={cv_full['train_roc_auc'].mean():.3f}±{cv_full['train_roc_auc'].std():.3f}"
          f"  |  val={cv_full['test_roc_auc'].mean():.3f}±{cv_full['test_roc_auc'].std():.3f}")

    # Full-data report
    y_pred = best.predict(X)
    print(f"\n  Full-data classification report:")
    print(classification_report(y, y_pred, target_names=["0 (no)", "1 (yes)"]))

    # Feature importances
    importances = pd.Series(best.feature_importances_, index=ML_FEATURES)
    importances = importances.sort_values(ascending=False)
    print(f"  Feature importances (gain):")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 40)
        print(f"    {feat:25s}  {imp:.4f}  {bar}")

    joblib.dump(best, model_path)
    print(f"\n  ✓ Saved → {model_path.split('/')[-1]}")
    return best


# ── Train p_valid ─────────────────────────────────────────────────────────────

y_valid = df["label_valid"].astype(int).values
clf_valid = train_and_evaluate(X, y_valid, "p_valid (Setup Valid)", MODEL_PVALID)

# ── Train p_win ───────────────────────────────────────────────────────────────

y_win = df["label_win"].astype(int).values
clf_win = train_and_evaluate(X, y_win, "p_win (Net P&L > 0)", MODEL_PWIN)

# Save in engine-compatible format: {'model': ..., 'features': [...]}
joblib.dump({'model': clf_win, 'features': ML_FEATURES}, MODEL_ENTRY)
print(f"  ✓ Saved → entry_model.joblib  (engine-ready, p_win)")


# ── Sanity check: score Jan 3 trade ──────────────────────────────────────────

jan3 = df[df["open_date"] == "2025-01-03"]
if not jan3.empty:
    x_jan3 = jan3[ML_FEATURES].values
    print()
    for i, (_, row) in enumerate(jan3.iterrows()):
        pv = clf_valid.predict_proba(x_jan3[[i]])[0][1]
        pw = clf_win.predict_proba(x_jan3[[i]])[0][1]
        print(f"Sanity Jan 3 ({row['direction']:5s}):  "
              f"p_valid={pv:.3f}  p_win={pw:.3f}  "
              f"label_valid={int(row['label_valid'])}  label_win={int(row['label_win'])}")

print("\n✓ Done.")
