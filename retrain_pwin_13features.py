#!/usr/bin/env python3
"""
retrain_pwin_from_backtest.py
-----------------------------
Reconstructs ML features for every unique trade in the backtest output CSV
(output_bnr_det_2025_allow_counter.csv) using the raw 30s and 1m bar data,
then retrains the pwin XGBoost model (P(win)) and saves to entry_model_pwin.joblib.

Label: outcome == 'win'  →  label_win = 1
       outcome == 'loss' →  label_win = 0
       (scale-out duplicate rows are deduplicated first)
"""

import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# ── Paths ───────────────────────────────────────────────────────────────────
BASE      = "/Users/radhikaarora/Documents/Trading ML/ML V2"
DATA_DIR  = "/Users/radhikaarora/Documents/New Project/output/market"

BACKTEST_CSV = os.path.join(BASE, "output_bnr_det_2025_allow_counter.csv")
BARS_30S     = os.path.join(DATA_DIR, "mnq_30s_full.csv")
BARS_1M      = os.path.join(DATA_DIR, "mnq_1m.csv")
OUT_MODEL    = os.path.join(BASE, "entry_model_pwin.joblib")
OUT_CSV      = os.path.join(BASE, "ml_features_from_backtest.csv")

ML_FEATURES = [
    "retrace", "pivot_flem_dist", "time_since_pivot_sec",
    "body_last", "body_sum", "body_mean",
    "in_dir_ratio", "max_in_dir_run", "bars_since_pivot",
    "zone_over_range", "pivot_over_range",
    "dist_to_extrema_atr", "zone_to_extrema_atr",
]

ET = "America/New_York"

# ── Load raw data ────────────────────────────────────────────────────────────
print("Loading 30s bars …")
bars30 = pd.read_csv(BARS_30S, parse_dates=['timestamp'])
bars30['timestamp'] = pd.to_datetime(bars30['timestamp'], utc=True).dt.tz_convert(ET)
bars30 = bars30.sort_values('timestamp').reset_index(drop=True)
print(f"  {len(bars30)} rows  {bars30['timestamp'].min()} → {bars30['timestamp'].max()}")

print("Loading 1m bars …")
bars1m = pd.read_csv(BARS_1M, parse_dates=['timestamp'])
bars1m['timestamp'] = pd.to_datetime(bars1m['timestamp'], utc=True).dt.tz_convert(ET)
bars1m = bars1m.sort_values('timestamp').reset_index(drop=True)

print("Loading backtest output …")
bt = pd.read_csv(BACKTEST_CSV)

# Parse timestamps
for col in ['entry_time','pivot_time','flem_saved_time']:
    if col in bt.columns:
        bt[col] = pd.to_datetime(bt[col], utc=True).dt.tz_convert(ET)
# 'day' is a plain date string like "2025-01-03" — keep as date
bt['day'] = pd.to_datetime(bt['day']).dt.date

# Deduplicate: keep first row per (day, direction, entry_time)
bt_u = bt.drop_duplicates(subset=['day','direction','entry_time']).copy()
print(f"  {len(bt)} raw rows → {len(bt_u)} unique trades")
print(f"  Win/loss: {bt_u['outcome'].value_counts().to_dict()}")

# ── Build feature matrix ─────────────────────────────────────────────────────
print("\nReconstructing features …")

rows = []
skipped = 0

for _, tr in bt_u.iterrows():
    day         = tr['day']
    direction   = tr['direction']
    entry_time  = tr['entry_time']
    pivot_time  = tr['pivot_time']
    pivot_price = tr['pivot']
    flem_price  = tr['flem']
    retrace_val = tr['retrace_at_entry']

    # ── 30s bars from pivot_time to entry_time (inclusive) ──────────────────
    mask30 = (
        (bars30['timestamp'] >= pivot_time) &
        (bars30['timestamp'] <= entry_time) &
        (bars30['timestamp'].dt.date == day)
    )
    w30 = bars30[mask30].copy()

    if len(w30) == 0:
        skipped += 1
        continue

    # Candle bodies (absolute size)
    w30['body'] = (w30['close'] - w30['open']).abs()

    # In-direction count (close > open = bullish = in-dir for long, vice versa)
    if direction == 'long':
        w30['in_dir'] = (w30['close'] > w30['open']).astype(int)
    else:
        w30['in_dir'] = (w30['close'] < w30['open']).astype(int)

    body_sum   = float(w30['body'].sum())
    body_mean  = float(w30['body'].mean())
    body_last  = float(w30['body'].iloc[-1])
    bars_since = len(w30)
    in_dir_ratio = float(w30['in_dir'].mean())

    # max_in_dir_run: longest consecutive in-direction streak
    run = max_run = 0
    for v in w30['in_dir']:
        if v == 1:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    time_since_pivot = (entry_time - pivot_time).total_seconds()
    pivot_flem_dist  = abs(flem_price - pivot_price)

    # ── zone_over_range / pivot_over_range from 1m bars ─────────────────────
    day_date = day  # already a date object
    mask1m_day = (
        (bars1m['timestamp'].dt.date == day_date) &
        (bars1m['timestamp'] >= pd.Timestamp(f"{day_date} 09:30:00", tz=ET)) &
        (bars1m['timestamp'] <= entry_time)
    )
    w1m = bars1m[mask1m_day]

    if len(w1m) > 0:
        day_high = w1m['high'].max()
        day_low  = w1m['low'].min()
        day_range = day_high - day_low
        # ATR at entry: compute true range on day's 1m bars up to entry_time
        h = w1m['high']; l = w1m['low']; c = w1m['close']
        pc = c.shift(1)
        tr_vals = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr_at_entry = float(tr_vals.rolling(14, min_periods=1).mean().iloc[-1])
    else:
        day_range = 1.0  # fallback
        atr_at_entry = 1.0

    # Zone bar = first 1m bar at 09:30
    zone_bar = bars1m[
        (bars1m['timestamp'].dt.date == day_date) &
        (bars1m['timestamp'].dt.hour == 9) &
        (bars1m['timestamp'].dt.minute == 30)
    ]
    if len(zone_bar) > 0:
        zone_h = zone_bar.iloc[0]['high']
        zone_l = zone_bar.iloc[0]['low']
        zone_range = zone_h - zone_l
    else:
        zone_range = pivot_flem_dist  # rough fallback
        zone_h = zone_l = None

    zone_over_range  = zone_range  / day_range if day_range > 0 else 0.0
    pivot_over_range = pivot_flem_dist / day_range if day_range > 0 else 0.0

    # HOD/LOD features normalized to ATR
    entry_price_est = float(w30['close'].iloc[-1]) if len(w30) > 0 else pivot_price
    if direction == 'long':
        dist_to_extrema_atr = (day_high - entry_price_est) / atr_at_entry if atr_at_entry > 0 else 0.0
        zone_to_extrema_atr = (day_high - zone_h) / atr_at_entry if (zone_h is not None and atr_at_entry > 0) else 0.0
    else:
        dist_to_extrema_atr = (entry_price_est - day_low) / atr_at_entry if atr_at_entry > 0 else 0.0
        zone_to_extrema_atr = (zone_l - day_low) / atr_at_entry if (zone_l is not None and atr_at_entry > 0) else 0.0

    label_win = 1 if str(tr['outcome']).lower() == 'win' else 0

    rows.append({
        'open_date':       str(day),
        'direction':       direction,
        'entry_time':      str(entry_time),
        'pivot_time':      str(pivot_time),
        'outcome':         tr['outcome'],
        'label_win':       label_win,
        # features
        'retrace':              retrace_val,
        'pivot_flem_dist':      pivot_flem_dist,
        'time_since_pivot_sec': time_since_pivot,
        'body_last':            body_last,
        'body_sum':             body_sum,
        'body_mean':            body_mean,
        'in_dir_ratio':         in_dir_ratio,
        'max_in_dir_run':       max_run,
        'bars_since_pivot':     bars_since,
        'zone_over_range':      zone_over_range,
        'pivot_over_range':     pivot_over_range,
        'dist_to_extrema_atr':  dist_to_extrema_atr,
        'zone_to_extrema_atr':  zone_to_extrema_atr,
    })

feat_df = pd.DataFrame(rows)
print(f"  Reconstructed {len(feat_df)} trades  |  skipped {skipped}")
print(f"  label_win: {feat_df['label_win'].value_counts().to_dict()}")
print(f"  Win rate: {feat_df['label_win'].mean():.3f}")
feat_df.to_csv(OUT_CSV, index=False)
print(f"  Saved feature matrix → {OUT_CSV}")

# ── Train ────────────────────────────────────────────────────────────────────
print("\nTraining pwin model …")
X = feat_df[ML_FEATURES].fillna(0.0).values
y = feat_df['label_win'].astype(int).values

n_pos = y.sum()
n_neg = (y == 0).sum()
spw   = n_neg / n_pos
print(f"  scale_pos_weight = {spw:.2f}  ({n_pos} wins, {n_neg} losses)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

clf = XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=spw,
    eval_metric="auc",
    max_depth=3,
    n_estimators=100,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.5,
    reg_lambda=2.0,
    verbosity=0,
    random_state=42,
)

cv_auc = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
print(f"  CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# Calibrate for better probability estimates
print("  Calibrating …")
cal = CalibratedClassifierCV(clf, method='isotonic', cv=5)
cal.fit(X, y)

# In-sample score distribution
probs = cal.predict_proba(X)[:, 1]
print(f"\n  Probability distribution (all {len(probs)} trades):")
for p in [10, 25, 50, 75, 90, 95]:
    print(f"    p{p:2d}: {np.percentile(probs, p):.4f}")

win_p  = probs[y == 1]
loss_p = probs[y == 0]
print(f"\n  Wins  : mean={win_p.mean():.4f}  median={np.median(win_p):.4f}")
print(f"  Losses: mean={loss_p.mean():.4f}  median={np.median(loss_p):.4f}")

# Threshold sweep
print("\n── Threshold sweep ──")
print(f"  {'thresh':>7}  {'trades':>7}  {'WR':>6}  {'coverage':>9}")
thresholds = np.round(np.arange(0.35, 0.75, 0.05), 2)
feat_df['prob'] = probs
total = len(feat_df)
for t in thresholds:
    sub = feat_df[feat_df['prob'] >= t]
    n   = len(sub)
    wr  = sub['label_win'].mean() if n > 0 else 0
    cov = n / total
    print(f"  {t:>7.2f}  {n:>7d}  {wr:>6.1%}  {cov:>9.1%}")

# ── Feature importances ─────────────────────────────────────────────────────
try:
    base_clf = cal.calibrated_classifiers_[0].estimator
    imp = base_clf.feature_importances_
    feat_imp = sorted(zip(ML_FEATURES, imp), key=lambda x: -x[1])
    print("\nFeature importances:")
    for feat, score in feat_imp:
        print(f"  {feat:<25} {score:.4f}")
except Exception as e:
    print(f"  (could not extract importances: {e})")

# ── Save ─────────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUT_MODEL} …")
joblib.dump(cal, OUT_MODEL)
print("Done.")
print(f"\nNext step: re-run backtest and compare vs top-2 baseline (278 trades, 50.7% WR)")
