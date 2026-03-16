#!/usr/bin/env python3
"""
analyze_thresholds.py
---------------------
Sweeps pwin / pvalid threshold combinations on ml_features_combined.csv
(one row per trade, label_win and label_valid already computed).

Usage:
    python3 analyze_thresholds.py
"""

import os
import numpy as np
import pandas as pd
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = "/Users/radhikaarora/Documents/Trading ML/ML V2"
DATASET      = os.path.join(BASE, "ml_features_combined.csv")
PWIN_MODEL   = os.path.join(BASE, "entry_model_pwin.joblib")
PVALID_MODEL = os.path.join(BASE, "entry_model_pvalid.joblib")
ENTRY_MODEL  = os.path.join(BASE, "entry_model.joblib")   # dict with 'features'

ML_FEATURES = joblib.load(ENTRY_MODEL)['features']

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATASET)
print(f"  {len(df)} trades  |  label_win: {df['label_win'].value_counts().to_dict()}")
print(f"  label_valid: {df['label_valid'].value_counts().to_dict()}")

print("Loading models …")
ml_pwin   = joblib.load(PWIN_MODEL)    # raw XGBClassifier
ml_pvalid = joblib.load(PVALID_MODEL)  # raw XGBClassifier

# ── Score every trade ──────────────────────────────────────────────────────────
X = df[ML_FEATURES].fillna(0.0)
df['pwin_score']   = ml_pwin.predict_proba(X)[:, 1]
df['pvalid_score'] = ml_pvalid.predict_proba(X)[:, 1]

print(f"\n  pwin  : min={df['pwin_score'].min():.3f}  max={df['pwin_score'].max():.3f}  "
      f"mean={df['pwin_score'].mean():.3f}  median={df['pwin_score'].median():.3f}")
print(f"  pvalid: min={df['pvalid_score'].min():.3f}  max={df['pvalid_score'].max():.3f}  "
      f"mean={df['pvalid_score'].mean():.3f}  median={df['pvalid_score'].median():.3f}")

total_trades = len(df)
baseline_wr  = df['label_win'].mean()
print(f"\nBaseline (no filter):  {total_trades} trades,  WR={baseline_wr:.1%}\n")

# ── Sweep thresholds ───────────────────────────────────────────────────────────
thresholds = np.round(np.arange(0.40, 0.80, 0.05), 2)
results = []

for pw in thresholds:
    for pv in thresholds:
        subset = df[(df['pwin_score'] >= pw) & (df['pvalid_score'] >= pv)]
        n      = len(subset)
        wins   = int(subset['label_win'].sum())
        wr     = wins / n if n > 0 else 0.0
        cov    = n / total_trades
        comp   = wr * (n ** 0.5)   # composite: WR × sqrt(trades)
        results.append({
            'pwin':      pw,
            'pvalid':    pv,
            'trades':    n,
            'wins':      wins,
            'wr':        round(wr, 4),
            'coverage':  round(cov, 4),
            'composite': round(comp, 4),
        })

res = pd.DataFrame(results).sort_values('composite', ascending=False)

# ── Win-rate grid ──────────────────────────────────────────────────────────────
print("=" * 72)
print("Win Rate Grid   format: WR%(N trades)")
print("pwin↓  pvalid→  ", end="")
pvs = sorted(res['pvalid'].unique())
print("  ".join(f"{pv:.2f}" for pv in pvs))
print("-" * 72)
for pw in sorted(res['pwin'].unique()):
    print(f"  {pw:.2f}          ", end="")
    for pv in pvs:
        row = res[(res['pwin'] == pw) & (res['pvalid'] == pv)].iloc[0]
        print(f"{row['wr']:.0%}({int(row['trades']):3d})  ", end="")
    print()

# ── Top-15 combos ──────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("Top-15  [ranked by WR × √trades]")
print(f"{'='*72}")
print(f"  {'pwin':>5}  {'pvalid':>6}  {'trades':>7}  {'WR':>6}  {'coverage':>9}  {'composite':>10}")
print(f"  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*9}  {'-'*10}")
for _, row in res.head(15).iterrows():
    marker = " ←" if row['wr'] > baseline_wr + 0.05 and row['trades'] >= 50 else ""
    print(f"  {row['pwin']:>5.2f}  {row['pvalid']:>6.2f}  {int(row['trades']):>7d}  "
          f"{row['wr']:>6.1%}  {row['coverage']:>9.1%}  {row['composite']:>10.4f}{marker}")

# ── Suggestion ────────────────────────────────────────────────────────────────
# Filter: must have at least 40% coverage and improve WR by >3pp
candidates = res[(res['coverage'] >= 0.40) & (res['wr'] > baseline_wr + 0.03)]
if candidates.empty:
    candidates = res  # fall back to top overall
best = candidates.iloc[0]
print(f"\nSuggested thresholds (>40% coverage, best WR improvement):")
print(f"  PWIN_THRESH   = {best['pwin']:.2f}")
print(f"  PVALID_THRESH = {best['pvalid']:.2f}")
print(f"  → {best['trades']} trades  WR={best['wr']:.1%}  coverage={best['coverage']:.1%}\n")
