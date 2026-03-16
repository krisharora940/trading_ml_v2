#!/usr/bin/env python3
"""
Analyze month-by-month breakdown for different pwin thresholds.
Shows how many trades, win rate, and P&L at each threshold.
"""

import pandas as pd
import numpy as np

# Load backtest output
csv_path = "/Users/radhikaarora/Documents/Trading ML/ML V2/output_bnr_det_2025_allow_counter.csv"
df = pd.read_csv(csv_path)

# Deduplicate by (entry_time, direction) to get unique trades
df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
df_unique = df.drop_duplicates(subset=['entry_time', 'direction'], keep='first')

# Add month column
df_unique['month'] = df_unique['entry_time'].dt.strftime('%Y-%m')

# Get unique months in order
months = sorted(df_unique['month'].unique())

# Thresholds to test
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

print("\n" + "="*120)
print("MONTH-BY-MONTH BREAKDOWN BY PWIN THRESHOLD")
print("="*120)

# Create summary for each threshold
for thresh in thresholds:
    print(f"\n{'='*120}")
    print(f"PWIN_THRESH = {thresh}")
    print(f"{'='*120}")
    print(f"{'Month':<12} {'Count':<8} {'Wins':<8} {'Losses':<8} {'Win%':<8} {'Avg P&L':<12} {'Total P&L':<12} {'Trades': <30}")
    print("-"*120)

    all_count = 0
    all_wins = 0
    all_pnl = 0.0

    for month in months:
        month_trades = df_unique[df_unique['month'] == month]
        filtered = month_trades[month_trades['pwin_score'] >= thresh]

        if len(filtered) == 0:
            print(f"{month:<12} {'0':<8} {'0':<8} {'0':<8} {'-':<8} {'-':<12} {'-':<12}")
            continue

        count = len(filtered)
        wins = (filtered['outcome'] == 'win').sum()
        losses = (filtered['outcome'] == 'loss').sum()
        win_pct = 100 * wins / count if count > 0 else 0
        total_pnl = filtered['pnl'].sum()
        avg_pnl = total_pnl / count if count > 0 else 0

        all_count += count
        all_wins += wins
        all_pnl += total_pnl

        trade_list = ", ".join([f"{row['direction'][0].upper()}" for _, row in filtered.iterrows()])

        print(f"{month:<12} {count:<8} {wins:<8} {losses:<8} {win_pct:<7.1f}% {avg_pnl:<11.2f} {total_pnl:<11.2f}")

    # Summary row
    if all_count > 0:
        all_win_pct = 100 * all_wins / all_count
        all_avg_pnl = all_pnl / all_count
        print("-"*120)
        print(f"{'TOTAL':<12} {all_count:<8} {all_wins:<8} {all_count - all_wins:<8} {all_win_pct:<7.1f}% {all_avg_pnl:<11.2f} {all_pnl:<11.2f}")

print("\n" + "="*120)
print("THRESHOLD SUMMARY (sorted by WR * sqrt(count))")
print("="*120)
print(f"{'Threshold':<12} {'Count':<8} {'Win%':<8} {'Total P&L':<12} {'Score':<12} {'Vs 0.55':<12}")
print("-"*120)

results = []
baseline_count = 0
baseline_pnl = 0
baseline_wins = 0

for thresh in thresholds:
    filtered = df_unique[df_unique['pwin_score'] >= thresh]
    count = len(filtered)
    wins = (filtered['outcome'] == 'win').sum()
    total_pnl = filtered['pnl'].sum()
    win_pct = 100 * wins / count if count > 0 else 0

    # Score: win_rate * sqrt(count) * total_pnl_per_trade
    if count > 0:
        pnl_per_trade = total_pnl / count
        score = (win_pct / 100) * np.sqrt(count) * pnl_per_trade
    else:
        score = 0

    # Track 0.55 baseline
    if thresh == 0.55:
        baseline_count = count
        baseline_pnl = total_pnl
        baseline_wins = wins

    results.append((thresh, count, win_pct, total_pnl, score))

# Sort by score (descending)
results_sorted = sorted(results, key=lambda x: x[4], reverse=True)

for thresh, count, win_pct, total_pnl, score in results_sorted:
    pnl_change = total_pnl - baseline_pnl if baseline_count > 0 else 0
    change_str = f"{pnl_change:+.0f}" if baseline_count > 0 else "—"
    print(f"{thresh:<12} {count:<8} {win_pct:<7.1f}% {total_pnl:<11.2f} {score:<11.2f} {change_str:<12}")

print("\n" + "="*120)
