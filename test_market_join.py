#!/usr/bin/env python3
"""
Test joining normalized trades data with market data to verify timestamp alignment.
"""

import pandas as pd
from datetime import datetime

def main():
    # Load normalized trades
    trades_file = "/Users/radhikaarora/Documents/Trading ML/ML V2/trades_20260315014828_with_flem_pivot.csv"
    trades_df = pd.read_csv(trades_file)
    trades_df['entry_ts'] = pd.to_datetime(trades_df['entry_ts'], utc=True)
    # Convert from UTC to naive for comparison (trades are in UTC, market is EDT)
    trades_df['entry_ts_edt'] = trades_df['entry_ts'].dt.tz_convert('America/New_York').dt.tz_localize(None)
    print(f"Loaded {len(trades_df)} trades")
    print(f"Trade date range (EDT): {trades_df['entry_ts_edt'].min()} to {trades_df['entry_ts_edt'].max()}")

    # Load market data (30s) - use full dataset
    market_30s_file = "/Users/radhikaarora/Documents/New Project/Input Data/market/mnq_30s.csv"
    market_df = pd.read_csv(market_30s_file)
    # Convert to UTC then to EDT naive for comparison
    market_df['timestamp'] = pd.to_datetime(market_df['timestamp'], utc=True)
    market_df['timestamp_edt'] = market_df['timestamp'].dt.tz_convert('America/New_York').dt.tz_localize(None)
    print(f"\nLoaded {len(market_df)} market 30s bars")
    print(f"Market date range (EDT): {market_df['timestamp_edt'].min()} to {market_df['timestamp_edt'].max()}")

    # Filter trades to only those within market data date range (using EDT times)
    trades_in_range = trades_df[
        (trades_df['entry_ts_edt'] >= market_df['timestamp_edt'].min()) &
        (trades_df['entry_ts_edt'] <= market_df['timestamp_edt'].max())
    ]
    print(f"\nTrades within market data range: {len(trades_in_range)}")

    if len(trades_in_range) == 0:
        print("WARNING: No trades within market data date range!")
        print(f"  Earliest market: {market_df['timestamp'].min()}")
        print(f"  Latest market: {market_df['timestamp'].max()}")
        print(f"  Earliest trade: {trades_df['entry_ts'].min()}")
        print(f"  Latest trade: {trades_df['entry_ts'].max()}")
        return

    # Attempt join
    print("\n=== TESTING JOIN ===")
    merged = pd.merge(
        trades_in_range[['entry_ts_edt', 'Entry Price', 'Side']],
        market_df[['timestamp_edt', 'close', 'volume']],
        left_on='entry_ts_edt',
        right_on='timestamp_edt',
        how='left'
    )

    matched = merged['close'].notna().sum()
    unmatched = merged['close'].isna().sum()

    print(f"Attempted join: {len(trades_in_range)} trades")
    print(f"  ✓ Matched with market data: {matched}")
    print(f"  ✗ Unmatched: {unmatched}")

    if unmatched > 0:
        print(f"\nUnmatched trades (could indicate missing market data):")
        unmatched_df = merged[merged['close'].isna()]
        print(unmatched_df[['entry_ts_edt', 'timestamp_edt', 'Entry Price']].head())

    if matched > 0:
        print(f"\n=== SAMPLE MATCHED TRADES ===")
        matched_df = merged[merged['close'].notna()]
        print(matched_df[['entry_ts_edt', 'Entry Price', 'close', 'volume']].head(10).to_string())

    # Summary
    print(f"\n=== SUMMARY ===")
    success_rate = (matched / len(trades_in_range)) * 100 if len(trades_in_range) > 0 else 0
    print(f"Success rate: {success_rate:.1f}% ({matched}/{len(trades_in_range)})")

    if success_rate == 100:
        print("✓ All trades successfully aligned with market data!")
    elif success_rate >= 90:
        print("✓ Timestamps mostly aligned. Small misalignments may exist.")
    else:
        print("⚠ Consider investigating timestamp alignment issues")


if __name__ == "__main__":
    main()
