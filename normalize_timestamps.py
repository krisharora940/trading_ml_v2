#!/usr/bin/env python3
"""
Normalize trade entry and exit timestamps to 30-second intervals.
Aligns timestamps with market data format (:00 and :30 seconds only).

Rounding logic:
  - :00, :30 → keep as-is
  - :01-:29 → round UP to :30 of same minute
  - :31-:59 → round UP to :00 of next minute
"""

import pandas as pd
from datetime import timedelta
import re

def normalize_timestamp(ts):
    """Round timestamp to nearest 30-second boundary (ceiling)."""
    if pd.isna(ts):
        return ts

    seconds = ts.second
    microseconds = ts.microsecond

    # Already on a 30-second boundary
    if seconds in (0, 30):
        return ts

    # Round up to :30 of same minute
    if seconds < 30:
        return ts.replace(second=30, microsecond=0)

    # Round up to :00 of next minute (seconds >= 31)
    return ts.replace(second=0, microsecond=0) + timedelta(minutes=1)


def parse_time_column(time_str):
    """Parse 'HH:MM:SS TZ' format (e.g. '09:33:00 EDT')."""
    if pd.isna(time_str) or time_str == '':
        return None
    try:
        # Extract just the time part (HH:MM:SS)
        match = re.match(r'(\d{1,2}):(\d{2}):(\d{2})', str(time_str))
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
    except:
        pass
    return None


def format_time_column(hour, minute, second, tz='EDT'):
    """Format time back to 'HH:MM:SS TZ' format."""
    return f"{hour:02d}:{minute:02d}:{second:02d} {tz}"


def normalize_time_string(time_str, tz='EDT'):
    """Normalize a time string (HH:MM:SS TZ) to nearest 30-second boundary."""
    if pd.isna(time_str) or time_str == '':
        return ''

    parsed = parse_time_column(time_str)
    if parsed is None:
        return time_str

    hour, minute, second = parsed

    # Already on a 30-second boundary
    if second in (0, 30):
        return format_time_column(hour, minute, second, tz)

    # Round up to :30 of same minute
    if second < 30:
        return format_time_column(hour, minute, 30, tz)

    # Round up to :00 of next minute (second >= 31)
    minute += 1
    if minute >= 60:
        minute = 0
        hour += 1
        if hour >= 24:
            hour = 0
    return format_time_column(hour, minute, 0, tz)


def main():
    trades_file = "/Users/radhikaarora/Documents/Trading ML/ML V2/trades_20260315014828_with_flem_pivot.csv"

    print(f"Loading trades from: {trades_file}")
    df = pd.read_csv(trades_file)
    print(f"  Loaded {len(df)} trades")

    # Display sample of problematic timestamps (only check seconds position)
    print("\n=== BEFORE NORMALIZATION ===")
    problematic = df[df['Open Time'].str.contains(r':(29|59) ', regex=True, na=False)]
    if len(problematic) > 0:
        print(f"Found {len(problematic)} rows with :29 or :59 in seconds:")
        print(problematic[['Open Date', 'Open Time', 'Close Time', 'entry_ts']].head(10).to_string())

    # Normalize Open Time
    print("\nNormalizing 'Open Time' column...")
    df['Open Time'] = df['Open Time'].apply(lambda x: normalize_time_string(x, 'EDT'))

    # Normalize Close Time
    print("Normalizing 'Close Time' column...")
    df['Close Time'] = df['Close Time'].apply(lambda x: normalize_time_string(x, 'EDT'))

    # Also normalize entry_ts (in case it has similar issues, convert from UTC)
    if 'entry_ts' in df.columns:
        print("Normalizing 'entry_ts' column...")
        df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
        df['entry_ts'] = df['entry_ts'].apply(normalize_timestamp)

    # Verify all timestamps are on 30-second boundaries
    print("\n=== VALIDATION ===")

    # Check Open Time (only seconds position matters)
    if 'Open Time' in df.columns:
        bad_open = df[df['Open Time'].str.contains(r':(29|59) ', regex=True, na=False)]
        if len(bad_open) > 0:
            print(f"⚠ WARNING: Found {len(bad_open)} rows with :29 or :59 in seconds position")
        else:
            print("✓ Open Time: All timestamps at :00 or :30 boundaries")

    # Check Close Time (only seconds position matters)
    if 'Close Time' in df.columns:
        bad_close = df[(df['Close Time'] != '') &
                       df['Close Time'].str.contains(r':(29|59) ', regex=True, na=False)]
        if len(bad_close) > 0:
            print(f"⚠ WARNING: Found {len(bad_close)} rows with :29 or :59 in seconds position")
        else:
            print("✓ Close Time: All timestamps at :00 or :30 boundaries (or empty)")

    # Check entry_ts
    if 'entry_ts' in df.columns:
        entry_seconds = df['entry_ts'].dt.second.unique()
        if all(s in (0, 30) for s in entry_seconds if pd.notna(s)):
            print("✓ entry_ts: All timestamps at :00 or :30 boundaries")
        else:
            print(f"⚠ WARNING: entry_ts has invalid seconds: {sorted(entry_seconds)}")

    print(f"\nTotal rows: {len(df)}")

    # Save the normalized trades
    print(f"\nSaving normalized trades to: {trades_file}")
    df.to_csv(trades_file, index=False)
    print("✓ Timestamps normalized and file saved")

    # Show sample of normalized data
    print("\n=== SAMPLE NORMALIZED TRADES ===")
    cols = ['Open Date', 'Open Time', 'Close Time', 'entry_ts', 'Entry Price', 'Side']
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(15).to_string())

    # Show the specific rows that were fixed
    print("\n=== NORMALIZED PROBLEM ROWS ===")
    if len(problematic) > 0:
        original_ot = problematic['Open Time'].tolist()
        normalized_ot = df.loc[problematic.index, 'Open Time'].tolist()
        print("\nOpen Time changes:")
        for orig, norm in zip(original_ot[:10], normalized_ot[:10]):
            print(f"  {orig:15} → {norm:15}")


if __name__ == "__main__":
    main()
