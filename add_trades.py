#!/usr/bin/env python3
"""
add_trades.py
-------------
Generic script to append a new trades CSV file to trades_combined.csv.
Handles all normalization automatically:
  - Ceiling-rounds timestamps to :30s boundary
  - Normalizes Setup Valid (empty/"" → "yes", "no" → "no")
  - Converts First Leg Max / Pivot Low/High (HH:MM) → full ISO FLEM/Pivot Times
  - Looks up FLEM Price and Pivot Price from 30s market data
  - Derives entry_ts (UTC) from Open Date + Open Time (handles EST and EDT)
  - Deduplicates on (Open Date, Open Time, Side, Entry Price)

Usage:
    python3 add_trades.py /path/to/new_trades.csv
"""

import sys
import re
import pandas as pd
import numpy as np
from datetime import timedelta

# ── Paths ─────────────────────────────────────────────────────────────────────

COMBINED     = "/Users/radhikaarora/Documents/Trading ML/ML V2/trades_combined.csv"
MKT_30S_MAIN = "/Users/radhikaarora/Documents/New Project/Input Data/market/mnq_30s.csv"
MKT_30S_JANFEB = "/Users/radhikaarora/Documents/New Project/Input Data/market/mnq_30s_jan_feb_2026.csv"

COLUMNS = [
    "Open Date", "Open Time", "Close Time", "Duration", "Net P&L",
    "Side", "Entry Price", "Symbol", "Executions", "Exit Price",
    "First Leg Max", "Pivot Low/High", "Setup Valid",
    "entry_ts",
    "FLEM Price", "FLEM Time", "Pivot Price", "Pivot Time",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def ceil_to_30s_str(time_str: str) -> str:
    s = str(time_str).strip()
    if not s:
        return s
    m = re.match(r"(\d{1,2}):(\d{2}):(\d{2})(.*)", s)
    if not m:
        return s
    h, mi, sec, rest = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
    if sec not in (0, 30):
        if 1 <= sec <= 29:
            sec = 30
        else:
            sec = 0; mi += 1
            if mi >= 60: mi = 0; h += 1
            if h >= 24: h = 0
    return f"{h:02d}:{mi:02d}:{sec:02d}{rest}"


def ceil_to_30s_ts(ts) -> pd.Timestamp:
    if pd.isna(ts):
        return ts
    s = ts.second
    if s in (0, 30):
        return ts
    if 1 <= s <= 29:
        return ts.replace(second=30, microsecond=0)
    return ts.replace(second=0, microsecond=0) + timedelta(minutes=1)


def normalize_setup_valid(val) -> str:
    if pd.isna(val):
        return "yes"
    cleaned = str(val).strip().strip('"').lower()
    return "no" if cleaned == "no" else "yes"


def tz_offset_from_str(time_str: str) -> str:
    """Extract UTC offset from time string containing EDT or EST."""
    s = str(time_str).upper()
    if "EDT" in s:
        return "-04:00"
    return "-05:00"   # default EST


def open_time_to_utc(date_str: str, open_time_str: str) -> pd.Timestamp | None:
    """Parse 'HH:MM:SS EDT/EST' + date into UTC Timestamp."""
    s = str(open_time_str).strip()
    m = re.match(r"(\d{1,2}):(\d{2}):(\d{2})\s*(EDT|EST)?", s, re.IGNORECASE)
    if not m:
        return None
    h, mi, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
    tz_label  = (m.group(4) or "EST").upper()
    tz_offset = "-04:00" if tz_label == "EDT" else "-05:00"
    try:
        ts_local = pd.Timestamp(f"{date_str} {h:02d}:{mi:02d}:{sec:02d}{tz_offset}")
        return ts_local.tz_convert("UTC")
    except Exception:
        return None


def hhmm_to_iso(date_str: str, hhmm: str, tz_offset: str) -> pd.Timestamp | None:
    """Convert 'H:MM' + date + tz_offset to timezone-aware Timestamp."""
    s = str(hhmm).strip().strip('"')
    if not s:
        return None
    m = re.match(r"(\d{1,2}):(\d{2})$", s)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    try:
        return pd.Timestamp(f"{date_str} {h:02d}:{mi:02d}:00{tz_offset}")
    except Exception:
        return None


def lookup_price(ts, side: str, mkt: dict, col_long: str, col_short: str):
    if ts is None or pd.isna(ts):
        return None
    ts_utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    bar = mkt.get(ts_utc)
    if bar is None:
        return None
    col = col_long if str(side).lower() == "long" else col_short
    return bar.get(col)


# ── Load market data ──────────────────────────────────────────────────────────

print("Loading market data …")
dfs_mkt = []
for path in [MKT_30S_MAIN, MKT_30S_JANFEB]:
    try:
        df_m = pd.read_csv(path, parse_dates=["timestamp"])
        df_m["timestamp"] = pd.to_datetime(df_m["timestamp"], utc=True)
        dfs_mkt.append(df_m)
        print(f"  ✓ {path.split('/')[-1]}  ({len(df_m):,} rows)")
    except FileNotFoundError:
        print(f"  ✗ Not found: {path}")

mkt_df   = pd.concat(dfs_mkt, ignore_index=True).drop_duplicates(subset="timestamp").set_index("timestamp")
mkt_dict = mkt_df[["open", "high", "low", "close"]].to_dict(orient="index")
print(f"  Market data: {len(mkt_dict):,} unique bars\n")


# ── Load existing combined file ───────────────────────────────────────────────

print(f"Loading existing trades_combined.csv …")
df_existing = pd.read_csv(COMBINED)
print(f"  {len(df_existing)} rows\n")


# ── Load & process new file ───────────────────────────────────────────────────

new_file = sys.argv[1] if len(sys.argv) > 1 else None
if not new_file:
    print("Usage: python3 add_trades.py /path/to/new_trades.csv")
    sys.exit(1)

print(f"Loading new file: {new_file.split('/')[-1]}")
df_new = pd.read_csv(new_file)
print(f"  {len(df_new)} rows")

# Normalize time strings
df_new["Open Time"]  = df_new["Open Time"].apply(ceil_to_30s_str)
df_new["Close Time"] = df_new["Close Time"].apply(ceil_to_30s_str)

# Normalize Setup Valid
before = df_new["Setup Valid"].value_counts(dropna=False).to_dict()
df_new["Setup Valid"] = df_new["Setup Valid"].apply(normalize_setup_valid)
after  = df_new["Setup Valid"].value_counts(dropna=False).to_dict()
print(f"  Setup Valid: {before} → {after}")

# Derive entry_ts from Open Date + Open Time (auto-detects EST/EDT)
df_new["entry_ts"] = df_new.apply(
    lambda r: open_time_to_utc(str(r["Open Date"]), r["Open Time"]), axis=1
)
df_new["entry_ts"] = df_new["entry_ts"].apply(
    lambda ts: ceil_to_30s_ts(ts) if ts is not None else pd.NaT
)

# Convert First Leg Max → FLEM Time  (HH:MM → full ISO, using timezone from Open Time)
def row_flem_time(r):
    tz_off = tz_offset_from_str(r["Open Time"])
    return hhmm_to_iso(str(r["Open Date"]), r["First Leg Max"], tz_off)

def row_pivot_time(r):
    tz_off = tz_offset_from_str(r["Open Time"])
    return hhmm_to_iso(str(r["Open Date"]), r["Pivot Low/High"], tz_off)

df_new["FLEM Time"]  = df_new.apply(row_flem_time,  axis=1)
df_new["Pivot Time"] = df_new.apply(row_pivot_time, axis=1)

# Look up FLEM/Pivot prices from market data
df_new["FLEM Price"]  = df_new.apply(
    lambda r: lookup_price(r["FLEM Time"],  r["Side"], mkt_dict, "high", "low"), axis=1)
df_new["Pivot Price"] = df_new.apply(
    lambda r: lookup_price(r["Pivot Time"], r["Side"], mkt_dict, "low", "high"), axis=1)

flem_filled  = df_new["FLEM Price"].notna().sum()
pivot_filled = df_new["Pivot Price"].notna().sum()
flem_total   = df_new["FLEM Time"].notna().sum()
pivot_total  = df_new["Pivot Time"].notna().sum()
print(f"  FLEM  Price populated: {flem_filled}/{flem_total}")
print(f"  Pivot Price populated: {pivot_filled}/{pivot_total}")

# Ensure all canonical columns exist
for col in COLUMNS:
    if col not in df_new.columns:
        df_new[col] = None


# ── Combine & deduplicate ─────────────────────────────────────────────────────

print("\nCombining …")
df_combined = pd.concat([df_existing[COLUMNS], df_new[COLUMNS]], ignore_index=True)
df_combined["entry_ts"] = pd.to_datetime(df_combined["entry_ts"], utc=True, errors="coerce")

# Dedup: keep first occurrence on (Open Date, Open Time, Side, Entry Price)
before_dedup = len(df_combined)
df_combined = df_combined.drop_duplicates(
    subset=["Open Date", "Open Time", "Side", "Entry Price"], keep="first"
)
after_dedup = len(df_combined)
if before_dedup != after_dedup:
    print(f"  Removed {before_dedup - after_dedup} duplicate rows")

df_combined = df_combined.sort_values("entry_ts").reset_index(drop=True)
print(f"  Combined: {len(df_combined)} total trades")
print(f"  Date range: {df_combined['entry_ts'].min()} → {df_combined['entry_ts'].max()}")


# ── Validation ────────────────────────────────────────────────────────────────

def has_bad_seconds(t):
    m = re.search(r":(\d{2}) ", str(t))
    if m:
        s = int(m.group(1))
        return s not in (0, 30)
    return False

bad_open  = df_combined["Open Time"].apply(has_bad_seconds).sum()
bad_close = df_combined["Close Time"].apply(has_bad_seconds).sum()
sv_counts = df_combined["Setup Valid"].value_counts(dropna=False)
null_ts   = df_combined["entry_ts"].isna().sum()

print(f"\n=== VALIDATION ===")
print(f"Non-:00/:30 Open Time:  {bad_open}")
print(f"Non-:00/:30 Close Time: {bad_close}")
print(f"Setup Valid: {sv_counts.to_dict()}")
print(f"entry_ts nulls: {null_ts}")


# ── Save ──────────────────────────────────────────────────────────────────────

df_combined.to_csv(COMBINED, index=False)
print(f"\n✓ Saved {len(df_combined)} rows → trades_combined.csv")
print("\nNext steps:")
print("  python3 populate_flem_pivot.py")
print("  python3 build_feature_matrix.py")
print("  python3 train_models.py")
