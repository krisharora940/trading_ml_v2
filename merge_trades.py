#!/usr/bin/env python3
"""
merge_trades.py
---------------
Merges two manually-backtested trades files into one clean dataset for ML training.

Actions performed:
  1. Normalize Open/Close Time strings to nearest :30s ceiling boundary
  2. Normalize Setup Valid  → strictly "yes" or "no"
  3. Existing file: FLEM Time / Pivot Time already full ISO — look up prices from market data
  4. New file: convert First Leg Max (HH:MM) → FLEM Time;
                      Pivot Low/High (HH:MM) → Pivot Time;
                      then look up prices
  5. Derive / keep entry_ts as UTC-normalized timestamp (join key)
  6. Combine, sort by entry_ts, save to trades_combined.csv
"""

import re
import pandas as pd
from datetime import timedelta

# ── Paths ─────────────────────────────────────────────────────────────────────

EXISTING_FILE = (
    "/Users/radhikaarora/Documents/Trading ML/ML V2/"
    "trades_20260315014828_with_flem_pivot.csv"
)
NEW_FILE = "/Users/radhikaarora/Downloads/trades_20260315031021.csv"

MKT_30S_MAIN   = "/Users/radhikaarora/Documents/New Project/Input Data/market/mnq_30s.csv"
MKT_30S_JANFEB = "/Users/radhikaarora/Documents/New Project/Input Data/market/mnq_30s_jan_feb_2026.csv"

OUTPUT = "/Users/radhikaarora/Documents/Trading ML/ML V2/trades_combined.csv"

# Canonical column order for output
COLUMNS = [
    "Open Date", "Open Time", "Close Time", "Duration", "Net P&L",
    "Side", "Entry Price", "Symbol", "Executions", "Exit Price",
    "First Leg Max", "Pivot Low/High", "Setup Valid",
    "entry_ts",
    "FLEM Price", "FLEM Time", "Pivot Price", "Pivot Time",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def ceil_to_30s_str(time_str: str) -> str:
    """
    Ceiling-round a 'HH:MM:SS TZ' string to the nearest :30s boundary.
    '09:36:59 EDT' → '09:37:00 EDT'
    '09:36:29 EDT' → '09:36:30 EDT'
    '09:36:30 EDT' → '09:36:30 EDT'  (unchanged)
    Non-matching strings are returned as-is.
    """
    s = str(time_str).strip()
    if not s:
        return s
    m = re.match(r"(\d{1,2}):(\d{2}):(\d{2})(.*)", s)
    if not m:
        return s

    h, mi, sec, rest = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)

    if sec in (0, 30):
        return s  # already on boundary

    if 1 <= sec <= 29:
        sec = 30
    else:  # 31–59
        sec = 0
        mi += 1
        if mi >= 60:
            mi = 0
            h += 1
            if h >= 24:
                h = 0

    return f"{h:02d}:{mi:02d}:{sec:02d}{rest}"


def ceil_to_30s_ts(ts: pd.Timestamp) -> pd.Timestamp:
    """Ceiling-round a timezone-aware Timestamp to nearest :30s."""
    if pd.isna(ts):
        return ts
    s = ts.second
    if s in (0, 30):
        return ts
    if 1 <= s <= 29:
        return ts.replace(second=30, microsecond=0)
    # 31–59 → :00 of next minute
    return (ts.replace(second=0, microsecond=0) + timedelta(minutes=1))


def normalize_setup_valid(val) -> str:
    """'no' → 'no'; everything else (blank, empty, any other value) → 'yes'."""
    if pd.isna(val):
        return "yes"
    cleaned = str(val).strip().strip('"').lower()
    return "no" if cleaned == "no" else "yes"


def hhmm_to_iso(date_str: str, hhmm: str, tz_offset: str) -> pd.Timestamp | None:
    """
    Convert Open Date + 'H:MM' or 'HH:MM' string to a timezone-aware Timestamp.
    Returns None if hhmm is empty or unparseable.
    tz_offset: e.g. '-05:00' for EST
    """
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


def open_time_to_utc(date_str: str, open_time_str: str) -> pd.Timestamp | None:
    """
    Parse 'HH:MM:SS EDT/EST' + date into a UTC Timestamp.
    EDT → -04:00, EST → -05:00.
    """
    s = str(open_time_str).strip()
    m = re.match(r"(\d{1,2}):(\d{2}):(\d{2})\s*(EDT|EST)?", s, re.IGNORECASE)
    if not m:
        return None
    h, mi, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
    tz_label = (m.group(4) or "EST").upper()
    tz_offset = "-04:00" if tz_label == "EDT" else "-05:00"
    try:
        ts_local = pd.Timestamp(f"{date_str} {h:02d}:{mi:02d}:{sec:02d}{tz_offset}")
        return ts_local.tz_convert("UTC")
    except Exception:
        return None


def lookup_price(ts, side: str, mkt: dict, col_long: str, col_short: str):
    """
    Look up OHLC price from mkt dict at timestamp ts.
    col_long / col_short: which column to return based on direction.
    """
    if ts is None or pd.isna(ts):
        return None
    # Normalise to UTC for lookup
    if ts.tzinfo is not None:
        ts_utc = ts.tz_convert("UTC")
    else:
        ts_utc = ts.tz_localize("UTC")
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

mkt_df = pd.concat(dfs_mkt, ignore_index=True).drop_duplicates(subset="timestamp")
mkt_df = mkt_df.set_index("timestamp")

# Build lookup dict: UTC Timestamp → row dict
mkt_dict = mkt_df[["open", "high", "low", "close"]].to_dict(orient="index")
print(f"  Market data: {len(mkt_dict):,} unique bars\n")


# ── Load & process EXISTING file ──────────────────────────────────────────────

print(f"Loading existing trades: {EXISTING_FILE.split('/')[-1]}")
df_ex = pd.read_csv(EXISTING_FILE)
print(f"  {len(df_ex)} rows")

# Normalize time strings
df_ex["Open Time"]  = df_ex["Open Time"].apply(ceil_to_30s_str)
df_ex["Close Time"] = df_ex["Close Time"].apply(ceil_to_30s_str)

# Normalize entry_ts (already UTC Timestamp strings)
df_ex["entry_ts"] = pd.to_datetime(df_ex["entry_ts"], utc=True, errors="coerce")
df_ex["entry_ts"] = df_ex["entry_ts"].apply(ceil_to_30s_ts)

# Normalize Setup Valid
df_ex["Setup Valid"] = df_ex["Setup Valid"].apply(normalize_setup_valid)

# Parse FLEM Time / Pivot Time (already full ISO timestamps)
df_ex["FLEM Time"]  = pd.to_datetime(df_ex["FLEM Time"],  errors="coerce", utc=False)
df_ex["Pivot Time"] = pd.to_datetime(df_ex["Pivot Time"], errors="coerce", utc=False)

# Look up FLEM Price and Pivot Price
def get_flem_price(row):
    return lookup_price(row["FLEM Time"], row["Side"], mkt_dict, "high", "low")

def get_pivot_price(row):
    return lookup_price(row["Pivot Time"], row["Side"], mkt_dict, "low", "high")

df_ex["FLEM Price"]  = df_ex.apply(get_flem_price,  axis=1)
df_ex["Pivot Price"] = df_ex.apply(get_pivot_price, axis=1)

flem_found  = df_ex["FLEM Price"].notna().sum()
pivot_found = df_ex["Pivot Price"].notna().sum()
flem_total  = df_ex["FLEM Time"].notna().sum()
pivot_total = df_ex["Pivot Time"].notna().sum()
print(f"  FLEM Price populated:  {flem_found}/{flem_total}")
print(f"  Pivot Price populated: {pivot_found}/{pivot_total}")


# ── Load & process NEW file ───────────────────────────────────────────────────

print(f"\nLoading new trades: {NEW_FILE.split('/')[-1]}")
df_new = pd.read_csv(NEW_FILE)
print(f"  {len(df_new)} rows")

# New file: all dates are EST (Jan/Feb/Nov 2025)
NEW_TZ_OFFSET = "-05:00"

# Normalize time strings
df_new["Open Time"]  = df_new["Open Time"].apply(ceil_to_30s_str)
df_new["Close Time"] = df_new["Close Time"].apply(ceil_to_30s_str)

# Normalize Setup Valid (all "" → "yes")
df_new["Setup Valid"] = df_new["Setup Valid"].apply(normalize_setup_valid)

# Convert First Leg Max (HH:MM) → FLEM Time (full ISO timestamp)
df_new["FLEM Time"] = df_new.apply(
    lambda r: hhmm_to_iso(str(r["Open Date"]), r["First Leg Max"], NEW_TZ_OFFSET),
    axis=1,
)

# Convert Pivot Low/High (HH:MM) → Pivot Time (full ISO timestamp)
df_new["Pivot Time"] = df_new.apply(
    lambda r: hhmm_to_iso(str(r["Open Date"]), r["Pivot Low/High"], NEW_TZ_OFFSET),
    axis=1,
)

# Look up FLEM Price and Pivot Price
df_new["FLEM Price"]  = df_new.apply(get_flem_price,  axis=1)
df_new["Pivot Price"] = df_new.apply(get_pivot_price, axis=1)

flem_found2  = df_new["FLEM Price"].notna().sum()
pivot_found2 = df_new["Pivot Price"].notna().sum()
flem_total2  = df_new["FLEM Time"].notna().sum()
pivot_total2 = df_new["Pivot Time"].notna().sum()
print(f"  FLEM Price populated:  {flem_found2}/{flem_total2}")
print(f"  Pivot Price populated: {pivot_found2}/{pivot_total2}")

# Derive entry_ts from Open Date + Open Time → UTC
df_new["entry_ts"] = df_new.apply(
    lambda r: open_time_to_utc(str(r["Open Date"]), r["Open Time"]), axis=1
)
# Normalize entry_ts to :30s ceiling (after string was already normalized, but let's be safe)
df_new["entry_ts"] = df_new["entry_ts"].apply(
    lambda ts: ceil_to_30s_ts(ts) if ts is not None else pd.NaT
)

# Add missing columns so concat aligns
for col in COLUMNS:
    if col not in df_new.columns:
        df_new[col] = None


# ── Combine ───────────────────────────────────────────────────────────────────

print("\nCombining …")
df_combined = pd.concat([df_ex[COLUMNS], df_new[COLUMNS]], ignore_index=True)
df_combined["entry_ts"] = pd.to_datetime(df_combined["entry_ts"], utc=True, errors="coerce")
df_combined = df_combined.sort_values("entry_ts").reset_index(drop=True)
print(f"  Combined: {len(df_combined)} total trades")
print(f"  Date range: {df_combined['entry_ts'].min()} → {df_combined['entry_ts'].max()}")


# ── Validation ────────────────────────────────────────────────────────────────

print("\n=== VALIDATION ===")

bad_open = df_combined[
    df_combined["Open Time"].astype(str).str.contains(r":[0-2][1-9] |:[3-5][1-9] ", regex=True, na=False)
]
# More precise: check seconds not in {00, 30}
def has_bad_seconds(t):
    m = re.search(r":(\d{2}) ", str(t))
    if m:
        s = int(m.group(1))
        return s not in (0, 30)
    return False

bad_open_precise  = df_combined[df_combined["Open Time"].apply(has_bad_seconds)]
bad_close_precise = df_combined[df_combined["Close Time"].apply(has_bad_seconds)]
print(f"Open Time  non-:00/:30 remaining:  {len(bad_open_precise)}")
print(f"Close Time non-:00/:30 remaining:  {len(bad_close_precise)}")

sv_counts = df_combined["Setup Valid"].value_counts(dropna=False)
print(f"Setup Valid distribution:\n{sv_counts.to_string()}")

null_ts = df_combined["entry_ts"].isna().sum()
print(f"entry_ts nulls: {null_ts}")

total_flem  = df_combined["FLEM Time"].notna().sum()
filled_flem = df_combined["FLEM Price"].notna().sum()
total_pvt   = df_combined["Pivot Time"].notna().sum()
filled_pvt  = df_combined["Pivot Price"].notna().sum()
print(f"FLEM  Price filled: {filled_flem}/{total_flem}")
print(f"Pivot Price filled: {filled_pvt}/{total_pvt}")

# Spot-check: show a few rows with FLEM/Pivot prices
spot = df_combined[df_combined["FLEM Price"].notna()][
    ["Open Date", "Open Time", "Side", "FLEM Time", "FLEM Price", "Pivot Time", "Pivot Price", "Setup Valid"]
].head(5)
print(f"\nSpot-check (first 5 with FLEM Price):\n{spot.to_string()}")


# ── Save ──────────────────────────────────────────────────────────────────────

print(f"\nSaving → {OUTPUT}")
df_combined.to_csv(OUTPUT, index=False)
print(f"✓ Saved {len(df_combined)} rows to trades_combined.csv")
