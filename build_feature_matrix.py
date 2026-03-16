#!/usr/bin/env python3
"""
build_feature_matrix.py
-----------------------
Extracts 11 ML features for each trade in trades_combined.csv by replaying
30s market data from Pivot Time → Entry Time.

Outputs: ml_features_combined.csv  (one row per trade)

Features (matching bnr_deterministic_engine.py):
  retrace, pivot_flem_dist, time_since_pivot_sec,
  body_last, body_sum, body_mean,
  in_dir_ratio, max_in_dir_run, bars_since_pivot,
  zone_over_range, pivot_over_range
"""

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

TRADES   = "/Users/radhikaarora/Documents/Trading ML/ML V2/trades_combined.csv"
MKT_30S  = "/Users/radhikaarora/Documents/New Project/Input Data/market/mnq_30s.csv"
MKT_1M   = "/Users/radhikaarora/Documents/New Project/Input Data/market/mnq_1m.csv"
OUTPUT   = "/Users/radhikaarora/Documents/Trading ML/ML V2/ml_features_combined.csv"

ML_FEATURES = [
    "retrace", "pivot_flem_dist", "time_since_pivot_sec",
    "body_last", "body_sum", "body_mean",
    "in_dir_ratio", "max_in_dir_run", "bars_since_pivot",
    "zone_over_range", "pivot_over_range",
]

# Displacement thresholds (from engine lines 523-528)
HIGH_ZONE  = 0.10568226033342312
HIGH_PIVOT = 0.3795379537953795
LOW_ZONE   = 0.0848692546366965
LOW_PIVOT  = 0.23516193082722905

SESSION_START = "09:30"
SESSION_END   = "12:00"


# ── Load market data ──────────────────────────────────────────────────────────

print("Loading market data …")

df_30s = pd.read_csv(MKT_30S, parse_dates=["timestamp"])
df_30s["timestamp"] = pd.to_datetime(df_30s["timestamp"], utc=True)
df_30s = df_30s.sort_values("timestamp").set_index("timestamp")
print(f"  30s bars: {len(df_30s):,}  ({df_30s.index[0]} → {df_30s.index[-1]})")

df_1m = pd.read_csv(MKT_1M, parse_dates=["timestamp"])
df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"], utc=True)
df_1m = df_1m.sort_values("timestamp").set_index("timestamp")
print(f"  1m  bars: {len(df_1m):,}  ({df_1m.index[0]} → {df_1m.index[-1]})")


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_utc(ts) -> pd.Timestamp | None:
    """Ensure a timestamp is UTC-aware."""
    if ts is None or pd.isna(ts):
        return None
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def compute_features(row, df_30s, df_1m) -> dict | None:
    """
    Extract 11 ML features for a single trade row.
    Returns None if data is insufficient.
    """
    side        = str(row["Side"]).lower().strip()
    pivot_price = row["Pivot Price"]
    flem_price  = row["FLEM Price"]
    entry_ts    = to_utc(row["entry_ts"])

    # Parse FLEM/Pivot times
    pivot_time_raw = row["Pivot Time"]
    if pd.isna(pivot_time_raw) or str(pivot_time_raw).strip() == "":
        return None
    pivot_time = to_utc(pd.Timestamp(pivot_time_raw))

    flem_time_raw = row["FLEM Time"]
    if pd.isna(flem_time_raw) or str(flem_time_raw).strip() == "":
        return None

    if entry_ts is None or pivot_time is None:
        return None

    # Validate prices
    if pd.isna(pivot_price) or pd.isna(flem_price):
        return None
    if abs(flem_price - pivot_price) < 1e-9:
        return None  # degenerate

    # ── 30s window: pivot_time ≤ bar ≤ entry_ts ──────────────────────────────
    window = df_30s.loc[pivot_time:entry_ts]
    if window.empty:
        return None

    bars = len(window)
    closes = window["close"].values
    opens  = window["open"].values
    bodies = np.abs(closes - opens)

    body_last = float(bodies[-1])
    body_sum  = float(bodies.sum())
    body_mean = body_sum / bars if bars > 0 else 0.0

    # In-direction ratio and max run
    if side == "long":
        in_dir = closes > opens        # bullish candle = in direction
    else:
        in_dir = closes < opens        # bearish candle = in direction

    in_dir_count = int(in_dir.sum())
    in_dir_ratio = in_dir_count / bars if bars > 0 else 0.0

    max_run = 0
    current_run = 0
    for d in in_dir:
        if d:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    # Retrace at entry bar close
    entry_close = float(closes[-1])
    denom = flem_price - pivot_price
    if abs(denom) < 1e-9:
        return None
    if side == "long":
        retrace = (entry_close - pivot_price) / denom
    else:
        retrace = (pivot_price - entry_close) / denom

    # Time since pivot
    time_since_pivot_sec = (entry_ts - pivot_time).total_seconds()

    # pivot_flem_dist
    pivot_flem_dist = abs(flem_price - pivot_price)

    # ── Zone and range from 1m data ────────────────────────────────────────────
    # Get date in local EST/EDT time from the entry_ts
    entry_local = entry_ts.tz_convert("America/New_York")
    date_str    = entry_local.date()

    # Session window: 09:30 – 12:00 local on that date
    session_start = pd.Timestamp(f"{date_str} {SESSION_START}:00",
                                 tz="America/New_York").tz_convert("UTC")
    session_end   = pd.Timestamp(f"{date_str} {SESSION_END}:00",
                                 tz="America/New_York").tz_convert("UTC")

    day_1m = df_1m.loc[session_start:session_end]
    if day_1m.empty:
        return None

    day_range = float(day_1m["high"].max() - day_1m["low"].min())
    if day_range < 1e-9:
        return None

    # Zone bar = first 1m bar at session start (09:30 local)
    zone_bar = df_1m.get(session_start)
    if zone_bar is None:
        # Try slicing instead
        zone_slice = df_1m.loc[session_start:session_start]
        if zone_slice.empty:
            return None
        zone_bar = zone_slice.iloc[0]

    if side == "long":
        zone_price  = float(zone_bar["high"])
        dist_zone   = flem_price - zone_price
    else:
        zone_price  = float(zone_bar["low"])
        dist_zone   = zone_price - flem_price

    zone_over_range  = dist_zone    / day_range
    pivot_over_range = pivot_flem_dist / day_range

    # Displacement category
    if zone_over_range >= HIGH_ZONE and pivot_over_range >= HIGH_PIVOT:
        displacement = "high"
    elif zone_over_range <= LOW_ZONE and pivot_over_range <= LOW_PIVOT:
        displacement = "low"
    else:
        displacement = "medium"

    return {
        "open_date":            str(row["Open Date"]),
        "direction":            side,
        "displacement_category": displacement,
        "entry_time":           str(entry_ts),
        "pivot_time":           str(pivot_time),
        "flem_time":            str(to_utc(pd.Timestamp(flem_time_raw))),
        # features
        "retrace":              round(retrace, 8),
        "pivot_flem_dist":      round(pivot_flem_dist, 4),
        "time_since_pivot_sec": round(time_since_pivot_sec, 1),
        "body_last":            round(body_last, 4),
        "body_sum":             round(body_sum, 4),
        "body_mean":            round(body_mean, 4),
        "in_dir_ratio":         round(in_dir_ratio, 4),
        "max_in_dir_run":       int(max_run),
        "bars_since_pivot":     int(bars),
        "zone_over_range":      round(zone_over_range, 8),
        "pivot_over_range":     round(pivot_over_range, 8),
        # labels
        "label_valid":          1 if str(row["Setup Valid"]).strip().lower() == "yes" else 0,
        "label_win":            1 if float(row["Net P&L"]) > 0 else 0,
        # diagnostics
        "entry_price":          row["Entry Price"],
        "net_pnl":              row["Net P&L"],
        "setup_valid_raw":      row["Setup Valid"],
    }


# ── Process all trades ────────────────────────────────────────────────────────

print("\nLoading trades …")
df_trades = pd.read_csv(TRADES)
print(f"  {len(df_trades)} trades")

records  = []
skipped  = []

for idx, row in df_trades.iterrows():
    result = compute_features(row, df_30s, df_1m)
    if result is None:
        skipped.append({
            "idx": idx,
            "date": row["Open Date"],
            "side": row["Side"],
            "entry": row["Open Time"],
            "reason": "missing FLEM/Pivot data or no market bars",
        })
    else:
        records.append(result)

print(f"\n  ✓ Extracted features for {len(records)} trades")
print(f"  ✗ Skipped {len(skipped)} trades (no FLEM/Pivot data)")
if skipped:
    for s in skipped:
        print(f"      [{s['idx']:3d}] {s['date']} {s['side']:5s} @{s['entry']}")


# ── Build and validate DataFrame ──────────────────────────────────────────────

df_feat = pd.DataFrame(records)

print("\n=== VALIDATION ===")
print(f"Rows: {len(df_feat)}")
print(f"NaN counts per feature:")
print(df_feat[ML_FEATURES].isna().sum().to_string())

print(f"\nlabel_valid distribution:\n{df_feat['label_valid'].value_counts().to_string()}")
print(f"\nlabel_win distribution:\n{df_feat['label_win'].value_counts().to_string()}")

print(f"\ndisplacement_category distribution:\n{df_feat['displacement_category'].value_counts().to_string()}")

# Spot-check: Jan 3 long trade
jan3 = df_feat[df_feat["open_date"] == "2025-01-03"]
if not jan3.empty:
    print(f"\nSpot-check Jan 3 (expected: retrace≈0.77, bars=5, time=120s):")
    print(jan3[["open_date", "direction", "retrace", "bars_since_pivot",
                "time_since_pivot_sec", "label_valid", "label_win"]].to_string())


# ── Save ──────────────────────────────────────────────────────────────────────

col_order = [
    "open_date", "direction", "displacement_category",
    "entry_time", "pivot_time", "flem_time",
] + ML_FEATURES + ["label_valid", "label_win", "entry_price", "net_pnl", "setup_valid_raw"]

df_feat = df_feat[col_order]
df_feat.to_csv(OUTPUT, index=False)
print(f"\n✓ Saved {len(df_feat)} rows → {OUTPUT}")
