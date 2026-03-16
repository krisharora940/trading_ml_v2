#!/usr/bin/env python3
"""
populate_flem_pivot.py
----------------------
For trades in trades_combined.csv that are missing FLEM/Pivot data,
this script replays the 1m market data using the same FLEM/Pivot detection
logic from bnr_deterministic_engine.py to automatically populate:
  - FLEM Price, FLEM Time
  - Pivot Price, Pivot Time

State machine mirrors the engine exactly:
  1. Zone: 09:30 1m bar → zone_high, zone_low
  2. Breakout: 1m close > zone_high (long) or < zone_low (short)
  3. FLEM: running max(high) for long / min(low) for short after breakout
  4. Retest: opposite-color 1m bar that spans the zone → locks FLEM, sets Pivot
  5. Post-retest pivot updates: pivot moves to new low/high until entry
  6. Retrace reset: if retrace > 120%, FLEM resets and seeks new retest

After running, re-run build_feature_matrix.py → train_models.py to retrain.
"""

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

TRADES_CSV = "/Users/radhikaarora/Documents/Trading ML/ML V2/trades_combined.csv"
MKT_1M     = "/Users/radhikaarora/Documents/New Project/Input Data/market/mnq_1m.csv"
OUTPUT     = "/Users/radhikaarora/Documents/Trading ML/ML V2/trades_combined.csv"  # overwrite

SESSION_START = "09:30"
MAX_RETRACE   = 1.2   # retrace reset threshold (matches engine)


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading market data (1m) …")
df_1m = pd.read_csv(MKT_1M, parse_dates=["timestamp"])
df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"], utc=True)
df_1m = df_1m.sort_values("timestamp")
print(f"  {len(df_1m):,} bars  ({df_1m['timestamp'].iloc[0]} → {df_1m['timestamp'].iloc[-1]})")

print("\nLoading trades …")
df = pd.read_csv(TRADES_CSV)
print(f"  {len(df)} trades")

# Identify trades with missing FLEM/Pivot
df["FLEM Time"]  = df["FLEM Time"].replace("", np.nan)
df["Pivot Time"] = df["Pivot Time"].replace("", np.nan)
missing_mask = df["FLEM Time"].isna() | df["Pivot Time"].isna()
print(f"  Missing FLEM/Pivot: {missing_mask.sum()} trades")


# ── FLEM/Pivot state machine ──────────────────────────────────────────────────

def detect_flem_pivot(direction: str, entry_ts_utc: pd.Timestamp,
                      df_1m: pd.DataFrame, verbose: bool = False) -> dict | None:
    """
    Replay 1m bars from 09:30 up to (but not including) entry_ts_utc and
    return detected FLEM and Pivot, or None if detection fails.

    Returns dict with keys:
        flem_price, flem_time, pivot_price, pivot_time
    """
    side = direction.lower().strip()

    # Convert entry_ts to local time for date extraction
    entry_local = entry_ts_utc.tz_convert("America/New_York")
    date_str    = str(entry_local.date())

    # Session start in UTC
    session_start_local = pd.Timestamp(
        f"{date_str} {SESSION_START}:00", tz="America/New_York"
    )
    session_start_utc = session_start_local.tz_convert("UTC")

    # Slice 1m bars for this date: from session start up to (not including) entry bar
    day_bars = df_1m[
        (df_1m["timestamp"] >= session_start_utc) &
        (df_1m["timestamp"] < entry_ts_utc)
    ].copy()

    if day_bars.empty:
        if verbose: print(f"    No 1m bars found for {date_str}")
        return None

    # ── Zone bar (09:30 bar) ──────────────────────────────────────────────────
    zone_bar = day_bars[day_bars["timestamp"] == session_start_utc]
    if zone_bar.empty:
        if verbose: print(f"    Zone bar (09:30) not found for {date_str}")
        return None

    zone_high = float(zone_bar.iloc[0]["high"])
    zone_low  = float(zone_bar.iloc[0]["low"])
    if verbose: print(f"    Zone: high={zone_high}  low={zone_low}")

    # ── State machine variables ───────────────────────────────────────────────
    breakout_seen = False
    flem          = None
    flem_time     = None   # bar timestamp (open time) when FLEM is locked
    retest_seen   = False
    pivot         = None
    pivot_time    = None   # bar timestamp (open time) of pivot bar

    # Iterate 1m bars (skip zone bar itself — zone is 09:30, first processed bar is 09:31)
    for _, bar in day_bars.iterrows():
        ts    = bar["timestamp"]
        open_ = float(bar["open"])
        high  = float(bar["high"])
        low   = float(bar["low"])
        close = float(bar["close"])

        if ts == session_start_utc:
            continue   # zone bar, skip

        # ── Phase 1: Waiting for breakout ─────────────────────────────────────
        if not breakout_seen:
            if side == "long"  and close > zone_high:
                breakout_seen = True
                flem = high
                if verbose: print(f"    Breakout LONG  @ {ts}  close={close}  flem_start={flem}")
            elif side == "short" and close < zone_low:
                breakout_seen = True
                flem = low
                if verbose: print(f"    Breakout SHORT @ {ts}  close={close}  flem_start={flem}")
            continue

        # ── Phase 2: Breakout seen, waiting for retest ────────────────────────
        if breakout_seen and not retest_seen:
            # Update running FLEM
            if side == "long":
                if high > flem:
                    flem = high
            else:
                if low < flem:
                    flem = low

            # Retest condition: opposite-color candle spanning the zone
            if side == "long":
                is_retest = (close < open_) and (low <= zone_high) and (high >= zone_low)
                if is_retest:
                    retest_seen = True
                    flem_time   = ts          # bar open time
                    pivot       = low
                    pivot_time  = ts
                    if verbose:
                        print(f"    Retest LONG  @ {ts}  flem={flem}  pivot={pivot}")
            else:
                is_retest = (close > open_) and (low <= zone_high) and (high >= zone_low)
                if is_retest:
                    retest_seen = True
                    flem_time   = ts
                    pivot       = high
                    pivot_time  = ts
                    if verbose:
                        print(f"    Retest SHORT @ {ts}  flem={flem}  pivot={pivot}")
            continue

        # ── Phase 3: Post-retest — update pivot, handle retrace reset ─────────
        if retest_seen:
            # Compute current retrace
            if side == "long":
                denom = flem - pivot if flem and pivot else None
            else:
                denom = pivot - flem if flem and pivot else None

            if denom and abs(denom) > 1e-9:
                if side == "long":
                    retrace = (close - pivot) / denom
                else:
                    retrace = (pivot - close) / denom

                # Retrace reset: if price moved > 120% retrace, reset and seek new retest
                if retrace > MAX_RETRACE:
                    if verbose:
                        print(f"    Retrace reset @ {ts}  retrace={retrace:.3f}  "
                              f"flem reset to current {'high' if side=='long' else 'low'}")
                    flem        = high if side == "long" else low
                    flem_time   = None
                    retest_seen = False
                    pivot       = None
                    pivot_time  = None
                    continue

            # Update pivot to new low/high
            if side == "long":
                if low < pivot:
                    pivot      = low
                    pivot_time = ts
                    if verbose: print(f"    Pivot updated LONG  → {pivot} @ {ts}")
            else:
                if high > pivot:
                    pivot      = high
                    pivot_time = ts
                    if verbose: print(f"    Pivot updated SHORT → {pivot} @ {ts}")

    # ── Return result ─────────────────────────────────────────────────────────
    if retest_seen and flem is not None and pivot is not None:
        return {
            "flem_price":  round(flem, 4),
            "flem_time":   flem_time,
            "pivot_price": round(pivot, 4),
            "pivot_time":  pivot_time,
        }
    else:
        if verbose:
            print(f"    Detection incomplete: retest_seen={retest_seen}  "
                  f"flem={flem}  pivot={pivot}")
        return None


def ts_to_iso(ts: pd.Timestamp | None, tz: str = "America/New_York") -> str:
    """Convert UTC Timestamp to local ISO string for storage."""
    if ts is None:
        return ""
    return str(ts.tz_convert(tz))


# ── Process missing trades ────────────────────────────────────────────────────

print("\nProcessing missing trades …\n")
populated = 0
still_missing = []

for idx in df[missing_mask].index:
    row = df.loc[idx]
    date     = str(row["Open Date"])
    side     = str(row["Side"])
    open_t   = str(row["Open Time"])
    entry_ts = pd.to_datetime(row["entry_ts"], utc=True)

    print(f"  [{idx:3d}] {date}  {side:5s}  @{open_t}", end="  →  ")

    result = detect_flem_pivot(side, entry_ts, df_1m, verbose=False)

    if result:
        # Convert bar timestamps (UTC) to local ISO for storage
        flem_iso  = ts_to_iso(result["flem_time"])
        pivot_iso = ts_to_iso(result["pivot_time"])

        df.at[idx, "FLEM Price"]  = result["flem_price"]
        df.at[idx, "FLEM Time"]   = flem_iso
        df.at[idx, "Pivot Price"] = result["pivot_price"]
        df.at[idx, "Pivot Time"]  = pivot_iso

        print(f"✓  FLEM={result['flem_price']} @{flem_iso.split(' ')[1][:5]}  "
              f"Pivot={result['pivot_price']} @{pivot_iso.split(' ')[1][:5]}")
        populated += 1
    else:
        # Retry with verbose output to understand why
        print("✗  (running verbose replay …)")
        detect_flem_pivot(side, entry_ts, df_1m, verbose=True)
        still_missing.append(idx)


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Populated:      {populated} / {missing_mask.sum()}")
print(f"Still missing:  {len(still_missing)}")
if still_missing:
    print(f"  Indices: {still_missing}")
    print("  (These may be trades where breakout/retest didn't occur before entry)")

# ── Save ──────────────────────────────────────────────────────────────────────

df.to_csv(OUTPUT, index=False)
print(f"\n✓ Saved updated trades_combined.csv  ({len(df)} rows)")
print("\nNext steps:")
print("  python3 build_feature_matrix.py")
print("  python3 train_models.py")
