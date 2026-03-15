#!/usr/bin/env python3
"""
export_to_generic.py
--------------------
Exports engine backtest trades to the generic.csv brokerage import format.

Sources:
  output_bnr_det_2025_allow_counter.csv  (451 rows, full 2025)
  output_bnr_det_2026_janfeb.csv         (49 rows, Jan-Feb 2026)

Each backtest row produces TWO output rows (entry leg + exit leg):
  Long : entry=Buy,  exit=Sell
  Short: entry=Sell, exit=Buy

Scale-out trades (same entry_time, multiple rows) are kept as separate legs,
each with their own contract count.

MNQ rolling expiration:
  Quarterly contracts: Mar, Jun, Sep, Dec
  Expiration date = 3rd Friday of the expiration month
  Rollover date   = 8 calendar days before expiration (Thursday before)
  Active contract = earliest quarterly expiry whose rollover_date >= trade_date
"""

import pandas as pd
from datetime import date, timedelta

# ── Paths ─────────────────────────────────────────────────────────────────────

BACKTEST_FILES = [
    "/Users/radhikaarora/Documents/Trading ML/ML V2/output_bnr_det_2025_allow_counter.csv",
    "/Users/radhikaarora/Documents/Trading ML/ML V2/output_bnr_det_2026_janfeb.csv",
]
OUTPUT_CSV = "/Users/radhikaarora/Documents/Trading ML/ML V2/trades_generic_export.csv"

COMMISSION_PER_CONTRACT = 0.65


# ── MNQ rolling expiration helpers ───────────────────────────────────────────

def third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given year/month."""
    d = date(year, month, 1)
    days_to_first_fri = (4 - d.weekday()) % 7
    first_fri = d + timedelta(days=days_to_first_fri)
    return first_fri + timedelta(weeks=2)


def mnq_expiry_for_date(trade_date: date) -> str:
    """
    Return the active MNQ contract label (e.g. 'Mar 25') for a trade date.
    Active contract = first quarterly expiry whose rollover_date >= trade_date,
    where rollover_date = expiry - 8 days.
    """
    QUARTERLY_MONTHS = [3, 6, 9, 12]
    MONTH_ABBR = {3:"Mar", 6:"Jun", 9:"Sep", 12:"Dec"}

    year = trade_date.year - 1
    candidates = []
    for _ in range(6):
        for m in QUARTERLY_MONTHS:
            expiry   = third_friday(year, m)
            rollover = expiry - timedelta(days=8)
            candidates.append((rollover, expiry, year, m))
        year += 1

    for rollover, expiry, yr, mo in sorted(candidates):
        if rollover >= trade_date:
            return f"{MONTH_ABBR[mo]} {str(yr)[-2:]}"

    raise ValueError(f"Could not determine MNQ contract for {trade_date}")


# ── Load and combine backtest files ───────────────────────────────────────────

print("Loading backtest files …")
dfs = []
for path in BACKTEST_FILES:
    df = pd.read_csv(path)
    print(f"  {path.split('/')[-1]}: {len(df)} rows")
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df = df.sort_values("entry_time").reset_index(drop=True)
print(f"  Total: {len(df)} rows\n")


# ── Build generic.csv rows ────────────────────────────────────────────────────

def parse_time(ts_str: str) -> str:
    """'2025-01-03 09:57:00-05:00' → '09:57:00'"""
    return pd.Timestamp(ts_str).strftime("%H:%M:%S")

def parse_date(ts_str: str) -> str:
    """'2025-01-03 09:57:00-05:00' → '01/03/25'"""
    return pd.Timestamp(ts_str).strftime("%m/%d/%y")

def trade_date_obj(ts_str: str) -> date:
    return pd.Timestamp(ts_str).date()


rows = []

# Group by trade identity so scale-outs share one entry row
for (entry_ts, entry_px, direction), group in df.groupby(
        ["entry_time", "entry_price", "direction"], sort=False):

    group      = group.sort_values("exit_time")
    entry_ts   = str(entry_ts)
    direction  = str(direction).strip().lower()
    entry_px   = float(entry_px)
    total_qty  = int(group["contracts"].sum())

    d          = trade_date_obj(entry_ts)
    date_fmt   = parse_date(entry_ts)
    expiry     = mnq_expiry_for_date(d)

    entry_side = "Buy"  if direction == "long"  else "Sell"
    exit_side  = "Sell" if direction == "long"  else "Buy"

    # Single entry row with total contracts
    rows.append({
        "Date":       date_fmt,
        "Time":       parse_time(entry_ts),
        "Symbol":     "MNQ",
        "Buy/Sell":   entry_side,
        "Quantity":   total_qty,
        "Price":      entry_px,
        "Spread":     "Future",
        "Expiration": expiry,
        "Strike":     "",
        "Call/Put":   "",
        "Commission": round(COMMISSION_PER_CONTRACT * total_qty, 4),
        "Fees":       "",
    })

    # One exit row per leg
    for _, leg in group.iterrows():
        exit_ts = str(leg["exit_time"])
        qty     = int(leg["contracts"])
        exit_px = float(leg["exit_price"])
        rows.append({
            "Date":       parse_date(exit_ts),
            "Time":       parse_time(exit_ts),
            "Symbol":     "MNQ",
            "Buy/Sell":   exit_side,
            "Quantity":   qty,
            "Price":      exit_px,
            "Spread":     "Future",
            "Expiration": expiry,
            "Strike":     "",
            "Call/Put":   "",
            "Commission": round(COMMISSION_PER_CONTRACT * qty, 4),
            "Fees":       "",
        })


# ── Save ──────────────────────────────────────────────────────────────────────

out = pd.DataFrame(rows, columns=[
    "Date", "Time", "Symbol", "Buy/Sell", "Quantity", "Price",
    "Spread", "Expiration", "Strike", "Call/Put", "Commission", "Fees",
])

out.to_csv(OUTPUT_CSV, index=False)
print(f"✓ Exported {len(df)} trades ({len(out)} rows) → {OUTPUT_CSV.split('/')[-1]}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\nDate range:", out["Date"].iloc[0], "→", out["Date"].iloc[-1])
print("\nExpiration breakdown (trades):")
# one row per trade = entry row
entries = out[out["Buy/Sell"].isin(["Buy", "Sell"])].iloc[::2]
print(entries.groupby("Expiration").size().to_string())

print("\nSample (first 8 rows):")
print(out.head(8).to_string(index=False))
