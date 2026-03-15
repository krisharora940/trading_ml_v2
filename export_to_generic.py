#!/usr/bin/env python3
"""
export_to_generic.py
--------------------
Exports trades_combined.csv to the generic.csv brokerage import format.

Each trade produces TWO rows (entry + exit):
  Long : entry=Buy,  exit=Sell
  Short: entry=Sell, exit=Buy

MNQ rolling expiration:
  Quarterly contracts: Mar, Jun, Sep, Dec
  Expiration date = 3rd Friday of the expiration month
  Rollover date   = Thursday before expiration (8 calendar days before)
  Active contract = earliest quarterly expiry whose rollover_date >= trade_date
"""

import pandas as pd
from datetime import date, timedelta

# ── Paths ─────────────────────────────────────────────────────────────────────

TRADES_CSV = "/Users/radhikaarora/Documents/Trading ML/ML V2/trades_combined.csv"
OUTPUT_CSV = "/Users/radhikaarora/Documents/Trading ML/ML V2/trades_generic_export.csv"

COMMISSION_PER_CONTRACT = 0.65   # $ per contract per leg


# ── MNQ rolling expiration helpers ───────────────────────────────────────────

def third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of the given year/month."""
    # Find first day of month, then first Friday, then +14 days
    d = date(year, month, 1)
    # weekday(): Mon=0 … Fri=4 … Sun=6
    days_to_first_fri = (4 - d.weekday()) % 7
    first_fri = d + timedelta(days=days_to_first_fri)
    return first_fri + timedelta(weeks=2)


def mnq_expiry_for_date(trade_date: date) -> str:
    """
    Return the active MNQ contract expiry label (e.g. 'Mar 25') for a given trade date.

    Logic:
      - MNQ trades quarterly: Mar, Jun, Sep, Dec
      - Contract expires on the 3rd Friday of the expiry month
      - Rollover to next contract on the Thursday before expiry
        (rollover_date = expiry_date - 8 days)
      - Active contract = the first quarterly expiry whose rollover_date >= trade_date
    """
    QUARTERLY_MONTHS = [3, 6, 9, 12]   # Mar, Jun, Sep, Dec
    MONTH_ABBR = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
                  7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}

    # Check 8 consecutive quarterly contracts starting from 2 years back
    year = trade_date.year - 1
    candidates = []
    for _ in range(12):   # enough quarters
        for m in QUARTERLY_MONTHS:
            expiry   = third_friday(year, m)
            rollover = expiry - timedelta(days=8)   # Thursday before
            candidates.append((expiry, rollover, year, m))
        year += 1

    for expiry, rollover, yr, mo in sorted(candidates):
        if rollover >= trade_date:
            yy = str(yr)[-2:]
            return f"{MONTH_ABBR[mo]} {yy}"

    raise ValueError(f"Could not determine MNQ contract for {trade_date}")


# ── Load trades ───────────────────────────────────────────────────────────────

print("Loading trades …")
df = pd.read_csv(TRADES_CSV)
print(f"  {len(df)} rows")


# ── Build output rows ─────────────────────────────────────────────────────────

def strip_tz(time_str: str) -> str:
    """'09:38:00 EST' → '09:38:00'"""
    return str(time_str).strip().split()[0]


def fmt_date(date_str: str) -> str:
    """'2025-01-03' → '01/03/25'"""
    d = pd.Timestamp(date_str)
    return d.strftime("%m/%d/%y")


rows = []

for _, trade in df.iterrows():
    open_date  = str(trade["Open Date"]).strip()
    open_time  = strip_tz(trade["Open Time"])
    close_time = strip_tz(trade["Close Time"])
    side       = str(trade["Side"]).strip().lower()
    qty        = int(trade["Executions"])
    entry_px   = float(trade["Entry Price"])
    exit_px    = float(trade["Exit Price"])
    symbol     = str(trade["Symbol"]).strip()

    trade_date = pd.Timestamp(open_date).date()
    date_fmt   = fmt_date(open_date)
    expiry     = mnq_expiry_for_date(trade_date)
    commission = round(COMMISSION_PER_CONTRACT * qty, 4)

    # Entry / exit buy-sell labels
    if side == "long":
        entry_side = "Buy"
        exit_side  = "Sell"
    else:
        entry_side = "Sell"
        exit_side  = "Buy"

    # Entry row
    rows.append({
        "Date":        date_fmt,
        "Time":        open_time,
        "Symbol":      symbol,
        "Buy/Sell":    entry_side,
        "Quantity":    qty,
        "Price":       entry_px,
        "Spread":      "Future",
        "Expiration":  expiry,
        "Strike":      "",
        "Call/Put":    "",
        "Commission":  commission,
        "Fees":        "",
    })

    # Exit row
    rows.append({
        "Date":        date_fmt,
        "Time":        close_time,
        "Symbol":      symbol,
        "Buy/Sell":    exit_side,
        "Quantity":    qty,
        "Price":       exit_px,
        "Spread":      "Future",
        "Expiration":  expiry,
        "Strike":      "",
        "Call/Put":    "",
        "Commission":  commission,
        "Fees":        "",
    })


# ── Save ──────────────────────────────────────────────────────────────────────

out = pd.DataFrame(rows, columns=[
    "Date", "Time", "Symbol", "Buy/Sell", "Quantity", "Price",
    "Spread", "Expiration", "Strike", "Call/Put", "Commission", "Fees",
])

out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Exported {len(df)} trades ({len(out)} rows) → trades_generic_export.csv")

# ── Spot-check ────────────────────────────────────────────────────────────────

print("\nFirst 10 rows:")
print(out.head(10).to_string(index=False))

# Verify expiration boundaries
print("\nExpiration summary:")
print(out.groupby("Expiration")["Date"].agg(["first", "last", "count"]).to_string())
