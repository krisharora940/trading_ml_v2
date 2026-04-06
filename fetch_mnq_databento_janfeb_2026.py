#!/usr/bin/env python3
"""Fetch Databento OHLCV data for MNQ, Jan-Feb 2026, 9:30 AM-1:00 PM ET.

Produces:
  data/mnq_1m_2026-01-01_to_2026-02-28.parquet
  data/mnq_30s_2026-01-01_to_2026-02-28.parquet

30-second bars are resampled from true 1-second Databento data.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import databento as db
import exchange_calendars as xcals
import pandas as pd

DATASET = "GLBX.MDP3"
SYMBOL = "MNQ.c.0"
STYPE_IN = "continuous"

ET = ZoneInfo("America/New_York")

SESSION_START = dtime(9, 30)
SESSION_END = dtime(13, 0)

START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 2, 28)

OUT_DIR = Path("data")
FETCH_1M = False
FETCH_1S = True


def open_trading_days() -> list[date]:
    cal = xcals.get_calendar("XNYS")
    sessions = cal.sessions_in_range(START_DATE, END_DATE)
    return [s.date() for s in sessions]


def fetch_per_day(client: db.Historical, schema: str, days: list[date]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in days:
        start_et = datetime.combine(day, SESSION_START, tzinfo=ET)
        end_et = datetime.combine(day, SESSION_END, tzinfo=ET)
        print(f"  {day} [{start_et.strftime('%H:%M')}–{end_et.strftime('%H:%M')} ET]", end="  ")
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[SYMBOL],
            schema=schema,
            start=start_et,
            end=end_et,
            stype_in=STYPE_IN,
        )
        df = data.to_df()
        print(f"{len(df):>6} rows")
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames).sort_index()
    combined.index = combined.index.tz_convert("America/New_York")
    return combined


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {k: v for k, v in {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }.items() if k in df.columns}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])


def main() -> None:
    api_key = os.getenv("DATABENTO_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DATABENTO_API_KEY not set.")

    client = db.Historical(key=api_key)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    days = open_trading_days()
    print(f"Open trading days in {START_DATE}–{END_DATE} ({len(days)}):")
    print(days)
    print()

    if FETCH_1M:
        print("Fetching 1-minute bars...")
        df_1m = fetch_per_day(client, "ohlcv-1m", days)
        out_1m = OUT_DIR / f"mnq_1m_{START_DATE}_to_{END_DATE}.parquet"
        df_1m.to_parquet(out_1m)
        print(f"Saved 1m -> {out_1m} ({len(df_1m):,} rows)\n")

    if FETCH_1S:
        print("Fetching 1-second bars for 30-second resample...")
        df_1s = fetch_per_day(client, "ohlcv-1s", days)
        df_30s = resample_ohlcv(df_1s, "30s")
        out_30s = OUT_DIR / f"mnq_30s_{START_DATE}_to_{END_DATE}.parquet"
        df_30s.to_parquet(out_30s)
        print(f"Saved 30s -> {out_30s} ({len(df_30s):,} rows)\n")


if __name__ == "__main__":
    main()
