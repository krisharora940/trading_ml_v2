#!/usr/bin/env python3
"""Run monthly OOS backtests with exclusionary retraining per month.

For each test month:
  1) Retrain pwin model excluding that month.
  2) Run BNR deterministic engine on that month using the retrained model.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import List

import pandas as pd
import importlib.util

BASE = "/Users/radhikaarora/Documents/Trading ML/ML V2"
RETRAIN_SCRIPT = os.path.join(BASE, "experiments/hodlod_entry_features/retrain_pwin_15features_hodlod_baseline_nogates.py")
ENGINE_PATH = os.path.join(BASE, "experiments/hodlod_entry_features/bnr_deterministic_engine_13feat_hodlod.py")

DATA_DIR_QUARTERLY = "/Users/radhikaarora/Documents/New Project/output/market/quarterly"
DATA_DIR_OUTPUT = "/Users/radhikaarora/Documents/New Project/output/market"
DATA_DIR_INPUT = "/Users/radhikaarora/Documents/New Project/Input Data/market"

P1_2025_Q = [
    os.path.join(DATA_DIR_QUARTERLY, "mnq_1m_2025_q1.csv"),
    os.path.join(DATA_DIR_QUARTERLY, "mnq_1m_2025_q2.csv"),
    os.path.join(DATA_DIR_QUARTERLY, "mnq_1m_2025_q3.csv"),
    os.path.join(DATA_DIR_QUARTERLY, "mnq_1m_2025_q4.csv"),
]
P30_2025_Q = [
    os.path.join(DATA_DIR_QUARTERLY, "mnq_30s_2025_q1.csv"),
    os.path.join(DATA_DIR_QUARTERLY, "mnq_30s_2025_q2.csv"),
    os.path.join(DATA_DIR_QUARTERLY, "mnq_30s_2025_q3.csv"),
    os.path.join(DATA_DIR_QUARTERLY, "mnq_30s_2025_q4.csv"),
]

P1_2026_JANFEB = os.path.join(DATA_DIR_INPUT, "mnq_1m_jan_feb_2026.csv")
P30_2026_Q1 = os.path.join(DATA_DIR_QUARTERLY, "mnq_30s_2026_q1.csv")
P1_2026_MAR = os.path.join(DATA_DIR_OUTPUT, "mnq_1m_mar26.csv")
P30_2026_MAR = os.path.join(DATA_DIR_OUTPUT, "mnq_30s_mar26.csv")

ET = "America/New_York"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly OOS retrain + backtest (baseline HOD/LOD no-gates).")
    p.add_argument("--months", default="", help="Comma-separated YYYY-MM list. If empty, use --start/--end.")
    p.add_argument("--start", default="2025-01", help="YYYY-MM")
    p.add_argument("--end", default="2026-03", help="YYYY-MM")
    p.add_argument("--pwin", type=float, default=0.40)
    p.add_argument("--out-dir", default=os.path.join(BASE, "outputs/oos_monthly_hodlod_baseline_nogates"))
    p.add_argument("--model-dir", default=os.path.join(BASE, "saved_training_sets/oos_models_hodlod_baseline_nogates"))
    p.add_argument("--backtest-csv", default=os.path.join(BASE, "outputs/output_bnr_det_2025_15feat_baseline_hodlod_scaleoutfix_nogates_pwin50.csv"))
    p.add_argument("--base-features-csv", default="")
    p.add_argument("--tag", default="")
    return p.parse_args()


def month_range(start: str, end: str) -> List[str]:
    start_p = pd.Period(start, freq="M")
    end_p = pd.Period(end, freq="M")
    months = []
    cur = start_p
    while cur <= end_p:
        months.append(str(cur))
        cur += 1
    return months


def load_concat(paths: List[str]) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    return df.drop_duplicates("timestamp").sort_values("timestamp")


def load_for_month(month: str):
    year, mon = month.split("-")
    year = int(year)
    mon = int(mon)
    if year == 2025:
        df_1m = load_concat(P1_2025_Q)
        df_30s = load_concat(P30_2025_Q)
    elif year == 2026 and mon in (1, 2):
        df_1m = load_concat([P1_2026_JANFEB])
        df_30s = load_concat([P30_2026_Q1])
    elif year == 2026 and mon == 3:
        df_1m = load_concat([P1_2026_MAR])
        df_30s = load_concat([P30_2026_MAR])
    else:
        raise ValueError(f"No data source configured for month {month}")

    month_mask_1m = df_1m["timestamp"].dt.to_period("M").astype(str) == month
    month_mask_30 = df_30s["timestamp"].dt.to_period("M").astype(str) == month
    return df_1m[month_mask_1m].reset_index(drop=True), df_30s[month_mask_30].reset_index(drop=True)


def summarize(out_path: str) -> str:
    df = pd.read_csv(out_path)
    if df.empty:
        return "0 trades, $0"
    df["day"] = pd.to_datetime(df["day"])
    grouped = df.groupby(["day", "direction", "entry_time"]).agg(pnl=("pnl", "sum")).reset_index()
    wins = (grouped["pnl"] > 0).sum()
    wr = wins / len(grouped) * 100 if len(grouped) else 0.0
    total_pnl = grouped["pnl"].sum() * 2
    avg_w = grouped[grouped["pnl"] > 0]["pnl"].mean() * 2
    avg_l = grouped[grouped["pnl"] < 0]["pnl"].mean() * 2
    return f"{len(grouped)} trades, {wr:.1f}% WR, ${total_pnl:,.0f}, AvgW ${avg_w:,.0f}, AvgL ${avg_l:,.0f}"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    model_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    months = [m.strip() for m in args.months.split(",") if m.strip()] if args.months else month_range(args.start, args.end)

    spec = importlib.util.spec_from_file_location("engine_hodlod_baseline_nogates", ENGINE_PATH)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)

    for month in months:
        tag = f"_{args.tag}" if args.tag else ""
        model_path = model_dir / f"entry_model_pwin_15features_hodlod_baseline_nogates_retrained{tag}_excl_{month}.joblib"
        feat_path = model_dir / f"ml_features_15feat_hodlod_baseline_nogates_retrained{tag}_excl_{month}.csv"

        # Retrain excluding this month
        cmd = [
            "python3",
            RETRAIN_SCRIPT,
            "--backtest-csv", args.backtest_csv,
            "--exclude-month", month,
            "--out-model", str(model_path),
            "--out-csv", str(feat_path),
        ]
        if args.base_features_csv:
            cmd.extend(["--base-features-csv", args.base_features_csv])
        print(f"\n[Retrain] Excluding {month} ...")
        subprocess.run(cmd, check=True)

        # Run backtest for this month using the retrained model
        os.environ["PWIN_MODEL_PATH"] = str(model_path)
        df_1m, df_30s = load_for_month(month)
        trades = engine.run_engine(df_1m, df_30s, allow_counter_candle_entry=True, pwin_thresh=args.pwin)
        out_path = out_dir / f"output_bnr_det_{month.replace('-', '_')}_15feat_pwin{int(args.pwin*100):02d}_hodlod_baseline_nogates{tag}_oos.csv"
        pd.DataFrame([t.__dict__ for t in trades]).to_csv(out_path, index=False)
        print(f"[Backtest] {month} -> {out_path}")
        print(f"  {summarize(str(out_path))}")


if __name__ == "__main__":
    main()
