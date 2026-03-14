import pandas as pd
from dataclasses import dataclass
from typing import Optional, List
from datetime import timedelta

ONE_MIN = pd.Timedelta(minutes=1)
THIRTY_SEC = pd.Timedelta(seconds=30)

@dataclass
class Trade:
    day: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    pnl: Optional[float]
    contracts: Optional[int]
    risk_dollars: Optional[float]
    outcome: Optional[str]
    exit_reason: Optional[str]
    pivot: float
    flem: float
    flem_saved_time: Optional[pd.Timestamp]
    reentry_time: Optional[pd.Timestamp]
    pivot_time: Optional[pd.Timestamp]
    risk: float
    target: float
    stop_time: Optional[pd.Timestamp]
    stop_price: Optional[float]
    retrace_at_entry: float
    strong_count_recent3: int
    very_strong: bool


def compute_atr(df_1m: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df_1m['high']
    low = df_1m['low']
    close = df_1m['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr


def run_engine(df_1m: pd.DataFrame, df_30s: pd.DataFrame, allow_counter_candle_entry: bool) -> List[Trade]:
    trades: List[Trade] = []

    # Add event_time at close
    df_1m = df_1m.copy()
    df_30s = df_30s.copy()
    df_1m['event_time'] = df_1m['timestamp'] + ONE_MIN
    df_30s['event_time'] = df_30s['timestamp'] + THIRTY_SEC

    # Precompute ATR on 1m
    df_1m['atr'] = compute_atr(df_1m)

    # Build per-day grouping on local date (timestamp already tz-aware)
    df_1m['day'] = df_1m['timestamp'].dt.date
    df_30s['day'] = df_30s['timestamp'].dt.date

    for day, day_1m in df_1m.groupby('day'):
        day_30s = df_30s[df_30s['day'] == day]
        if day_30s.empty or day_1m.empty:
            continue

        # Focus session 09:30-12:00 local
        session_start = pd.Timestamp(f"{day} 09:30:00", tz=day_1m['timestamp'].dt.tz)
        session_end = pd.Timestamp(f"{day} 12:00:00", tz=day_1m['timestamp'].dt.tz)
        day_1m = day_1m[(day_1m['timestamp'] >= session_start) & (day_1m['timestamp'] <= session_end)].copy()
        day_30s = day_30s[(day_30s['timestamp'] >= session_start) & (day_30s['timestamp'] <= session_end)].copy()
        if day_1m.empty or day_30s.empty:
            continue

        # For quick lookup of 30s bars in a 1m window
        day_30s_sorted = day_30s.sort_values('timestamp')

        def first_30s_target_hit(bar_open_time: pd.Timestamp, direction: str, target: float) -> Optional[pd.Timestamp]:
            window = day_30s_sorted[
                (day_30s_sorted['timestamp'] >= bar_open_time) &
                (day_30s_sorted['timestamp'] < bar_open_time + ONE_MIN)
            ]
            if window.empty:
                return None
            for _, r in window.iterrows():
                if direction == 'long':
                    if r['high'] >= target:
                        return r['event_time']
                else:
                    if r['low'] <= target:
                        return r['event_time']
            return None

        # Zone candle is 09:30 1m bar (known at 09:31)
        zone_row = day_1m[day_1m['timestamp'] == session_start]
        if zone_row.empty:
            continue
        zone_high = float(zone_row.iloc[0]['high'])
        zone_low = float(zone_row.iloc[0]['low'])
        zone_known_time = zone_row.iloc[0]['event_time']

        # Event stream
        ev_1m = day_1m.assign(kind='1m')
        ev_30s = day_30s.assign(kind='30s')
        events = pd.concat([
            ev_1m[['event_time', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'atr', 'kind']],
            ev_30s[['event_time', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'kind']]
        ], ignore_index=True)
        # Ensure 1m closes are processed before 30s events at the same event_time
        events['kind_order'] = events['kind'].map({'1m': 0, '30s': 1}).fillna(2)
        events = events.sort_values(['event_time', 'kind_order'])

        # State
        direction = None
        candidate_active = False
        reentry_seen = False
        reentry_time = None
        flem = None
        flem_saved_time = None
        pivot = None
        pivot_time = None
        entry_triggered = False
        entry_time = None
        entry_price = None
        last_3_strong = []

        # For trades-in-progress
        in_trade = False
        stop_price = None
        target_price = None

        for _, ev in events.iterrows():
            t = ev['event_time']
            if t < zone_known_time:
                continue

            # Invalidation on 30s close
            if ev['kind'] == '30s' and candidate_active and not in_trade:
                close = ev['close']
                if direction == 'long' and close < zone_low:
                    # invalidation
                    direction = None
                    candidate_active = False
                    reentry_seen = False
                    flem = None
                    pivot = None
                    entry_triggered = False
                    last_3_strong = []
                    continue
                if direction == 'short' and close > zone_high:
                    direction = None
                    candidate_active = False
                    reentry_seen = False
                    flem = None
                    pivot = None
                    entry_triggered = False
                    last_3_strong = []
                    continue

            if ev['kind'] == '1m':
                close = ev['close']
                high = ev['high']
                low = ev['low']
                atr = ev['atr']

                # Direction flip on 1m close through other side
                if candidate_active and direction == 'long' and close < zone_low:
                    if in_trade and trades and trades[-1].outcome is None:
                        current = trades[-1]
                        current.outcome = 'loss'
                        current.exit_time = t
                        current.exit_price = float(close)
                        current.stop_time = t
                        current.stop_price = float(stop_price) if stop_price is not None else None
                        current.pnl = float(current.exit_price - current.entry_price)
                        current.exit_reason = 'direction_flip'
                        in_trade = False
                    direction = 'short'
                    candidate_active = True
                    reentry_seen = False
                    flem = None
                    pivot = None
                    entry_triggered = False
                    last_3_strong = []
                    stop_price = None
                    target_price = None

                elif candidate_active and direction == 'short' and close > zone_high:
                    if in_trade and trades and trades[-1].outcome is None:
                        current = trades[-1]
                        current.outcome = 'loss'
                        current.exit_time = t
                        current.exit_price = float(close)
                        current.stop_time = t
                        current.stop_price = float(stop_price) if stop_price is not None else None
                        current.pnl = float(current.entry_price - current.exit_price)
                        current.exit_reason = 'direction_flip'
                        in_trade = False
                    direction = 'long'
                    candidate_active = True
                    reentry_seen = False
                    flem = None
                    pivot = None
                    entry_triggered = False
                    last_3_strong = []
                    stop_price = None
                    target_price = None

                # Break detection if not active
                if not candidate_active:
                    if close > zone_high:
                        direction = 'long'
                        candidate_active = True
                    elif close < zone_low:
                        direction = 'short'
                        candidate_active = True
                    else:
                        continue

                # Track FLEM until a valid retest close into the zone
                if candidate_active and not reentry_seen:
                    if direction == 'long':
                        flem = high if flem is None else max(flem, high)
                        # Retest: candle closes opposite direction and touches/enters zone
                        if close < ev['open'] and low <= zone_high and high >= zone_low:
                            reentry_seen = True
                            reentry_time = t
                            flem_saved_time = t
                            pivot = low
                            pivot_time = t
                    else:
                        flem = low if flem is None else min(flem, low)
                        if close > ev['open'] and low <= zone_high and high >= zone_low:
                            reentry_seen = True
                            reentry_time = t
                            flem_saved_time = t
                            pivot = high
                            pivot_time = t

                # After reentry: update pivot until entry
                if reentry_seen and not entry_triggered:
                    # If retrace violates the far side of the zone before entry, invalidate candidate
                    if direction == 'long' and low < zone_low:
                        direction = None
                        candidate_active = False
                        reentry_seen = False
                        reentry_time = None
                        flem = None
                        flem_saved_time = None
                        pivot = None
                        pivot_time = None
                        entry_triggered = False
                        last_3_strong = []
                        continue
                    if direction == 'short' and high > zone_high:
                        direction = None
                        candidate_active = False
                        reentry_seen = False
                        reentry_time = None
                        flem = None
                        flem_saved_time = None
                        pivot = None
                        pivot_time = None
                        entry_triggered = False
                        last_3_strong = []
                        continue

                    if direction == 'long':
                        if pivot is None or low < pivot:
                            pivot = low
                            pivot_time = t
                    else:
                        if pivot is None or high > pivot:
                            pivot = high
                            pivot_time = t

                    # Strong candle logic
                    body = abs(close - ev['open'])
                    strong = body >= 0.8 * atr
                    very_strong = body >= 1.3 * atr
                    if not allow_counter_candle_entry:
                        if direction == 'long' and close <= ev['open']:
                            strong = False
                            very_strong = False
                        if direction == 'short' and close >= ev['open']:
                            strong = False
                            very_strong = False
                    last_3_strong.append(bool(strong))
                    last_3_strong = last_3_strong[-3:]
                    strong_count = sum(last_3_strong)

                    # Retracement measure
                    retrace = None
                    if flem is not None and pivot is not None and flem != pivot:
                        if direction == 'long':
                            retrace = (close - pivot) / (flem - pivot)
                        else:
                            retrace = (pivot - close) / (pivot - flem)

                    # Entry condition
                    if retrace is not None and retrace >= 0.35 and retrace <= 1.2:
                        if strong_count >= 2 or very_strong:
                            entry_triggered = True
                            in_trade = True
                            entry_time = t
                            entry_price = close
                            if direction == 'long':
                                stop_price = pivot
                                risk = entry_price - stop_price
                                target_price = entry_price + 1.2 * risk
                            else:
                                stop_price = pivot
                                risk = stop_price - entry_price
                                target_price = entry_price - 1.2 * risk

                            # Position sizing: max $500 risk, MNQ = $2/point
                            risk_dollars_per_contract = risk * 2.0
                            contracts = int(500 // risk_dollars_per_contract) if risk_dollars_per_contract > 0 else 0
                            if contracts < 1:
                                # Skip trade if risk > $500
                                entry_triggered = False
                                in_trade = False
                                stop_price = None
                                target_price = None
                                continue
                            trades.append(Trade(
                                day=str(day),
                                direction=direction,
                                entry_time=entry_time,
                                entry_price=float(entry_price),
                                exit_time=None,
                                exit_price=None,
                                pnl=None,
                                contracts=contracts,
                                risk_dollars=float(risk_dollars_per_contract * contracts),
                                outcome=None,
                                exit_reason=None,
                                pivot=float(pivot),
                                flem=float(flem),
                                flem_saved_time=flem_saved_time,
                                reentry_time=reentry_time,
                                pivot_time=pivot_time,
                                risk=float(risk),
                                target=float(target_price),
                                stop_time=None,
                                stop_price=None,
                                retrace_at_entry=float(retrace),
                                strong_count_recent3=strong_count,
                                very_strong=bool(very_strong)
                            ))

                # Manage trade (stop/target) using 1m bars
                if in_trade and trades:
                    current = trades[-1]
                    if current.outcome is None:
                        # Prevent same-bar exits that would rely on earlier 30s data
                        if current.entry_time == t:
                            continue
                        # Dynamic stop/target updates (based on 1m bars)
                        entry_price = current.entry_price
                        entry_risk = current.risk
                        elapsed = t - current.entry_time

                        # If trade has run at least 30 minutes:
                        # - if above BE, move stop to BE
                        # - if below BE, move target to BE
                        if elapsed >= pd.Timedelta(minutes=30):
                            if direction == 'long':
                                if close > entry_price:
                                    stop_price = max(stop_price, entry_price)
                                else:
                                    if target_price > entry_price:
                                        target_price = entry_price
                            else:
                                if close < entry_price:
                                    stop_price = min(stop_price, entry_price)
                                else:
                                    if target_price < entry_price:
                                        target_price = entry_price

                        # If trade has run up at least 1R, move stop to -0.5R
                        if direction == 'long':
                            if high >= entry_price + entry_risk:
                                stop_price = max(stop_price, entry_price - 0.5 * entry_risk)
                        else:
                            if low <= entry_price - entry_risk:
                                stop_price = min(stop_price, entry_price + 0.5 * entry_risk)

                        if direction == 'long':
                            # Stop is based on 1m close below pivot
                            if close <= stop_price:
                                current.outcome = 'loss'
                                current.exit_time = t
                                current.exit_price = float(close)
                                current.pnl = float(current.exit_price - current.entry_price) * float(current.contracts or 0)
                                current.stop_time = t
                                current.stop_price = float(stop_price)
                                current.exit_reason = 'stop'
                                in_trade = False
                            elif high >= target_price:
                                current.outcome = 'win'
                                hit_time = first_30s_target_hit(ev['timestamp'], direction, target_price)
                                current.exit_time = hit_time if hit_time is not None else t
                                current.exit_price = float(target_price)
                                current.pnl = float(current.exit_price - current.entry_price) * float(current.contracts or 0)
                                current.exit_reason = 'target'
                                in_trade = False
                        else:
                            # Stop is based on 1m close above pivot
                            if close >= stop_price:
                                current.outcome = 'loss'
                                current.exit_time = t
                                current.exit_price = float(close)
                                current.pnl = float(current.entry_price - current.exit_price) * float(current.contracts or 0)
                                current.stop_time = t
                                current.stop_price = float(stop_price)
                                current.exit_reason = 'stop'
                                in_trade = False
                            elif low <= target_price:
                                current.outcome = 'win'
                                hit_time = first_30s_target_hit(ev['timestamp'], direction, target_price)
                                current.exit_time = hit_time if hit_time is not None else t
                                current.exit_price = float(target_price)
                                current.pnl = float(current.entry_price - current.exit_price) * float(current.contracts or 0)
                                current.exit_reason = 'target'
                                in_trade = False

        # End of day cleanup: force-close any unclosed trade at session end
        if trades and trades[-1].outcome is None:
            last_bar = day_1m[day_1m['timestamp'] <= session_end].iloc[-1]
            t = last_bar['event_time']
            close = float(last_bar['close'])
            current = trades[-1]
            current.outcome = 'forced_close'
            current.exit_time = t
            current.exit_price = close
            if current.direction == 'long':
                current.pnl = float(close - current.entry_price) * float(current.contracts or 0)
            else:
                current.pnl = float(current.entry_price - close) * float(current.contracts or 0)
            current.exit_reason = 'forced_close'
            in_trade = False

    return trades


def main():
    p1 = "/Users/radhikaarora/Documents/New Project/Input Data/market/mnq_1m.csv"
    p2 = "/Users/radhikaarora/Documents/New Project/Input Data/market/mnq_30s.csv"

    df_1m = pd.read_csv(p1)
    df_30s = pd.read_csv(p2)

    # Parse timestamps
    # Normalize to a single timezone
    df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], utc=True).dt.tz_convert("America/New_York")
    df_30s['timestamp'] = pd.to_datetime(df_30s['timestamp'], utc=True).dt.tz_convert("America/New_York")

    ranges = [
        ("2025-01-01", "2025-12-31"),
    ]

    def run_for_ranges(allow_counter: bool) -> pd.DataFrame:
        all_trades = []
        for start_s, end_s in ranges:
            start = pd.Timestamp(start_s)
            end = pd.Timestamp(end_s)
            slice_1m = df_1m[(df_1m['timestamp'].dt.date >= start.date()) & (df_1m['timestamp'].dt.date <= end.date())]
            slice_30s = df_30s[(df_30s['timestamp'].dt.date >= start.date()) & (df_30s['timestamp'].dt.date <= end.date())]
            all_trades.extend(run_engine(slice_1m, slice_30s, allow_counter_candle_entry=allow_counter))
        return pd.DataFrame([t.__dict__ for t in all_trades])

    out_allow = run_for_ranges(True)
    out_allow_path = "/Users/radhikaarora/Documents/Trading ML/ML V2/output_bnr_det_2025_allow_counter.csv"
    out_allow.to_csv(out_allow_path, index=False)
    print(f"Wrote {len(out_allow)} trades to {out_allow_path}")

    out_no = run_for_ranges(False)
    out_no_path = "/Users/radhikaarora/Documents/Trading ML/ML V2/output_bnr_det_2025_no_counter.csv"
    out_no.to_csv(out_no_path, index=False)
    print(f"Wrote {len(out_no)} trades to {out_no_path}")


if __name__ == "__main__":
    main()
