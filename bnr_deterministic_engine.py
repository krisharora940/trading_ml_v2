import pandas as pd
from dataclasses import dataclass
from typing import Optional, List
from datetime import timedelta
import joblib
import os

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

    # Precompute ATR on 1m (day-wide)
    df_1m['atr'] = compute_atr(df_1m)

    # Build per-day grouping on local date (timestamp already tz-aware)
    df_1m['day'] = df_1m['timestamp'].dt.date
    df_30s['day'] = df_30s['timestamp'].dt.date

    # Load ML model for entry validation (if available)
    model_path = "/Users/radhikaarora/Documents/Trading ML/ML V2/entry_model.joblib"
    ml_bundle = joblib.load(model_path) if os.path.exists(model_path) else None
    ml_model = ml_bundle['model'] if ml_bundle else None
    ml_features = ml_bundle['features'] if ml_bundle else None

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
        # ML retrace window accumulators (30s candles since pivot_time)
        body_sum_30s = 0.0
        body_count_30s = 0
        in_dir_count_30s = 0
        max_run_30s = 0
        run_30s = 0
        last_body_30s = None
        last_30s_close = None
        top2_scores = []

        # For trades-in-progress
        in_trade = False
        stop_price = None
        target_price = None
        scale_out_active = False
        scale_out_done = False
        scale_out_first_contracts = 0
        scale_out_second_contracts = 0
        scale_out_second_target_r = None
        daily_pnl_dollars = 0.0

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
                    body_sum_30s = 0.0
                    body_count_30s = 0
                    in_dir_count_30s = 0
                    max_run_30s = 0
                    run_30s = 0
                    last_body_30s = None
                    last_30s_close = None
                    top2_scores = []
                    continue
                if direction == 'short' and close > zone_high:
                    direction = None
                    candidate_active = False
                    reentry_seen = False
                    flem = None
                    pivot = None
                    entry_triggered = False
                    last_3_strong = []
                    body_sum_30s = 0.0
                    body_count_30s = 0
                    in_dir_count_30s = 0
                    max_run_30s = 0
                    run_30s = 0
                    last_body_30s = None
                    last_30s_close = None
                    top2_scores = []
                    continue

            # Update ML retrace window accumulators on 30s bars
            if ev['kind'] == '30s' and reentry_seen and not entry_triggered and pivot_time is not None:
                if ev['timestamp'] >= pivot_time:
                    body = abs(ev['close'] - ev['open'])
                    last_body_30s = body
                    last_30s_close = ev['close']
                    body_sum_30s += body
                    body_count_30s += 1
                    in_dir = (ev['close'] > ev['open']) if direction == 'long' else (ev['close'] < ev['open'])
                    in_dir_count_30s += 1 if in_dir else 0
                    if in_dir:
                        run_30s += 1
                        max_run_30s = max(max_run_30s, run_30s)
                    else:
                        run_30s = 0

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
                        current.pnl = float(current.exit_price - current.entry_price) * float(current.contracts or 0)
                        current.exit_reason = 'direction_flip'
                        in_trade = False
                        daily_pnl_dollars += float(current.pnl) * 2.0
                        scale_out_active = False
                        scale_out_done = False
                    direction = 'short'
                    candidate_active = True
                    reentry_seen = False
                    flem = None
                    pivot = None
                    entry_triggered = False
                    last_3_strong = []
                    stop_price = None
                    target_price = None
                    body_sum_30s = 0.0
                    body_count_30s = 0
                    in_dir_count_30s = 0
                    max_run_30s = 0
                    run_30s = 0
                    last_body_30s = None
                    last_30s_close = None
                    top2_scores = []

                elif candidate_active and direction == 'short' and close > zone_high:
                    if in_trade and trades and trades[-1].outcome is None:
                        current = trades[-1]
                        current.outcome = 'loss'
                        current.exit_time = t
                        current.exit_price = float(close)
                        current.stop_time = t
                        current.stop_price = float(stop_price) if stop_price is not None else None
                        current.pnl = float(current.entry_price - current.exit_price) * float(current.contracts or 0)
                        current.exit_reason = 'direction_flip'
                        in_trade = False
                        daily_pnl_dollars += float(current.pnl) * 2.0
                        scale_out_active = False
                        scale_out_done = False
                    direction = 'long'
                    candidate_active = True
                    reentry_seen = False
                    flem = None
                    pivot = None
                    entry_triggered = False
                    last_3_strong = []
                    stop_price = None
                    target_price = None
                    body_sum_30s = 0.0
                    body_count_30s = 0
                    in_dir_count_30s = 0
                    max_run_30s = 0
                    run_30s = 0
                    last_body_30s = None
                    last_30s_close = None
                    top2_scores = []

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
                        if flem is None or high > flem:
                            flem = high if flem is None else max(flem, high)
                        # Retest: candle closes opposite direction and touches/enters zone
                        if close < ev['open'] and low <= zone_high and high >= zone_low:
                            reentry_seen = True
                            reentry_time = t
                            flem_saved_time = t
                            pivot = low
                            pivot_time = t
                            body_sum_30s = 0.0
                            body_count_30s = 0
                            in_dir_count_30s = 0
                            max_run_30s = 0
                            run_30s = 0
                            last_body_30s = None
                            last_30s_close = None
                            top2_scores = []
                    else:
                        if flem is None or low < flem:
                            flem = low if flem is None else min(flem, low)
                        if close > ev['open'] and low <= zone_high and high >= zone_low:
                            reentry_seen = True
                            reentry_time = t
                            flem_saved_time = t
                            pivot = high
                            pivot_time = t
                            body_sum_30s = 0.0
                            body_count_30s = 0
                            in_dir_count_30s = 0
                            max_run_30s = 0
                            run_30s = 0
                            last_body_30s = None
                            last_30s_close = None
                            top2_scores = []

                # After reentry: update pivot until entry
                if reentry_seen and not entry_triggered:
                    # If retrace exceeds 1.2 before entry, reset setup to seek next re-entry
                    if flem is not None and pivot is not None and flem != pivot:
                        if direction == 'long':
                            retrace_reset = (close - pivot) / (flem - pivot)
                            if retrace_reset > 1.2:
                                flem = high
                                flem_saved_time = None
                                reentry_seen = False
                                reentry_time = None
                                pivot = None
                                pivot_time = None
                                entry_triggered = False
                                last_3_strong = []
                                body_sum_30s = 0.0
                                body_count_30s = 0
                                in_dir_count_30s = 0
                                max_run_30s = 0
                                run_30s = 0
                                last_body_30s = None
                                last_30s_close = None
                                top2_scores = []
                                continue
                        else:
                            retrace_reset = (pivot - close) / (pivot - flem)
                            if retrace_reset > 1.2:
                                flem = low
                                flem_saved_time = None
                                reentry_seen = False
                                reentry_time = None
                                pivot = None
                                pivot_time = None
                                entry_triggered = False
                                last_3_strong = []
                                body_sum_30s = 0.0
                                body_count_30s = 0
                                in_dir_count_30s = 0
                                max_run_30s = 0
                                run_30s = 0
                                last_body_30s = None
                                last_30s_close = None
                                top2_scores = []
                                continue
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
                        body_sum_30s = 0.0
                        body_count_30s = 0
                        in_dir_count_30s = 0
                        max_run_30s = 0
                        run_30s = 0
                        last_body_30s = None
                        last_30s_close = None
                        top2_scores = []
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
                        body_sum_30s = 0.0
                        body_count_30s = 0
                        in_dir_count_30s = 0
                        max_run_30s = 0
                        run_30s = 0
                        last_body_30s = None
                        last_30s_close = None
                        top2_scores = []
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

                    # Entry condition (ML overrides strong/very-strong if model is available)
                    ml_allows_entry = False
                    if ml_model is not None and ml_features is not None and last_30s_close is not None:
                        pivot_flem_dist = abs(pivot - flem) if (pivot is not None and flem is not None) else None
                        time_since_pivot = (t - pivot_time).total_seconds() if pivot_time is not None else None
                        if pivot_flem_dist not in (0, None) and time_since_pivot is not None and body_count_30s > 0:
                            if direction == 'long':
                                retrace_ml = (last_30s_close - pivot) / (flem - pivot) if flem != pivot else None
                            else:
                                retrace_ml = (pivot - last_30s_close) / (pivot - flem) if flem != pivot else None
                            if retrace_ml is not None:
                                body_mean = body_sum_30s / body_count_30s
                                in_dir_ratio = in_dir_count_30s / body_count_30s
                                # 9:30 zone and day range for zone/pivot over range features
                                day_key = t.date()
                                zone_time = pd.Timestamp(f"{day_key} 09:30:00", tz=day_1m['timestamp'].dt.tz)
                                try:
                                    zone_bar = day_1m.loc[day_1m['timestamp'] == zone_time].iloc[0]
                                except Exception:
                                    zone_bar = None
                                day_1m_to_now = day_1m[day_1m['timestamp'] <= t]
                                day_range = float(day_1m_to_now['high'].max() - day_1m_to_now['low'].min()) if not day_1m_to_now.empty else 0.0
                                if direction == 'long':
                                    zone_price = float(zone_bar['high']) if zone_bar is not None else None
                                    dist_zone = (flem - zone_price) if zone_price is not None else None
                                else:
                                    zone_price = float(zone_bar['low']) if zone_bar is not None else None
                                    dist_zone = (zone_price - flem) if zone_price is not None else None
                                zone_over_range = (dist_zone / day_range) if (dist_zone is not None and day_range not in (0.0, None)) else 0.0
                                pivot_over_range = (pivot_flem_dist / day_range) if (day_range not in (0.0, None)) else 0.0

                                features = {
                                    'retrace': retrace_ml,
                                    'pivot_flem_dist': pivot_flem_dist,
                                    'time_since_pivot_sec': time_since_pivot,
                                    'body_last': last_body_30s or 0.0,
                                    'body_sum': body_sum_30s,
                                    'body_mean': body_mean,
                                    'in_dir_ratio': in_dir_ratio,
                                    'max_in_dir_run': max_run_30s,
                                    'bars_since_pivot': body_count_30s,
                                    'zone_over_range': zone_over_range,
                                    'pivot_over_range': pivot_over_range
                                }
                                x = pd.DataFrame([features])[ml_features].fillna(0.0)
                                score = float(ml_model.predict_proba(x)[0][1])
                                top2_scores.append(score)
                                top2_scores = sorted(top2_scores, reverse=True)[:2]
                                ml_allows_entry = score >= min(top2_scores) if len(top2_scores) == 2 else False

                    # Displacement category based on zone/pivot distance over day range
                    day_key = t.date()
                    pivot_flem_dist = abs(flem - pivot)
                    zone_time = pd.Timestamp(f"{day_key} 09:30:00", tz=day_1m['timestamp'].dt.tz)
                    try:
                        zone_bar = day_1m.loc[day_1m['timestamp'] == zone_time].iloc[0]
                    except Exception:
                        zone_bar = None
                    day_1m_to_now = day_1m[day_1m['timestamp'] <= t]
                    day_range = float(day_1m_to_now['high'].max() - day_1m_to_now['low'].min()) if not day_1m_to_now.empty else 0.0
                    if direction == 'long':
                        zone_price = float(zone_bar['high']) if zone_bar is not None else None
                        dist_zone = (flem - zone_price) if zone_price is not None else None
                    else:
                        zone_price = float(zone_bar['low']) if zone_bar is not None else None
                        dist_zone = (zone_price - flem) if zone_price is not None else None
                    zone_over_range = (dist_zone / day_range) if (dist_zone is not None and day_range not in (0.0, None)) else 0.0
                    pivot_over_range = (pivot_flem_dist / day_range) if (day_range not in (0.0, None)) else 0.0

                    if zone_over_range >= 0.10568226033342312 and pivot_over_range >= 0.3795379537953795:
                        displacement_category = 'high'
                    elif zone_over_range <= 0.0848692546366965 and pivot_over_range <= 0.23516193082722905:
                        displacement_category = 'low'
                    else:
                        displacement_category = 'medium'

                    min_retrace = 0.45 if displacement_category == 'high' else 0.35

                    # Require prior 1m candle to close in the direction of the trade
                    prev_1m = day_1m.loc[day_1m['timestamp'] == (t - ONE_MIN)]
                    prev_color_ok = True
                    if not prev_1m.empty:
                        prev_close = float(prev_1m.iloc[0]['close'])
                        prev_open = float(prev_1m.iloc[0]['open'])
                        if direction == 'long':
                            prev_color_ok = prev_close > prev_open
                        else:
                            prev_color_ok = prev_close < prev_open

                    if retrace is not None and retrace >= min_retrace and retrace <= 1.2 and prev_color_ok:
                        if (ml_model is not None and ml_allows_entry) or (ml_model is None and (strong_count >= 2 or very_strong)):
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
                            if contracts <= 2:
                                # Skip trade if position size is too small
                                entry_triggered = False
                                in_trade = False
                                stop_price = None
                                target_price = None
                                continue
                            if contracts >= 29:
                                # Skip trade if position size is too large
                                entry_triggered = False
                                in_trade = False
                                stop_price = None
                                target_price = None
                                continue

                            # Scale-out rule:
                            # - contracts >= 13 => 50% at 1.2R, remainder at 4R
                            # - contracts < 13  => 50% at 1.2R, remainder at 2R
                            scale_out_active = True
                            scale_out_done = False
                            if scale_out_active:
                                scale_out_first_contracts = contracts // 2
                                scale_out_second_contracts = contracts - scale_out_first_contracts
                                if scale_out_first_contracts == 0 or scale_out_second_contracts == 0:
                                    scale_out_active = False
                                    scale_out_first_contracts = 0
                                    scale_out_second_contracts = 0
                            scale_out_second_target_r = None
                            if contracts >= 13:
                                scale_out_second_target_r = 4.0
                            else:
                                scale_out_second_target_r = 2.0
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
                                daily_pnl_dollars += float(current.pnl) * 2.0
                                scale_out_active = False
                                scale_out_done = False
                            elif high >= target_price:
                                # Scale-out: take partial at 1.2R, keep remainder to fixed-R target
                                if scale_out_active and not scale_out_done:
                                    hit_time = first_30s_target_hit(ev['timestamp'], direction, target_price)
                                    first_exit_time = hit_time if hit_time is not None else t
                                    first_trade = Trade(
                                        day=current.day,
                                        direction=current.direction,
                                        entry_time=current.entry_time,
                                        entry_price=current.entry_price,
                                        exit_time=first_exit_time,
                                        exit_price=float(target_price),
                                        pnl=float(target_price - current.entry_price) * float(scale_out_first_contracts),
                                        contracts=scale_out_first_contracts,
                                        risk_dollars=float(current.risk * 2.0 * scale_out_first_contracts),
                                        outcome='win',
                                        exit_reason='target_scale1',
                                        pivot=current.pivot,
                                        flem=current.flem,
                                        flem_saved_time=current.flem_saved_time,
                                        reentry_time=current.reentry_time,
                                        pivot_time=current.pivot_time,
                                        risk=current.risk,
                                        target=current.target,
                                        stop_time=None,
                                        stop_price=current.stop_price,
                                        retrace_at_entry=current.retrace_at_entry,
                                        strong_count_recent3=current.strong_count_recent3,
                                        very_strong=current.very_strong
                                    )
                                    trades[-1] = first_trade
                                    daily_pnl_dollars += float(first_trade.pnl) * 2.0

                                    # Continue with remaining size toward fixed-R target
                                    scale_out_done = True
                                    target_price = current.entry_price + float(scale_out_second_target_r or 4.0) * current.risk
                                    remaining_trade = Trade(
                                        day=current.day,
                                        direction=current.direction,
                                        entry_time=current.entry_time,
                                        entry_price=current.entry_price,
                                        exit_time=None,
                                        exit_price=None,
                                        pnl=None,
                                        contracts=scale_out_second_contracts,
                                        risk_dollars=float(current.risk * 2.0 * scale_out_second_contracts),
                                        outcome=None,
                                        exit_reason=None,
                                        pivot=current.pivot,
                                        flem=current.flem,
                                        flem_saved_time=current.flem_saved_time,
                                        reentry_time=current.reentry_time,
                                        pivot_time=current.pivot_time,
                                        risk=current.risk,
                                        target=float(target_price),
                                        stop_time=None,
                                        stop_price=current.stop_price,
                                        retrace_at_entry=current.retrace_at_entry,
                                        strong_count_recent3=current.strong_count_recent3,
                                        very_strong=current.very_strong
                                    )
                                    trades.append(remaining_trade)
                                    in_trade = True
                                else:
                                    current.outcome = 'win'
                                    hit_time = first_30s_target_hit(ev['timestamp'], direction, target_price)
                                    current.exit_time = hit_time if hit_time is not None else t
                                    current.exit_price = float(target_price)
                                    current.pnl = float(current.exit_price - current.entry_price) * float(current.contracts or 0)
                                    current.exit_reason = 'target'
                                    in_trade = False
                                    daily_pnl_dollars += float(current.pnl) * 2.0
                                    scale_out_active = False
                                    scale_out_done = False
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
                                daily_pnl_dollars += float(current.pnl) * 2.0
                                scale_out_active = False
                                scale_out_done = False
                            elif low <= target_price:
                                if scale_out_active and not scale_out_done:
                                    hit_time = first_30s_target_hit(ev['timestamp'], direction, target_price)
                                    first_exit_time = hit_time if hit_time is not None else t
                                    first_trade = Trade(
                                        day=current.day,
                                        direction=current.direction,
                                        entry_time=current.entry_time,
                                        entry_price=current.entry_price,
                                        exit_time=first_exit_time,
                                        exit_price=float(target_price),
                                        pnl=float(current.entry_price - target_price) * float(scale_out_first_contracts),
                                        contracts=scale_out_first_contracts,
                                        risk_dollars=float(current.risk * 2.0 * scale_out_first_contracts),
                                        outcome='win',
                                        exit_reason='target_scale1',
                                        pivot=current.pivot,
                                        flem=current.flem,
                                        flem_saved_time=current.flem_saved_time,
                                        reentry_time=current.reentry_time,
                                        pivot_time=current.pivot_time,
                                        risk=current.risk,
                                        target=current.target,
                                        stop_time=None,
                                        stop_price=current.stop_price,
                                        retrace_at_entry=current.retrace_at_entry,
                                        strong_count_recent3=current.strong_count_recent3,
                                        very_strong=current.very_strong
                                    )
                                    trades[-1] = first_trade
                                    daily_pnl_dollars += float(first_trade.pnl) * 2.0

                                    scale_out_done = True
                                    target_price = current.entry_price - float(scale_out_second_target_r or 4.0) * current.risk
                                    remaining_trade = Trade(
                                        day=current.day,
                                        direction=current.direction,
                                        entry_time=current.entry_time,
                                        entry_price=current.entry_price,
                                        exit_time=None,
                                        exit_price=None,
                                        pnl=None,
                                        contracts=scale_out_second_contracts,
                                        risk_dollars=float(current.risk * 2.0 * scale_out_second_contracts),
                                        outcome=None,
                                        exit_reason=None,
                                        pivot=current.pivot,
                                        flem=current.flem,
                                        flem_saved_time=current.flem_saved_time,
                                        reentry_time=current.reentry_time,
                                        pivot_time=current.pivot_time,
                                        risk=current.risk,
                                        target=float(target_price),
                                        stop_time=None,
                                        stop_price=current.stop_price,
                                        retrace_at_entry=current.retrace_at_entry,
                                        strong_count_recent3=current.strong_count_recent3,
                                        very_strong=current.very_strong
                                    )
                                    trades.append(remaining_trade)
                                    in_trade = True
                                else:
                                    current.outcome = 'win'
                                    hit_time = first_30s_target_hit(ev['timestamp'], direction, target_price)
                                    current.exit_time = hit_time if hit_time is not None else t
                                    current.exit_price = float(target_price)
                                    current.pnl = float(current.entry_price - current.exit_price) * float(current.contracts or 0)
                                    current.exit_reason = 'target'
                                    in_trade = False
                                    daily_pnl_dollars += float(current.pnl) * 2.0
                                    scale_out_active = False
                                    scale_out_done = False

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
            daily_pnl_dollars += float(current.pnl) * 2.0
            scale_out_active = False
            scale_out_done = False

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
