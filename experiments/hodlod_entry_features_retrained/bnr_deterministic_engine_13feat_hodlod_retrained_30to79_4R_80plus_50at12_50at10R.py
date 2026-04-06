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
    pwin_score: float = 0.0


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


PWIN_THRESH_DEFAULT = 0.64   # tuned after bucket analysis
MAX_CONTRACTS = 250
SLIPPAGE_TICKS = 1
TICK_SIZE = 0.25
MIN_RISK_PTS = 500.0 / (MAX_CONTRACTS * 2.0)  # $500 max risk, MNQ = $2/pt

def run_engine(df_1m: pd.DataFrame, df_30s: pd.DataFrame, allow_counter_candle_entry: bool, pwin_thresh: float = PWIN_THRESH_DEFAULT) -> List[Trade]:
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

    # Load pwin model (P(net P&L > 0) — trained via retrain_pwin_from_backtest.py)
    # Tune threshold with threshold sweep script
    PWIN_THRESH  = pwin_thresh
    _pwin_path   = os.environ.get(
        "PWIN_MODEL_PATH",
        "/Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pwin_13features_retrained_hodlod_entry_retrainedset.joblib",
    )
    ml_model     = joblib.load(_pwin_path) if os.path.exists(_pwin_path) else None
    ml_features  = ['retrace','pivot_flem_dist','time_since_pivot_sec','body_last','body_sum','body_mean','in_dir_ratio','max_in_dir_run','bars_since_pivot','zone_over_range','pivot_over_range','dist_to_extrema_atr','zone_to_extrema_atr','hod_rel_entry_atr','lod_rel_entry_atr']

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
        ml_allows_entry = False
        pwin_score = 0.0

        # For trades-in-progress
        in_trade = False
        stop_price = None
        target_price = None
        scale_out_active = False
        scale_out_stage = 0   # index into scale_out_plan
        scale_out_plan = []   # [(contracts, target_R), ...] — one entry per leg
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
                    ml_allows_entry = False
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
                    ml_allows_entry = False
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
                        scale_out_stage = 0
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
                    ml_allows_entry = False

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
                        scale_out_stage = 0
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
                    ml_allows_entry = False

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
                            ml_allows_entry = False
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
                            ml_allows_entry = False

                # After reentry: update pivot until entry
                if reentry_seen and not entry_triggered:
                    # Retrace/zone invalidations removed for robustness sweep

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

                                # HOD/LOD features (current point in day)
                                hod_now = float(day_1m_to_now['high'].max()) if not day_1m_to_now.empty else last_30s_close
                                lod_now = float(day_1m_to_now['low'].min())  if not day_1m_to_now.empty else last_30s_close
                                entry_price_est = last_30s_close
                                if direction == 'long':
                                    dist_to_extrema_atr  = (hod_now - entry_price_est) / atr if atr else 0.0
                                    zone_to_extrema_atr  = (hod_now - zone_price) / atr if (zone_price and atr) else 0.0
                                    hod_rel_entry_atr    = (hod_now - entry_price_est) / atr if atr else 0.0
                                    lod_rel_entry_atr    = 0.0
                                else:
                                    dist_to_extrema_atr  = (entry_price_est - lod_now) / atr if atr else 0.0
                                    zone_to_extrema_atr  = (zone_price - lod_now) / atr if (zone_price and atr) else 0.0
                                    hod_rel_entry_atr    = 0.0
                                    lod_rel_entry_atr    = (entry_price_est - lod_now) / atr if atr else 0.0

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
                                    'pivot_over_range': pivot_over_range,
                                    'dist_to_extrema_atr': dist_to_extrema_atr,
                                    'zone_to_extrema_atr': zone_to_extrema_atr,
                                    'hod_rel_entry_atr': hod_rel_entry_atr,
                                    'lod_rel_entry_atr': lod_rel_entry_atr,
                                }
                                x = pd.DataFrame([features])[ml_features].fillna(0.0)
                                pwin_score = float(ml_model.predict_proba(x)[0][1])
                                # Hard gate to ensure sweep thresholds are actually enforced
                                ml_allows_entry = (pwin_score >= PWIN_THRESH)

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

                    # Removed retrace min/max gating and prior 1m candle color gate
                    if retrace is not None:
                        # Recompute ML score at the moment of entry to avoid stale gating
                        if ml_model is not None and ml_features is not None and last_30s_close is not None:
                            pivot_flem_dist = abs(pivot - flem) if (pivot is not None and flem is not None) else None
                            time_since_pivot = (t - pivot_time).total_seconds() if pivot_time is not None else None
                            if pivot_flem_dist not in (0, None) and time_since_pivot is not None and body_count_30s > 0:
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

                                # HOD/LOD features (current point in day)
                                hod_now = float(day_1m_to_now['high'].max()) if not day_1m_to_now.empty else last_30s_close
                                lod_now = float(day_1m_to_now['low'].min())  if not day_1m_to_now.empty else last_30s_close
                                entry_price_est = last_30s_close
                                if direction == 'long':
                                    dist_to_extrema_atr  = (hod_now - entry_price_est) / atr if atr else 0.0
                                    zone_to_extrema_atr  = (hod_now - zone_price) / atr if (zone_price and atr) else 0.0
                                    hod_rel_entry_atr    = (hod_now - entry_price_est) / atr if atr else 0.0
                                    lod_rel_entry_atr    = 0.0
                                else:
                                    dist_to_extrema_atr  = (entry_price_est - lod_now) / atr if atr else 0.0
                                    zone_to_extrema_atr  = (zone_price - lod_now) / atr if (zone_price and atr) else 0.0
                                    hod_rel_entry_atr    = 0.0
                                    lod_rel_entry_atr    = (entry_price_est - lod_now) / atr if atr else 0.0

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
                                    'pivot_over_range': pivot_over_range,
                                    'dist_to_extrema_atr': dist_to_extrema_atr,
                                    'zone_to_extrema_atr': zone_to_extrema_atr,
                                    'hod_rel_entry_atr': hod_rel_entry_atr,
                                    'lod_rel_entry_atr': lod_rel_entry_atr,
                                }
                                x = pd.DataFrame([features])[ml_features].fillna(0.0)
                                pwin_score = float(ml_model.predict_proba(x)[0][1])
                                ml_allows_entry = (pwin_score >= PWIN_THRESH)

                        if ml_model is not None and not ml_allows_entry:
                            continue
                        if (ml_model is not None and ml_allows_entry) or (ml_model is None and (strong_count >= 2 or very_strong)):
                            if ml_model is not None and pwin_score < PWIN_THRESH:
                                raise RuntimeError(f"Entry with pwin {pwin_score:.6f} below thresh {PWIN_THRESH:.2f} at {t}")
                            entry_triggered = True
                            in_trade = True
                            entry_time = t
                            # Apply slippage per execution (entry)
                            if direction == 'long':
                                entry_price = close + (SLIPPAGE_TICKS * TICK_SIZE)
                            else:
                                entry_price = close - (SLIPPAGE_TICKS * TICK_SIZE)
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
                            if risk < MIN_RISK_PTS:
                                entry_triggered = False
                                in_trade = False
                                stop_price = None
                                target_price = None
                                continue
                            contracts = int(500 // risk_dollars_per_contract) if risk_dollars_per_contract > 0 else 0
                            if contracts > MAX_CONTRACTS:
                                contracts = MAX_CONTRACTS
                            if contracts < 1:
                                # Skip trade if risk > $500
                                entry_triggered = False
                                in_trade = False
                                stop_price = None
                                target_price = None
                                continue

                            # Scale-out plan:
                            #   contracts >= 80 => 50% at 1.2R, 50% at 10R
                            #   30 <= contracts < 80 => 75% at 1.2R, 25% at 4R
                            #   contracts < 30 => preserve prior small-trade behavior
                            scale_out_stage = 0
                            if contracts >= 80:
                                q1 = max(1, int(round(contracts * 0.50)))
                                q2 = max(1, contracts - q1)
                                scale_out_plan = [(q1, 1.2), (q2, 10.0)]
                            elif contracts >= 30:
                                q1 = max(1, int(round(contracts * 0.75)))
                                q2 = max(1, contracts - q1)
                                scale_out_plan = [(q1, 1.2), (q2, 4.0)]
                            elif contracts >= 13:
                                q = contracts // 4
                                rem = contracts - 3 * q
                                scale_out_plan = [(q, 1.2), (q, 2.5), (q, 4.0), (rem, 6.5)]
                            else:
                                q1 = contracts // 4
                                q2 = int(contracts * 0.35)
                                q3 = contracts - q1 - q2
                                scale_out_plan = [(q1, 1.2), (q2, 2.0), (q3, 3.0)]
                            scale_out_active = len(scale_out_plan) > 1 and all(q > 0 for q, _ in scale_out_plan)
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
                                very_strong=bool(very_strong),
                                pwin_score=pwin_score if ml_model is not None else 0.0
                            ))
            # Manage trade (stop/target) using 1m bars
            if in_trade and trades:
                current = trades[-1]
                if current.outcome is None:
                        # Prevent same-bar exits that would rely on earlier 30s data
                        if current.entry_time == t:
                            continue
                        # Fixed stop/target — no trail, no 30-min rule
                        if direction == 'long':
                            # Stop is based on 1m close below pivot
                            if close <= stop_price:
                                current.outcome = 'loss'
                                current.exit_time = t
                                current.exit_price = float(close - (SLIPPAGE_TICKS * TICK_SIZE))
                                current.pnl = float(current.exit_price - current.entry_price) * float(current.contracts or 0)
                                current.stop_time = t
                                current.stop_price = float(stop_price)
                                current.exit_reason = 'stop'
                                in_trade = False
                                daily_pnl_dollars += float(current.pnl) * 2.0
                                scale_out_active = False
                                scale_out_stage = 0
                                # Reset setup after close (keep flem for re-entry)
                                entry_triggered = False
                                reentry_seen = False
                                pivot = None
                                pivot_time = None
                                reentry_time = None
                                last_3_strong = []
                            elif high >= target_price:
                                if scale_out_active and scale_out_stage < len(scale_out_plan) - 1:
                                    # Intermediate scale-out leg (long)
                                    leg_q, _ = scale_out_plan[scale_out_stage]
                                    next_q, next_r = scale_out_plan[scale_out_stage + 1]
                                    hit_time = first_30s_target_hit(ev['timestamp'], direction, target_price)
                                    leg_exit_time = hit_time if hit_time is not None else t
                                    leg_exit_price = float(target_price - (SLIPPAGE_TICKS * TICK_SIZE))
                                    leg_trade = Trade(
                                        day=current.day,
                                        direction=current.direction,
                                        entry_time=current.entry_time,
                                        entry_price=current.entry_price,
                                        exit_time=leg_exit_time,
                                        exit_price=leg_exit_price,
                                        pnl=float(leg_exit_price - current.entry_price) * float(leg_q),
                                        contracts=leg_q,
                                        risk_dollars=float(current.risk * 2.0 * leg_q),
                                        outcome='win',
                                        exit_reason=f'target_scale{scale_out_stage + 1}',
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
                                        very_strong=current.very_strong,
                                        pwin_score=current.pwin_score
                                    )
                                    trades[-1] = leg_trade
                                    daily_pnl_dollars += float(leg_trade.pnl) * 2.0
                                    scale_out_stage += 1
                                    target_price = current.entry_price + next_r * current.risk
                                    remaining_qty = sum(q for q, _ in scale_out_plan[scale_out_stage:])
                                    remaining_trade = Trade(
                                        day=current.day,
                                        direction=current.direction,
                                        entry_time=current.entry_time,
                                        entry_price=current.entry_price,
                                        exit_time=None,
                                        exit_price=None,
                                        pnl=None,
                                        contracts=remaining_qty,
                                        risk_dollars=float(current.risk * 2.0 * remaining_qty),
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
                                        very_strong=current.very_strong,
                                        pwin_score=current.pwin_score
                                    )
                                    trades.append(remaining_trade)
                                    in_trade = True
                                else:
                                    # Final exit (last scale-out leg or no scale-out)
                                    current.outcome = 'win'
                                    hit_time = first_30s_target_hit(ev['timestamp'], direction, target_price)
                                    current.exit_time = hit_time if hit_time is not None else t
                                    current.exit_price = float(target_price - (SLIPPAGE_TICKS * TICK_SIZE))
                                    current.pnl = float(current.exit_price - current.entry_price) * float(current.contracts or 0)
                                    current.exit_reason = 'target'
                                    in_trade = False
                                    daily_pnl_dollars += float(current.pnl) * 2.0
                                    scale_out_active = False
                                    scale_out_stage = 0
                                    # Reset setup after close (keep flem for re-entry)
                                    entry_triggered = False
                                    reentry_seen = False
                                    pivot = None
                                    pivot_time = None
                                    reentry_time = None
                                    last_3_strong = []
                        else:
                            # Stop is based on 1m close above pivot
                            if close >= stop_price:
                                current.outcome = 'loss'
                                current.exit_time = t
                                current.exit_price = float(close + (SLIPPAGE_TICKS * TICK_SIZE))
                                current.pnl = float(current.entry_price - current.exit_price) * float(current.contracts or 0)
                                current.stop_time = t
                                current.stop_price = float(stop_price)
                                current.exit_reason = 'stop'
                                in_trade = False
                                daily_pnl_dollars += float(current.pnl) * 2.0
                                scale_out_active = False
                                scale_out_stage = 0
                                # Reset setup after close (keep flem for re-entry)
                                entry_triggered = False
                                reentry_seen = False
                                pivot = None
                                pivot_time = None
                                reentry_time = None
                                last_3_strong = []
                            elif low <= target_price:
                                if scale_out_active and scale_out_stage < len(scale_out_plan) - 1:
                                    # Intermediate scale-out leg (short)
                                    leg_q, _ = scale_out_plan[scale_out_stage]
                                    next_q, next_r = scale_out_plan[scale_out_stage + 1]
                                    hit_time = first_30s_target_hit(ev['timestamp'], direction, target_price)
                                    leg_exit_time = hit_time if hit_time is not None else t
                                    leg_exit_price = float(target_price + (SLIPPAGE_TICKS * TICK_SIZE))
                                    leg_trade = Trade(
                                        day=current.day,
                                        direction=current.direction,
                                        entry_time=current.entry_time,
                                        entry_price=current.entry_price,
                                        exit_time=leg_exit_time,
                                        exit_price=leg_exit_price,
                                        pnl=float(current.entry_price - leg_exit_price) * float(leg_q),
                                        contracts=leg_q,
                                        risk_dollars=float(current.risk * 2.0 * leg_q),
                                        outcome='win',
                                        exit_reason=f'target_scale{scale_out_stage + 1}',
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
                                        very_strong=current.very_strong,
                                        pwin_score=current.pwin_score
                                    )
                                    trades[-1] = leg_trade
                                    daily_pnl_dollars += float(leg_trade.pnl) * 2.0
                                    scale_out_stage += 1
                                    target_price = current.entry_price - next_r * current.risk
                                    remaining_qty = sum(q for q, _ in scale_out_plan[scale_out_stage:])
                                    remaining_trade = Trade(
                                        day=current.day,
                                        direction=current.direction,
                                        entry_time=current.entry_time,
                                        entry_price=current.entry_price,
                                        exit_time=None,
                                        exit_price=None,
                                        pnl=None,
                                        contracts=remaining_qty,
                                        risk_dollars=float(current.risk * 2.0 * remaining_qty),
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
                                        very_strong=current.very_strong,
                                        pwin_score=current.pwin_score
                                    )
                                    trades.append(remaining_trade)
                                    in_trade = True
                                else:
                                    # Final exit (last scale-out leg or no scale-out)
                                    current.outcome = 'win'
                                    hit_time = first_30s_target_hit(ev['timestamp'], direction, target_price)
                                    current.exit_time = hit_time if hit_time is not None else t
                                    current.exit_price = float(target_price + (SLIPPAGE_TICKS * TICK_SIZE))
                                    current.pnl = float(current.entry_price - current.exit_price) * float(current.contracts or 0)
                                    current.exit_reason = 'target'
                                    in_trade = False
                                    daily_pnl_dollars += float(current.pnl) * 2.0
                                    scale_out_active = False
                                    scale_out_stage = 0
                                    # Reset setup after close (keep flem for re-entry)
                                    entry_triggered = False
                                    reentry_seen = False
                                    pivot = None
                                    pivot_time = None
                                    reentry_time = None
                                    last_3_strong = []

        # End of day cleanup: force-close any unclosed trade at session end
        if trades and trades[-1].outcome is None:
            last_bar = day_1m[day_1m['timestamp'] <= session_end].iloc[-1]
            t = last_bar['event_time']
            close = float(last_bar['close'])
            current = trades[-1]
            current.outcome = 'forced_close'
            current.exit_time = t
            if current.direction == 'long':
                current.exit_price = close - (SLIPPAGE_TICKS * TICK_SIZE)
            else:
                current.exit_price = close + (SLIPPAGE_TICKS * TICK_SIZE)
            if current.direction == 'long':
                current.pnl = float(current.exit_price - current.entry_price) * float(current.contracts or 0)
            else:
                current.pnl = float(current.entry_price - current.exit_price) * float(current.contracts or 0)
            current.exit_reason = 'forced_close'
            in_trade = False
            daily_pnl_dollars += float(current.pnl) * 2.0
            scale_out_active = False
            scale_out_stage = 0

    return trades


def main():
    p1_2025_q = [
        "/Users/radhikaarora/Documents/New Project/output/market/quarterly/mnq_1m_2025_q1.csv",
        "/Users/radhikaarora/Documents/New Project/output/market/quarterly/mnq_1m_2025_q2.csv",
        "/Users/radhikaarora/Documents/New Project/output/market/quarterly/mnq_1m_2025_q3.csv",
        "/Users/radhikaarora/Documents/New Project/output/market/quarterly/mnq_1m_2025_q4.csv",
    ]
    p30s_2025_q = [
        "/Users/radhikaarora/Documents/New Project/output/market/quarterly/mnq_30s_2025_q1.csv",
        "/Users/radhikaarora/Documents/New Project/output/market/quarterly/mnq_30s_2025_q2.csv",
        "/Users/radhikaarora/Documents/New Project/output/market/quarterly/mnq_30s_2025_q3.csv",
        "/Users/radhikaarora/Documents/New Project/output/market/quarterly/mnq_30s_2025_q4.csv",
    ]

    df_1m_2025  = pd.concat([pd.read_csv(p) for p in p1_2025_q], ignore_index=True)
    df_30s_2025 = pd.concat([pd.read_csv(p) for p in p30s_2025_q], ignore_index=True)

    # Parse timestamps
    for df in (df_1m_2025, df_30s_2025):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert("America/New_York")

    df_1m = df_1m_2025.drop_duplicates('timestamp').sort_values('timestamp')
    df_30s_all = df_30s_2025.drop_duplicates('timestamp').sort_values('timestamp')

    ranges_2025  = [("2025-01-01", "2025-12-31")]
    # Only 2025 data available in provided quarterly files

    def run_for_ranges(allow_counter: bool, ranges, bars_30s=None) -> pd.DataFrame:
        src_30s = bars_30s if bars_30s is not None else df_30s_all
        all_trades = []
        for start_s, end_s in ranges:
            start = pd.Timestamp(start_s)
            end = pd.Timestamp(end_s)
            slice_1m  = df_1m[(df_1m['timestamp'].dt.date >= start.date()) & (df_1m['timestamp'].dt.date <= end.date())]
            slice_30s = src_30s[(src_30s['timestamp'].dt.date >= start.date()) & (src_30s['timestamp'].dt.date <= end.date())]
            all_trades.extend(run_engine(slice_1m, slice_30s, allow_counter_candle_entry=allow_counter))
        return pd.DataFrame([t.__dict__ for t in all_trades])

    out_allow = run_for_ranges(True, ranges_2025)
    out_allow_path = "/Users/radhikaarora/Documents/Trading ML/ML V2/output_bnr_det_2025_13feat_pwin64_retrained_hodlod_entry.csv"
    out_allow.to_csv(out_allow_path, index=False)
    print(f"Wrote {len(out_allow)} trades to {out_allow_path}")

    # 2026 data not available in provided quarterly files


if __name__ == "__main__":
    main()
