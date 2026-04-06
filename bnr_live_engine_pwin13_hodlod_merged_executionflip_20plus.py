#!/usr/bin/env python3
"""
BNR Live Trading Engine — merged HOD/LOD pwin model variant

Live-paper variant of the current merged model:
  - Uses the merged HOD/LOD retrained joblib
  - PWIN_THRESH = 0.40
  - Uses the 15-feature matrix from the current no-gates backtest
  - No slippage adjustment in fills

Accepts bars via:
    engine.on_bar_1m(bar)   # bar: {'1': time_ms, '2': O, '3': H, '4': L, '5': C, '6': V}
    engine.on_bar_30s(bar)  # bar: {'time_ms': ms, 'open': O, 'high': H, 'low': L, 'close': C}
"""

import os
import csv
import joblib
import pytz
from datetime import datetime, timedelta, date
from typing import Optional
from dataclasses import dataclass, field

# ─── constants ────────────────────────────────────────────────────────────────
ET = pytz.timezone("America/New_York")
MODEL_PATH   = "/Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pwin_13features_retrained_hodlod_entry_merged_live.joblib"
PWIN_THRESH  = 0.40
ML_FEATURES  = [
    'retrace', 'pivot_flem_dist', 'time_since_pivot_sec',
    'body_last', 'body_sum', 'body_mean', 'in_dir_ratio',
    'max_in_dir_run', 'bars_since_pivot', 'zone_over_range', 'pivot_over_range',
    'dist_to_extrema_atr', 'zone_to_extrema_atr', 'hod_rel_entry_atr', 'lod_rel_entry_atr',
]
MNQ_DOLLARS_PER_POINT = 2.0
MAX_RISK_DOLLARS = 500.0
MAX_CONTRACTS = 250
MIN_RISK_PTS = MAX_RISK_DOLLARS / (MAX_CONTRACTS * MNQ_DOLLARS_PER_POINT)
MIN_PROFIT_TARGET_PTS = 2.5
TICK_SIZE = 0.25
FLIP_SIDE_THRESHOLD_CONTRACTS = 20


def mirrored_direction(direction: str, mirror_mode: bool) -> str:
    if not mirror_mode:
        return direction
    return 'short' if direction == 'long' else 'long'


def realized_pnl(virtual_pnl_points_times_contracts: float, mirror_mode: bool) -> float:
    return -virtual_pnl_points_times_contracts if mirror_mode else virtual_pnl_points_times_contracts


def round_to_tick(price: float) -> float:
    return round(price / TICK_SIZE) * TICK_SIZE

SESSION_OPEN_H,  SESSION_OPEN_M  = 9, 30
SESSION_CLOSE_H, SESSION_CLOSE_M = 12, 1

DISP_HIGH_ZONE_THRESHOLD  = 0.10568226033342312
DISP_HIGH_PIVOT_THRESHOLD = 0.3795379537953795
DISP_LOW_ZONE_THRESHOLD   = 0.0848692546366965
DISP_LOW_PIVOT_THRESHOLD  = 0.23516193082722905


# ─── trade record ─────────────────────────────────────────────────────────────
@dataclass
class LiveTrade:
    day: str
    direction: str
    virtual_direction: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    contracts: Optional[int] = None
    risk_dollars: Optional[float] = None
    outcome: Optional[str] = None
    exit_reason: Optional[str] = None
    pivot: float = 0.0
    flem: float = 0.0
    reentry_time: Optional[datetime] = None
    pivot_time: Optional[datetime] = None
    risk: float = 0.0
    target: float = 0.0
    stop_price: Optional[float] = None
    retrace_at_entry: float = 0.0
    displacement: str = ''
    pwin_score: float = 0.0
    mirror_mode: bool = False


# ─── rolling ATR (14-period Wilder smoothing) ─────────────────────────────────
class RollingATR:
    def __init__(self, period: int = 14):
        self._period = period
        self._bars: list[dict] = []
        self._atr: Optional[float] = None

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self._bars:
            prev_c = self._bars[-1]['close']
            tr = max(high - low, abs(high - prev_c), abs(low - prev_c))
        else:
            tr = high - low

        self._bars.append({'high': high, 'low': low, 'close': close, 'tr': tr})
        n = len(self._bars)
        if n < self._period + 1:
            self._atr = sum(b['tr'] for b in self._bars) / n
        else:
            if n == self._period + 1:
                self._atr = sum(b['tr'] for b in self._bars[:self._period]) / self._period
            self._atr = (self._atr * (self._period - 1) + tr) / self._period
        return self._atr

    @property
    def value(self) -> Optional[float]:
        return self._atr


# ─── main engine ──────────────────────────────────────────────────────────────
class BNRLiveEngine13HodlodMergedExecutionFlip20Plus:
    """
    BNR live engine using the merged HOD/LOD pwin model (PWIN_THRESH=0.40) with execution-side flip for 20+ contract trades.
    This mirrors the current no-gates backtest entry filter, without slippage.
    """

    def __init__(self, out_dir: str = "output/bnr_live_pwin13_hodlod_merged",
                 allow_counter_candle: bool = True):
        self._out_dir = out_dir
        self._allow_counter = allow_counter_candle
        os.makedirs(out_dir, exist_ok=True)

        if os.path.exists(MODEL_PATH):
            self._model = joblib.load(MODEL_PATH)
            self._features = ML_FEATURES
            print(f"[BNR13HMX20] ML model loaded ({len(self._features)} features, thresh={PWIN_THRESH})")
        else:
            self._model = None
            self._features = None
            print("[BNR13HMX20] ML model not found — strong-candle fallback active")

        self._atr_calc = RollingATR(14)
        self._trades: list[LiveTrade] = []
        self._daily_pnl: float = 0.0

        self._current_date: Optional[date] = None
        self._zone_high: Optional[float] = None
        self._zone_low: Optional[float] = None

        self._direction: Optional[str] = None
        self._candidate_active: bool = False
        self._reentry_seen: bool = False
        self._reentry_time: Optional[datetime] = None
        self._flem: Optional[float] = None
        self._flem_saved_time: Optional[datetime] = None
        self._pivot: Optional[float] = None
        self._pivot_time: Optional[datetime] = None
        self._entry_triggered: bool = False
        self._last_3_strong: list[bool] = []

        self._body_sum_30s: float = 0.0
        self._body_count_30s: int = 0
        self._in_dir_count_30s: int = 0
        self._max_run_30s: int = 0
        self._run_30s: int = 0
        self._last_body_30s: Optional[float] = None
        self._last_30s_close: Optional[float] = None

        self._in_trade: bool = False
        self._stop_price: Optional[float] = None
        self._target_price: Optional[float] = None
        self._scale_out_active: bool = False
        self._scale_out_stage: int = 0
        self._scale_out_plan: list[tuple[int, float]] = []

        self._session_1m_bars: list[dict] = []
        self._prev_1m_bar: Optional[dict] = None

    # ── session helpers ───────────────────────────────────────────────────────

    def _reset_session(self):
        self._zone_high = None
        self._zone_low = None
        self._session_1m_bars = []
        self._prev_1m_bar = None
        self._atr_calc = RollingATR(14)
        self._reset_candidate()
        self._daily_pnl = 0.0
        self._in_trade = False
        self._stop_price = None
        self._target_price = None
        self._scale_out_active = False
        self._scale_out_stage = 0
        self._scale_out_plan = []

    def _reset_candidate(self):
        self._direction = None
        self._candidate_active = False
        self._reentry_seen = False
        self._reentry_time = None
        self._flem = None
        self._flem_saved_time = None
        self._pivot = None
        self._pivot_time = None
        self._entry_triggered = False
        self._last_3_strong = []
        self._reset_retrace_accumulators()

    def _reset_after_close(self):
        """After any trade close, reset pivot/reentry state but keep flem intact."""
        self._reentry_seen = False
        self._reentry_time = None
        self._pivot = None
        self._pivot_time = None
        self._entry_triggered = False
        self._last_3_strong = []
        self._reset_retrace_accumulators()

    def _reset_retrace_accumulators(self):
        self._body_sum_30s = 0.0
        self._body_count_30s = 0
        self._in_dir_count_30s = 0
        self._max_run_30s = 0
        self._run_30s = 0
        self._last_body_30s = None
        self._last_30s_close = None

    def _is_in_session(self, ts: datetime) -> bool:
        h, m = ts.hour, ts.minute
        after_open   = (h > SESSION_OPEN_H)  or (h == SESSION_OPEN_H  and m >= SESSION_OPEN_M)
        before_close = (h < SESSION_CLOSE_H) or (h == SESSION_CLOSE_H and m == 0)
        return after_open and before_close

    def _session_open_ts(self, ts: datetime) -> datetime:
        return ts.replace(hour=SESSION_OPEN_H, minute=SESSION_OPEN_M,
                          second=0, microsecond=0)

    # ── bar parsing ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_1m(bar: dict) -> dict:
        open_ms  = int(bar['1'])
        open_ts  = datetime.fromtimestamp(open_ms / 1000, tz=ET)
        close_ts = open_ts + timedelta(minutes=1)
        return {
            'open_ts': open_ts,
            'close_ts': close_ts,
            'open':  float(bar['2']),
            'high':  float(bar['3']),
            'low':   float(bar['4']),
            'close': float(bar['5']),
            'volume': int(bar['6']),
        }

    @staticmethod
    def _parse_30s(bar: dict) -> dict:
        ts = datetime.fromtimestamp(bar['time_ms'] / 1000, tz=ET)
        return {
            'ts':    ts,
            'open':  float(bar['open']),
            'high':  float(bar['high']),
            'low':   float(bar['low']),
            'close': float(bar['close']),
        }

    # ── day-range / HOD-LOD helpers ───────────────────────────────────────────

    def _day_range(self) -> float:
        if not self._session_1m_bars:
            return 0.0
        return (max(b['high'] for b in self._session_1m_bars) -
                min(b['low']  for b in self._session_1m_bars))

    def _hod(self) -> float:
        if not self._session_1m_bars:
            return 0.0
        return max(b['high'] for b in self._session_1m_bars)

    def _lod(self) -> float:
        if not self._session_1m_bars:
            return 0.0
        return min(b['low'] for b in self._session_1m_bars)

    # ── displacement category ─────────────────────────────────────────────────

    def _displacement_category(self) -> str:
        if self._flem is None or self._pivot is None or self._zone_high is None:
            return 'medium'
        dr = self._day_range()
        if dr == 0:
            return 'medium'
        pivot_flem_dist = abs(self._flem - self._pivot)
        if self._direction == 'long':
            dist_zone = self._flem - self._zone_high
        else:
            dist_zone = self._zone_low - self._flem
        zone_over_range  = dist_zone / dr if dist_zone is not None else 0.0
        pivot_over_range = pivot_flem_dist / dr
        if (zone_over_range  >= DISP_HIGH_ZONE_THRESHOLD and
                pivot_over_range >= DISP_HIGH_PIVOT_THRESHOLD):
            return 'high'
        elif (zone_over_range  <= DISP_LOW_ZONE_THRESHOLD and
              pivot_over_range <= DISP_LOW_PIVOT_THRESHOLD):
            return 'low'
        return 'medium'

    # ── ML scoring (15 features) ──────────────────────────────────────────────

    def _ml_score(self, retrace_ml: float, entry_price_est: float, current_ts: datetime) -> Optional[float]:
        if self._model is None or self._body_count_30s == 0:
            return None
        if self._pivot is None or self._flem is None or self._flem == self._pivot:
            return None

        import pandas as pd
        pivot_flem_dist = abs(self._flem - self._pivot)
        dr = self._day_range()
        atr = self._atr_calc.value or 1.0

        if self._direction == 'long':
            dist_zone = self._flem - self._zone_high
            dist_to_extrema_atr  = (self._hod() - entry_price_est) / atr
            zone_to_extrema_atr  = (self._hod() - self._zone_high) / atr if self._zone_high else 0.0
            hod_rel_entry_atr    = (self._hod() - entry_price_est) / atr
            lod_rel_entry_atr    = 0.0
        else:
            dist_zone = self._zone_low - self._flem
            dist_to_extrema_atr  = (entry_price_est - self._lod()) / atr
            zone_to_extrema_atr  = (self._zone_low - self._lod()) / atr if self._zone_low else 0.0
            hod_rel_entry_atr    = 0.0
            lod_rel_entry_atr    = (entry_price_est - self._lod()) / atr

        zone_over_range  = dist_zone / dr if dr > 0 else 0.0
        pivot_over_range = pivot_flem_dist / dr if dr > 0 else 0.0
        body_mean    = self._body_sum_30s / self._body_count_30s
        in_dir_ratio = self._in_dir_count_30s / self._body_count_30s
        time_since_pivot = (
            (current_ts - self._pivot_time).total_seconds()
            if self._pivot_time else 0.0
        )

        features = {
            'retrace':              retrace_ml,
            'pivot_flem_dist':      pivot_flem_dist,
            'time_since_pivot_sec': time_since_pivot,
            'body_last':            self._last_body_30s or 0.0,
            'body_sum':             self._body_sum_30s,
            'body_mean':            body_mean,
            'in_dir_ratio':         in_dir_ratio,
            'max_in_dir_run':       self._max_run_30s,
            'bars_since_pivot':     self._body_count_30s,
            'zone_over_range':      zone_over_range,
            'pivot_over_range':     pivot_over_range,
            'dist_to_extrema_atr':  dist_to_extrema_atr,
            'zone_to_extrema_atr':  zone_to_extrema_atr,
            'hod_rel_entry_atr':    hod_rel_entry_atr,
            'lod_rel_entry_atr':    lod_rel_entry_atr,
        }
        x = pd.DataFrame([features])[self._features].fillna(0.0)
        return float(self._model.predict_proba(x)[0][1])

    # ── trade exit helper ─────────────────────────────────────────────────────

    def _close_trade(self, trade: LiveTrade, exit_ts: datetime,
                     exit_price: float, exit_reason: str, outcome: str):
        trade.exit_time   = exit_ts
        trade.exit_price  = exit_price
        trade.exit_reason = exit_reason
        trade.outcome     = outcome
        trade.stop_price  = self._stop_price

        if trade.virtual_direction == 'long':
            virtual_pnl = (exit_price - trade.entry_price) * (trade.contracts or 0)
        else:
            virtual_pnl = (trade.entry_price - exit_price) * (trade.contracts or 0)
        trade.pnl = realized_pnl(virtual_pnl, trade.mirror_mode)

        pnl_dollars = trade.pnl * MNQ_DOLLARS_PER_POINT
        self._daily_pnl += pnl_dollars

        tag = '✓ WIN ' if outcome == 'win' else '✗ LOSS'
        print(f"[BNR13HMX20] EXIT {tag}  {exit_ts.strftime('%H:%M:%S')}  {exit_reason} | "
              f"entry={trade.entry_price:.2f} → exit={exit_price:.2f} | "
              f"pnl={pnl_dollars:+.0f}$  daily={self._daily_pnl:+.0f}$")

        self._in_trade = False
        self._scale_out_active = False
        self._scale_out_stage = 0
        self._reset_after_close()

    # ── on_bar_30s ────────────────────────────────────────────────────────────

    def on_bar_30s(self, bar: dict):
        b = self._parse_30s(bar)
        ts    = b['ts']
        close = b['close']
        open_ = b['open']

        if not self._is_in_session(ts):
            return
        if self._current_date is None or self._zone_high is None:
            return

        if self._candidate_active and not self._in_trade:
            if self._direction == 'long' and close < self._zone_low:
                print(f"[BNR13HMX20] INVALIDATED long (30s close {close:.2f} < zone_low {self._zone_low:.2f})")
                self._reset_candidate()
                if close < self._zone_low:
                    self._direction = 'short'
                    self._candidate_active = True
                    print(f"[BNR13HMX20] Breakout SHORT @ {close:.2f}  [30s]")
                return
            if self._direction == 'short' and close > self._zone_high:
                print(f"[BNR13HMX20] INVALIDATED short (30s close {close:.2f} > zone_high {self._zone_high:.2f})")
                self._reset_candidate()
                if close > self._zone_high:
                    self._direction = 'long'
                    self._candidate_active = True
                    print(f"[BNR13HMX20] Breakout LONG  @ {close:.2f}  [30s]")
                return

        if not self._candidate_active:
            if close > self._zone_high:
                self._direction = 'long'
                self._candidate_active = True
                print(f"[BNR13HMX20] Breakout LONG  @ {close:.2f}  (zone_high={self._zone_high:.2f}) [30s]")
            elif close < self._zone_low:
                self._direction = 'short'
                self._candidate_active = True
                print(f"[BNR13HMX20] Breakout SHORT @ {close:.2f}  (zone_low={self._zone_low:.2f}) [30s]")

        if self._reentry_seen and not self._entry_triggered and self._pivot_time is not None:
            if ts >= self._pivot_time:
                body = abs(close - open_)
                self._last_body_30s   = body
                self._last_30s_close  = close
                self._body_sum_30s   += body
                self._body_count_30s += 1
                in_dir = (close > open_) if self._direction == 'long' else (close < open_)
                if in_dir:
                    self._in_dir_count_30s += 1
                    self._run_30s += 1
                    self._max_run_30s = max(self._max_run_30s, self._run_30s)
                else:
                    self._run_30s = 0

    # ── on_bar_1m ─────────────────────────────────────────────────────────────

    def on_bar_1m(self, bar: dict):
        b  = self._parse_1m(bar)
        ts = b['close_ts']
        open_ts = b['open_ts']

        today = ts.astimezone(ET).date()
        if today != self._current_date:
            if self._in_trade and self._trades and self._trades[-1].outcome is None:
                self._close_trade(self._trades[-1], ts, b['close'], 'forced_close', 'forced_close')
            if self._current_date is not None:
                self.save_trades_csv()
                self.print_summary()
            self._current_date = today
            self._reset_session()
            print(f"\n[BNR13HMX20] ── New session: {today} ──")

        if not self._is_in_session(open_ts):
            return

        session_open = self._session_open_ts(open_ts)
        if open_ts == session_open and self._zone_high is None:
            self._zone_high = b['high']
            self._zone_low  = b['low']
            print(f"[BNR13HMX20] Zone: H={self._zone_high:.2f}  L={self._zone_low:.2f}")

        self._session_1m_bars.append(b)
        atr = self._atr_calc.update(b['high'], b['low'], b['close'])

        if self._zone_high is None:
            return

        close = b['close']
        high  = b['high']
        low   = b['low']

        # Direction flip
        if self._candidate_active and self._direction == 'long' and close < self._zone_low:
            if self._in_trade and self._trades and self._trades[-1].outcome is None:
                self._close_trade(self._trades[-1], ts, close, 'direction_flip', 'loss')
            self._direction = 'short'
            self._candidate_active = True
            self._reentry_seen = False
            self._reentry_time = None
            self._flem = None
            self._pivot = None
            self._entry_triggered = False
            self._last_3_strong = []
            self._stop_price = None
            self._target_price = None
            self._reset_retrace_accumulators()
            print(f"[BNR13HMX20] Direction flip → SHORT @ {close:.2f}")

        elif self._candidate_active and self._direction == 'short' and close > self._zone_high:
            if self._in_trade and self._trades and self._trades[-1].outcome is None:
                self._close_trade(self._trades[-1], ts, close, 'direction_flip', 'loss')
            self._direction = 'long'
            self._candidate_active = True
            self._reentry_seen = False
            self._reentry_time = None
            self._flem = None
            self._pivot = None
            self._entry_triggered = False
            self._last_3_strong = []
            self._stop_price = None
            self._target_price = None
            self._reset_retrace_accumulators()
            print(f"[BNR13HMX20] Direction flip → LONG  @ {close:.2f}")

        if not self._candidate_active:
            if close > self._zone_high:
                self._direction = 'long'
                self._candidate_active = True
                print(f"[BNR13HMX20] Breakout LONG  @ {close:.2f}  (zone_high={self._zone_high:.2f})")
            elif close < self._zone_low:
                self._direction = 'short'
                self._candidate_active = True
                print(f"[BNR13HMX20] Breakout SHORT @ {close:.2f}  (zone_low={self._zone_low:.2f})")
            else:
                self._prev_1m_bar = b
                return

        # FLEM + retest
        if self._candidate_active and not self._reentry_seen:
            if self._direction == 'long':
                self._flem = high if self._flem is None else max(self._flem, high)
                if close < b['open'] and low <= self._zone_high and high >= self._zone_low:
                    self._reentry_seen = True
                    self._reentry_time = ts
                    self._flem_saved_time = ts
                    self._pivot = low
                    self._pivot_time = ts
                    self._reset_retrace_accumulators()
                    print(f"[BNR13HMX20] Retest/Pivot LONG  pivot={self._pivot:.2f}  flem={self._flem:.2f}")
            else:
                self._flem = low if self._flem is None else min(self._flem, low)
                if close > b['open'] and low <= self._zone_high and high >= self._zone_low:
                    self._reentry_seen = True
                    self._reentry_time = ts
                    self._flem_saved_time = ts
                    self._pivot = high
                    self._pivot_time = ts
                    self._reset_retrace_accumulators()
                    print(f"[BNR13HMX20] Retest/Pivot SHORT pivot={self._pivot:.2f}  flem={self._flem:.2f}")

        # Retrace + entry
        if self._reentry_seen and not self._entry_triggered:
            if self._flem is not None and self._pivot is not None and self._flem != self._pivot:
                if self._direction == 'long':
                    retrace_reset = (close - self._pivot) / (self._flem - self._pivot)
                    if retrace_reset > 1.2:
                        self._flem = high
                        self._flem_saved_time = None
                        self._reentry_seen = False
                        self._reentry_time = None
                        self._pivot = None
                        self._pivot_time = None
                        self._entry_triggered = False
                        self._last_3_strong = []
                        self._reset_retrace_accumulators()
                        self._prev_1m_bar = b
                        return
                else:
                    retrace_reset = (self._pivot - close) / (self._pivot - self._flem)
                    if retrace_reset > 1.2:
                        self._flem = low
                        self._flem_saved_time = None
                        self._reentry_seen = False
                        self._reentry_time = None
                        self._pivot = None
                        self._pivot_time = None
                        self._entry_triggered = False
                        self._last_3_strong = []
                        self._reset_retrace_accumulators()
                        self._prev_1m_bar = b
                        return

            if self._direction == 'long' and low < self._zone_low:
                print(f"[BNR13HMX20] INVALIDATED long (1m low {low:.2f} < zone_low {self._zone_low:.2f})")
                self._reset_candidate()
                if close < self._zone_low:
                    self._direction = 'short'
                    self._candidate_active = True
                    print(f"[BNR13HMX20] Breakout SHORT @ {close:.2f}")
                self._prev_1m_bar = b
                return
            if self._direction == 'short' and high > self._zone_high:
                print(f"[BNR13HMX20] INVALIDATED short (1m high {high:.2f} > zone_high {self._zone_high:.2f})")
                self._reset_candidate()
                if close > self._zone_high:
                    self._direction = 'long'
                    self._candidate_active = True
                    print(f"[BNR13HMX20] Breakout LONG  @ {close:.2f}")
                self._prev_1m_bar = b
                return

            if self._direction == 'long':
                if self._pivot is None or low < self._pivot:
                    self._pivot = low
                    self._pivot_time = ts
            else:
                if self._pivot is None or high > self._pivot:
                    self._pivot = high
                    self._pivot_time = ts

            body = abs(close - b['open'])
            strong      = (atr is not None) and (body >= 0.8 * atr)
            very_strong = (atr is not None) and (body >= 1.3 * atr)
            if not self._allow_counter:
                if self._direction == 'long'  and close <= b['open']: strong = very_strong = False
                if self._direction == 'short' and close >= b['open']: strong = very_strong = False
            self._last_3_strong.append(bool(strong))
            self._last_3_strong = self._last_3_strong[-3:]
            strong_count = sum(self._last_3_strong)

            retrace = None
            if self._flem is not None and self._pivot is not None and self._flem != self._pivot:
                if self._direction == 'long':
                    retrace = (close - self._pivot) / (self._flem - self._pivot)
                else:
                    retrace = (self._pivot - close) / (self._pivot - self._flem)

            # ML score (merged HOD/LOD model)
            ml_allows_entry = False
            score = None
            if self._model is not None and self._last_30s_close is not None:
                if self._flem is not None and self._pivot is not None and self._flem != self._pivot:
                    if self._direction == 'long':
                        retrace_ml = (self._last_30s_close - self._pivot) / (self._flem - self._pivot)
                    else:
                        retrace_ml = (self._pivot - self._last_30s_close) / (self._pivot - self._flem)
                    score = self._ml_score(retrace_ml, self._last_30s_close, ts)
                    if score is not None:
                        ml_allows_entry = score >= PWIN_THRESH

            disp = self._displacement_category()
            if retrace is not None:
                gate_ok = ((self._model is not None and ml_allows_entry) or
                           (self._model is None and (strong_count >= 2 or very_strong)))
                if gate_ok:
                    if self._direction == 'long':
                        stop   = self._pivot
                        risk   = close - stop
                    else:
                        stop   = self._pivot
                        risk   = stop - close

                    if risk < MIN_RISK_PTS:
                        self._prev_1m_bar = b
                        return
                    risk_per_contract = risk * MNQ_DOLLARS_PER_POINT
                    contracts = int(MAX_RISK_DOLLARS // risk_per_contract) if risk_per_contract > 0 else 0
                    if contracts < 1 or contracts > MAX_CONTRACTS:
                        self._prev_1m_bar = b
                        return

                    self._scale_out_stage = 0
                    self._scale_out_plan = []
                    self._scale_out_active = False

                    self._entry_triggered = True
                    self._in_trade = True
                    entry_price = round_to_tick(close)
                    stop = round_to_tick(stop)
                    if self._direction == 'long':
                        risk = entry_price - stop
                        target_distance = max(1.2 * risk, MIN_PROFIT_TARGET_PTS)
                        target = round_to_tick(entry_price + target_distance)
                    else:
                        risk = stop - entry_price
                        target_distance = max(1.2 * risk, MIN_PROFIT_TARGET_PTS)
                        target = round_to_tick(entry_price - target_distance)

                    self._stop_price  = stop
                    self._target_price = target

                    mirror_mode = contracts >= FLIP_SIDE_THRESHOLD_CONTRACTS
                    trade = LiveTrade(
                        day=str(today),
                        direction=mirrored_direction(self._direction, mirror_mode),
                        virtual_direction=self._direction,
                        entry_time=ts,
                        entry_price=entry_price,
                        contracts=contracts,
                        risk_dollars=float(risk * MNQ_DOLLARS_PER_POINT * contracts),
                        pivot=self._pivot,
                        flem=self._flem,
                        reentry_time=self._reentry_time,
                        pivot_time=self._pivot_time,
                        risk=float(risk),
                        target=float(target),
                        retrace_at_entry=float(retrace),
                        displacement=disp,
                        pwin_score=float(score) if score is not None else 0.0,
                        mirror_mode=mirror_mode,
                    )
                    self._trades.append(trade)

                    score_val = self._ml_score(
                        (self._last_30s_close - self._pivot) / (self._flem - self._pivot)
                        if self._direction == 'long' else
                        (self._pivot - self._last_30s_close) / (self._pivot - self._flem),
                        self._last_30s_close,
                        ts,
                    ) if (self._last_30s_close is not None and self._flem != self._pivot) else None
                    ml_info = f"  ml_score={score_val:.3f}" if score_val is not None else ""
                    print(f"[BNR13HMX20] ★ ENTRY {self._direction.upper()}  {ts.strftime('%H:%M:%S')} "
                          f"@ {close:.2f} | stop={stop:.2f}  risk={risk:.1f}pts  "
                          f"contracts={contracts}  retrace={retrace:.2%}  disp={disp}{ml_info}")
                    self._prev_1m_bar = b
                    return

        # Trade management
        if self._in_trade and self._trades and self._trades[-1].outcome is None:
            trade = self._trades[-1]
            if trade.entry_time == ts:
                self._prev_1m_bar = b
                return

            entry_price = trade.entry_price

            if ts.hour >= SESSION_CLOSE_H and ts.minute >= SESSION_CLOSE_M:
                self._close_trade(trade, ts, close, 'forced_close', 'forced_close')
                self._prev_1m_bar = b
                return

            if self._direction == 'long':
                if high >= self._target_price:
                    self._handle_target_hit(trade, ts, 'long')
                elif close <= self._stop_price:
                    self._close_trade(trade, ts, close, 'stop', 'loss')
            else:
                if low <= self._target_price:
                    self._handle_target_hit(trade, ts, 'short')
                elif close >= self._stop_price:
                    self._close_trade(trade, ts, close, 'stop', 'loss')

        self._prev_1m_bar = b

    def _handle_target_hit(self, trade: LiveTrade, ts: datetime, direction: str):
        exit_price = self._target_price
        trade.exit_time   = ts
        trade.exit_price  = exit_price
        trade.exit_reason = 'target'
        trade.outcome     = 'win'
        trade.stop_price  = self._stop_price
        if trade.virtual_direction == 'long':
            virtual_pnl = (exit_price - trade.entry_price) * (trade.contracts or 0)
        else:
            virtual_pnl = (trade.entry_price - exit_price) * (trade.contracts or 0)
        trade.pnl = realized_pnl(virtual_pnl, trade.mirror_mode)
        pnl_dollars = trade.pnl * MNQ_DOLLARS_PER_POINT
        self._daily_pnl += pnl_dollars
        self._in_trade = False
        self._scale_out_active = False
        self._scale_out_stage = 0
        self._reset_after_close()
        print(f"[BNR13HMX20] EXIT ✓ WIN  target @ {exit_price:.2f} | "
              f"pnl={pnl_dollars:+.0f}$  daily={self._daily_pnl:+.0f}$")

    # ── persistence ───────────────────────────────────────────────────────────

    def save_trades_csv(self):
        if not self._trades:
            print("[BNR13HMX20] No trades to save.")
            return
        day_str = str(self._current_date or datetime.now(tz=ET).date())
        path = os.path.join(self._out_dir, f"bnr13_hodlod_merged_trades_{day_str}.csv")
        fields = [
            'day', 'direction', 'entry_time', 'entry_price', 'exit_time', 'exit_price',
            'pnl', 'contracts', 'risk_dollars', 'outcome', 'exit_reason',
            'pivot', 'flem', 'reentry_time', 'pivot_time', 'risk', 'target',
            'stop_price', 'retrace_at_entry', 'displacement', 'pwin_score', 'virtual_direction', 'mirror_mode',
        ]
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in self._trades:
                w.writerow({k: getattr(t, k, '') for k in fields})
        print(f"[BNR13HMX20] Saved {len(self._trades)} trade records → {path}")

    def get_combined_trades(self) -> list:
        out = []
        for t in self._trades:
            if t.outcome is None:
                continue
            out.append({
                "Engine":      "bnr_pwin13hm_execflip20",
                "Date":        str(t.day),
                "Open Time":   t.entry_time.strftime("%H:%M:%S") if t.entry_time else "",
                "Close Time":  t.exit_time.strftime("%H:%M:%S")  if t.exit_time  else "",
                "Side":        t.direction,
                "Entry Price": t.entry_price,
                "Exit Price":  t.exit_price,
                "Qty":         t.contracts,
                "PnL ($)":     round((t.pnl or 0) * MNQ_DOLLARS_PER_POINT, 2),
                "Exit Reason": t.exit_reason,
                "pwin_score":  round(t.pwin_score, 4),
            })
        return out

    def print_summary(self):
        completed = [t for t in self._trades if t.outcome is not None]
        wins = [t for t in completed if t.outcome == 'win']
        total_pnl = sum((t.pnl or 0) * MNQ_DOLLARS_PER_POINT for t in completed)
        wr = len(wins) / len(completed) * 100 if completed else 0.0
        print(f"\n[BNR13HMX20] ── Daily Summary ──")
        print(f"  Trades: {len(completed)}  WR: {wr:.1f}%  P&L: ${total_pnl:+,.0f}")
        print(f"  (paper trading — no orders placed)")
