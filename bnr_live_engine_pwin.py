#!/usr/bin/env python3
"""
BNR Live Trading Engine — real-time port of bnr_deterministic_engine.py

Accepts bars via the same interface as the existing live engines:
    engine.on_bar_1m(bar)   # bar: {'1': time_ms, '2': O, '3': H, '4': L, '5': C, '6': V}
    engine.on_bar_30s(bar)  # bar: {'time_ms': ms, 'open': O, 'high': H, 'low': L, 'close': C, 'volume': V}

Strategy logic is identical to bnr_deterministic_engine.py:
  - Zone from 9:30 1m bar
  - Breakout → FLEM → Retest/Pivot → Retrace → ML entry validation
  - Scale-out plan based on contract size
  - Stops, targets, 30-min rule, 1R trail, direction flip
  - Session: 09:30–12:00 ET
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
MODEL_PATH   = "/Users/radhikaarora/Documents/Trading ML/ML V2/entry_model_pwin.joblib"
PWIN_THRESH  = 0.50
ML_FEATURES  = [
    'retrace', 'pivot_flem_dist', 'time_since_pivot_sec',
    'body_last', 'body_sum', 'body_mean', 'in_dir_ratio',
    'max_in_dir_run', 'bars_since_pivot', 'zone_over_range', 'pivot_over_range',
]
MNQ_DOLLARS_PER_POINT = 2.0
MAX_RISK_DOLLARS = 300.0
MIN_CONTRACTS = 3
MAX_CONTRACTS = 28          # skip if >= 29

SESSION_OPEN_H,  SESSION_OPEN_M  = 9, 30
SESSION_CLOSE_H, SESSION_CLOSE_M = 12, 0

DISP_HIGH_ZONE_THRESHOLD  = 0.10568226033342312
DISP_HIGH_PIVOT_THRESHOLD = 0.3795379537953795
DISP_LOW_ZONE_THRESHOLD   = 0.0848692546366965
DISP_LOW_PIVOT_THRESHOLD  = 0.23516193082722905


# ─── trade record ─────────────────────────────────────────────────────────────
@dataclass
class LiveTrade:
    day: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None           # in points (multiply by 2 for $)
    contracts: Optional[int] = None
    risk_dollars: Optional[float] = None
    outcome: Optional[str] = None         # 'win' | 'loss' | 'forced_close'
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


# ─── rolling ATR (14-period Wilder smoothing) ─────────────────────────────────
class RollingATR:
    def __init__(self, period: int = 14):
        self._period = period
        self._bars: list[dict] = []   # {close, high, low}
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
            # Simple average until we have enough bars
            self._atr = sum(b['tr'] for b in self._bars) / n
        else:
            # Wilder smoothing
            if n == self._period + 1:
                # Seed with simple average of first 14 TRs
                self._atr = sum(b['tr'] for b in self._bars[:self._period]) / self._period
            self._atr = (self._atr * (self._period - 1) + tr) / self._period

        return self._atr

    @property
    def value(self) -> Optional[float]:
        return self._atr


# ─── main engine ──────────────────────────────────────────────────────────────
class BNRLiveEngine:
    """
    Real-time BNR deterministic strategy engine.

    Drop-in replacement for the existing TradingEngine class — call
    on_bar_1m() and on_bar_30s() with bars as they arrive from stream_test.py.

    Signals are printed to stdout as [BNR] lines. Trades are saved to CSV
    on shutdown via save_trades_csv() / print_summary().
    """

    def __init__(self, out_dir: str = "output/bnr_live",
                 allow_counter_candle: bool = True):
        self._out_dir = out_dir
        self._allow_counter = allow_counter_candle
        os.makedirs(out_dir, exist_ok=True)

        # Load ML model
        if os.path.exists(MODEL_PATH):
            self._model = joblib.load(MODEL_PATH)
            self._features = ML_FEATURES
            print(f"[BNR] ML model loaded ({len(self._features)} features, thresh={PWIN_THRESH})")
        else:
            self._model = None
            self._features = None
            print("[BNR] ML model not found — strong-candle fallback active")

        # Persistent state
        self._atr_calc = RollingATR(14)
        self._trades: list[LiveTrade] = []
        self._daily_pnl: float = 0.0

        # Per-session state (reset on new day)
        self._current_date: Optional[date] = None
        self._zone_high: Optional[float] = None
        self._zone_low: Optional[float] = None

        # Candidate / setup state
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

        # ML / retrace accumulators (since pivot_time)
        self._body_sum_30s: float = 0.0
        self._body_count_30s: int = 0
        self._in_dir_count_30s: int = 0
        self._max_run_30s: int = 0
        self._run_30s: int = 0
        self._last_body_30s: Optional[float] = None
        self._last_30s_close: Optional[float] = None

        # Trade-in-progress state
        self._in_trade: bool = False
        self._stop_price: Optional[float] = None
        self._target_price: Optional[float] = None
        self._scale_out_active: bool = False
        self._scale_out_stage: int = 0
        self._scale_out_plan: list[tuple[int, float]] = []

        # Rolling 1m bars for day range calculation (session bars only)
        self._session_1m_bars: list[dict] = []   # {ts, open, high, low, close}
        self._prev_1m_bar: Optional[dict] = None  # previous bar for color check

    # ── session helpers ───────────────────────────────────────────────────────

    def _reset_session(self):
        """Reset all strategy state at the start of a new trading day."""
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
        """After any trade close, reset pivot/reentry state but keep flem intact.
        This allows a new retest/pivot to form before re-entering the same setup.
        """
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
        after_open  = (h > SESSION_OPEN_H)  or (h == SESSION_OPEN_H  and m >= SESSION_OPEN_M)
        before_close = (h < SESSION_CLOSE_H) or (h == SESSION_CLOSE_H and m == 0)
        return after_open and before_close

    def _session_open_ts(self, ts: datetime) -> datetime:
        return ts.replace(hour=SESSION_OPEN_H, minute=SESSION_OPEN_M,
                          second=0, microsecond=0)

    # ── bar parsing ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_1m(bar: dict) -> dict:
        """Convert Schwab CHART_FUTURES bar to internal format."""
        # field '1' is bar *open* time in epoch-ms (same convention as live_trader.py);
        # bar closes 1 minute later.
        open_ms  = int(bar['1'])
        open_ts  = datetime.fromtimestamp(open_ms / 1000, tz=ET)
        close_ts = open_ts + timedelta(minutes=1)
        return {
            'open_ts': open_ts,   # when bar opened (= bar['1'])
            'close_ts': close_ts, # when bar closed (= bar['1'] + 1m)
            'open':  float(bar['2']),
            'high':  float(bar['3']),
            'low':   float(bar['4']),
            'close': float(bar['5']),
            'volume': int(bar['6']),
        }

    @staticmethod
    def _parse_30s(bar: dict) -> dict:
        """Convert aggregated 30s bar to internal format."""
        ts = datetime.fromtimestamp(bar['time_ms'] / 1000, tz=ET)
        return {
            'ts':    ts,
            'open':  float(bar['open']),
            'high':  float(bar['high']),
            'low':   float(bar['low']),
            'close': float(bar['close']),
        }

    # ── day-range helper ──────────────────────────────────────────────────────

    def _day_range(self) -> float:
        if not self._session_1m_bars:
            return 0.0
        return max(b['high'] for b in self._session_1m_bars) - \
               min(b['low']  for b in self._session_1m_bars)

    # ── displacement category ─────────────────────────────────────────────────

    def _displacement_category(self) -> str:
        if self._flem is None or self._pivot is None or self._zone_high is None:
            return 'medium'
        dr = self._day_range()
        if dr == 0:
            return 'medium'

        pivot_flem_dist = abs(self._flem - self._pivot)

        if self._direction == 'long':
            zone_price = self._zone_high
            dist_zone  = self._flem - zone_price
        else:
            zone_price = self._zone_low
            dist_zone  = zone_price - self._flem

        zone_over_range  = dist_zone / dr if dist_zone is not None else 0.0
        pivot_over_range = pivot_flem_dist / dr

        if zone_over_range  >= DISP_HIGH_ZONE_THRESHOLD and \
           pivot_over_range >= DISP_HIGH_PIVOT_THRESHOLD:
            return 'high'
        elif zone_over_range  <= DISP_LOW_ZONE_THRESHOLD and \
             pivot_over_range <= DISP_LOW_PIVOT_THRESHOLD:
            return 'low'
        return 'medium'

    # ── ML scoring ────────────────────────────────────────────────────────────

    def _ml_score(self, retrace_ml: float) -> Optional[float]:
        if self._model is None or self._body_count_30s == 0:
            return None
        if self._pivot is None or self._flem is None or self._flem == self._pivot:
            return None

        import pandas as pd
        pivot_flem_dist = abs(self._flem - self._pivot)
        dr = self._day_range()

        if self._direction == 'long':
            dist_zone = self._flem - self._zone_high
        else:
            dist_zone = self._zone_low - self._flem

        zone_over_range  = dist_zone / dr if dr > 0 else 0.0
        pivot_over_range = pivot_flem_dist / dr if dr > 0 else 0.0

        body_mean    = self._body_sum_30s / self._body_count_30s
        in_dir_ratio = self._in_dir_count_30s / self._body_count_30s
        time_since_pivot = (self._pivot_time and
                            (datetime.now(tz=ET) - self._pivot_time).total_seconds()) or 0.0

        features = {
            'retrace':             retrace_ml,
            'pivot_flem_dist':     pivot_flem_dist,
            'time_since_pivot_sec': time_since_pivot,
            'body_last':           self._last_body_30s or 0.0,
            'body_sum':            self._body_sum_30s,
            'body_mean':           body_mean,
            'in_dir_ratio':        in_dir_ratio,
            'max_in_dir_run':      self._max_run_30s,
            'bars_since_pivot':    self._body_count_30s,
            'zone_over_range':     zone_over_range,
            'pivot_over_range':    pivot_over_range,
        }
        x = pd.DataFrame([features])[self._features].fillna(0.0)
        return float(self._model.predict_proba(x)[0][1])

    # ── trade exit helper ─────────────────────────────────────────────────────

    def _close_trade(self, trade: LiveTrade, exit_ts: datetime,
                     exit_price: float, exit_reason: str, outcome: str):
        trade.exit_time  = exit_ts
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.outcome    = outcome
        trade.stop_price = self._stop_price

        if trade.direction == 'long':
            trade.pnl = (exit_price - trade.entry_price) * (trade.contracts or 0)
        else:
            trade.pnl = (trade.entry_price - exit_price) * (trade.contracts or 0)

        pnl_dollars = trade.pnl * MNQ_DOLLARS_PER_POINT
        self._daily_pnl += pnl_dollars

        tag = '✓ WIN ' if outcome == 'win' else '✗ LOSS'
        print(f"[BNR] EXIT {tag}  {exit_ts.strftime('%H:%M:%S')}  {exit_reason} | "
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
            return  # session not initialized yet

        # ── invalidation: 30s close through opposite zone ─────────────────────
        if self._candidate_active and not self._in_trade:
            if self._direction == 'long' and close < self._zone_low:
                print(f"[BNR] INVALIDATED long (30s close {close:.2f} < zone_low {self._zone_low:.2f})")
                self._reset_candidate()
                # same bar closed below zone → immediately start SHORT
                if close < self._zone_low:
                    self._direction = 'short'
                    self._candidate_active = True
                    print(f"[BNR] Breakout SHORT @ {close:.2f}  (zone_low={self._zone_low:.2f}) [30s]")
                return
            if self._direction == 'short' and close > self._zone_high:
                print(f"[BNR] INVALIDATED short (30s close {close:.2f} > zone_high {self._zone_high:.2f})")
                self._reset_candidate()
                # same bar closed above zone → immediately start LONG
                if close > self._zone_high:
                    self._direction = 'long'
                    self._candidate_active = True
                    print(f"[BNR] Breakout LONG  @ {close:.2f}  (zone_high={self._zone_high:.2f}) [30s]")
                return

        # ── initial breakout on 30s close (no active candidate) ───────────────
        if not self._candidate_active:
            if close > self._zone_high:
                self._direction = 'long'
                self._candidate_active = True
                print(f"[BNR] Breakout LONG  @ {close:.2f}  (zone_high={self._zone_high:.2f}) [30s]")
            elif close < self._zone_low:
                self._direction = 'short'
                self._candidate_active = True
                print(f"[BNR] Breakout SHORT @ {close:.2f}  (zone_low={self._zone_low:.2f}) [30s]")

        # ── accumulate retrace window (30s bars since pivot) ──────────────────
        if self._reentry_seen and not self._entry_triggered and self._pivot_time is not None:
            if ts >= self._pivot_time:
                body = abs(close - open_)
                self._last_body_30s  = body
                self._last_30s_close = close
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
        ts = b['close_ts']   # event_time = bar close
        open_ts = b['open_ts']

        # ── session / day boundary ────────────────────────────────────────────
        today = ts.astimezone(ET).date()
        if today != self._current_date:
            # Force-close any open trade before rolling to next day
            if self._in_trade and self._trades and self._trades[-1].outcome is None:
                t = self._trades[-1]
                self._close_trade(t, ts, b['close'], 'forced_close', 'forced_close')
            if self._current_date is not None:
                self.save_trades_csv()
                self.print_summary()
            self._current_date = today
            self._reset_session()
            print(f"\n[BNR] ── New session: {today} ──")

        # ── ignore bars outside session ───────────────────────────────────────
        if not self._is_in_session(open_ts):
            return

        # ── zone bar (09:30 1m bar) ───────────────────────────────────────────
        session_open = self._session_open_ts(open_ts)
        if open_ts == session_open and self._zone_high is None:
            self._zone_high = b['high']
            self._zone_low  = b['low']
            print(f"[BNR] Zone: H={self._zone_high:.2f}  L={self._zone_low:.2f}")

        # Always update rolling session bars and ATR
        self._session_1m_bars.append(b)
        atr = self._atr_calc.update(b['high'], b['low'], b['close'])

        # Wait for zone to be set (zone bar fires at 09:31 event_time = 09:30 + 1m)
        if self._zone_high is None:
            return

        close = b['close']
        high  = b['high']
        low   = b['low']

        # ── direction flip: candidate long, but close breaks below zone ───────
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
            print(f"[BNR] Direction flip → SHORT @ {close:.2f}  (zone_low={self._zone_low:.2f})")

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
            print(f"[BNR] Direction flip → LONG  @ {close:.2f}  (zone_high={self._zone_high:.2f})")

        # ── initial breakout ──────────────────────────────────────────────────
        if not self._candidate_active:
            if close > self._zone_high:
                self._direction = 'long'
                self._candidate_active = True
                print(f"[BNR] Breakout LONG  @ {close:.2f}  (zone_high={self._zone_high:.2f})")
            elif close < self._zone_low:
                self._direction = 'short'
                self._candidate_active = True
                print(f"[BNR] Breakout SHORT @ {close:.2f}  (zone_low={self._zone_low:.2f})")
            else:
                self._prev_1m_bar = b
                return

        # ── FLEM tracking & retest detection ──────────────────────────────────
        if self._candidate_active and not self._reentry_seen:
            if self._direction == 'long':
                self._flem = high if self._flem is None else max(self._flem, high)
                # Retest: bearish candle that touches back into zone
                if close < b['open'] and low <= self._zone_high and high >= self._zone_low:
                    self._reentry_seen = True
                    self._reentry_time = ts
                    self._flem_saved_time = ts
                    self._pivot = low
                    self._pivot_time = ts
                    self._reset_retrace_accumulators()
                    print(f"[BNR] Retest/Pivot LONG  pivot={self._pivot:.2f}  flem={self._flem:.2f}")
            else:
                self._flem = low if self._flem is None else min(self._flem, low)
                if close > b['open'] and low <= self._zone_high and high >= self._zone_low:
                    self._reentry_seen = True
                    self._reentry_time = ts
                    self._flem_saved_time = ts
                    self._pivot = high
                    self._pivot_time = ts
                    self._reset_retrace_accumulators()
                    print(f"[BNR] Retest/Pivot SHORT pivot={self._pivot:.2f}  flem={self._flem:.2f}")

        # ── after retest: pivot update, retrace, entry ────────────────────────
        if self._reentry_seen and not self._entry_triggered:
            # Retrace-reset: if price blows past FLEM, reset setup
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

            # Far-side invalidation
            if self._direction == 'long' and low < self._zone_low:
                print(f"[BNR] INVALIDATED long (1m low {low:.2f} < zone_low {self._zone_low:.2f})")
                self._reset_candidate()
                if close < self._zone_low:
                    # Bar closed below zone — immediately start SHORT candidate
                    self._direction = 'short'
                    self._candidate_active = True
                    print(f"[BNR] Breakout SHORT @ {close:.2f}  (zone_low={self._zone_low:.2f})")
                self._prev_1m_bar = b
                return
            if self._direction == 'short' and high > self._zone_high:
                print(f"[BNR] INVALIDATED short (1m high {high:.2f} > zone_high {self._zone_high:.2f})")
                self._reset_candidate()
                if close > self._zone_high:
                    # Bar closed above zone — immediately start LONG candidate
                    self._direction = 'long'
                    self._candidate_active = True
                    print(f"[BNR] Breakout LONG  @ {close:.2f}  (zone_high={self._zone_high:.2f})")
                self._prev_1m_bar = b
                return

            # Update pivot (worst point before entry)
            if self._direction == 'long':
                if self._pivot is None or low < self._pivot:
                    self._pivot = low
                    self._pivot_time = ts
            else:
                if self._pivot is None or high > self._pivot:
                    self._pivot = high
                    self._pivot_time = ts

            # Strong candle tracking
            body = abs(close - b['open'])
            strong = (atr is not None) and (body >= 0.8 * atr)
            very_strong = (atr is not None) and (body >= 1.3 * atr)
            if not self._allow_counter:
                if self._direction == 'long'  and close <= b['open']: strong = very_strong = False
                if self._direction == 'short' and close >= b['open']: strong = very_strong = False
            self._last_3_strong.append(bool(strong))
            self._last_3_strong = self._last_3_strong[-3:]
            strong_count = sum(self._last_3_strong)

            # Retracement measure
            retrace = None
            if self._flem is not None and self._pivot is not None and self._flem != self._pivot:
                if self._direction == 'long':
                    retrace = (close - self._pivot) / (self._flem - self._pivot)
                else:
                    retrace = (self._pivot - close) / (self._pivot - self._flem)

            # ML score
            ml_allows_entry = False
            if self._model is not None and self._last_30s_close is not None:
                if self._flem is not None and self._pivot is not None and self._flem != self._pivot:
                    if self._direction == 'long':
                        retrace_ml = (self._last_30s_close - self._pivot) / (self._flem - self._pivot)
                    else:
                        retrace_ml = (self._pivot - self._last_30s_close) / (self._pivot - self._flem)
                    score = self._ml_score(retrace_ml)
                    if score is not None:
                        ml_allows_entry = score >= PWIN_THRESH

            # Displacement & retrace bounds
            disp = self._displacement_category()
            min_retrace = 0.45 if disp == 'high' else 0.35
            max_retrace = 0.90 if disp == 'high' else 1.20

            # Prev-bar color check
            prev_color_ok = True
            if self._prev_1m_bar is not None and not self._allow_counter:
                pc = self._prev_1m_bar['close']
                po = self._prev_1m_bar['open']
                if self._direction == 'long':
                    prev_color_ok = pc > po
                else:
                    prev_color_ok = pc < po

            # Entry gate
            if (retrace is not None and
                    min_retrace <= retrace <= max_retrace and
                    prev_color_ok):
                gate_ok = ((self._model is not None and ml_allows_entry) or
                           (self._model is None and (strong_count >= 2 or very_strong)))
                if gate_ok:
                    # Position sizing
                    if self._direction == 'long':
                        stop = self._pivot
                        risk = close - stop
                        target = close + 1.2 * risk
                    else:
                        stop = self._pivot
                        risk = stop - close
                        target = close - 1.2 * risk

                    risk_per_contract = risk * MNQ_DOLLARS_PER_POINT
                    contracts = int(MAX_RISK_DOLLARS // risk_per_contract) if risk_per_contract > 0 else 0
                    if contracts < MIN_CONTRACTS or contracts > MAX_CONTRACTS:
                        self._prev_1m_bar = b
                        return  # skip: size out of bounds

                    # Scale-out plan
                    self._scale_out_stage = 0
                    if contracts >= 13:
                        q = contracts // 4
                        rem = contracts - 3 * q
                        self._scale_out_plan = [(q, 1.2), (q, 2.5), (q, 4.0), (rem, 6.5)]
                    elif contracts == 1:
                        # Single contract: no scale-out
                        self._scale_out_plan = []
                    elif contracts == 2:
                        # Two contracts: scale at 1.5R and 3.0R
                        self._scale_out_plan = [(1, 1.5), (1, 3.0)]
                    elif contracts == 3:
                        # Three contracts: 1 @ 1.2R, 1 @ 2.0R, 1 @ 3.0R
                        self._scale_out_plan = [(1, 1.2), (1, 2.0), (1, 3.0)]
                    else:
                        q1 = contracts // 4
                        q2 = int(contracts * 0.35)
                        q3 = contracts - q1 - q2
                        self._scale_out_plan = [(q1, 1.2), (q2, 2.0), (q3, 3.0)]
                    self._scale_out_active = (len(self._scale_out_plan) > 1 and
                                               all(q > 0 for q, _ in self._scale_out_plan))

                    self._entry_triggered = True
                    self._in_trade = True
                    self._stop_price  = stop
                    self._target_price = target

                    trade = LiveTrade(
                        day=str(today),
                        direction=self._direction,
                        entry_time=ts,
                        entry_price=close,
                        contracts=contracts,
                        risk_dollars=float(risk_per_contract * contracts),
                        pivot=self._pivot,
                        flem=self._flem,
                        reentry_time=self._reentry_time,
                        pivot_time=self._pivot_time,
                        risk=float(risk),
                        target=float(target),
                        retrace_at_entry=float(retrace),
                        displacement=disp,
                    )
                    self._trades.append(trade)

                    score_val = self._ml_score(
                        (self._last_30s_close - self._pivot) / (self._flem - self._pivot)
                        if self._direction == 'long' else
                        (self._pivot - self._last_30s_close) / (self._pivot - self._flem)
                    ) if (self._last_30s_close is not None and self._flem != self._pivot) else None
                    ml_info = f"  ml_score={score_val:.3f}" if score_val is not None else ""
                    print(f"[BNR] ★ ENTRY {self._direction.upper()}  {ts.strftime('%H:%M:%S')} @ {close:.2f} | "
                          f"stop={stop:.2f}  risk={risk:.1f}pts ({risk_per_contract * contracts:.0f}$) "
                          f"contracts={contracts}  retrace={retrace:.2%}  disp={disp}{ml_info}")
                    self._prev_1m_bar = b
                    return

        # ── trade management (exits) ──────────────────────────────────────────
        if self._in_trade and self._trades and self._trades[-1].outcome is None:
            trade = self._trades[-1]

            # Skip same-bar exit (entry bar)
            if trade.entry_time == ts:
                self._prev_1m_bar = b
                return

            entry_price = trade.entry_price
            entry_risk  = trade.risk
            elapsed = ts - trade.entry_time

            # 30-minute rule
            if elapsed >= timedelta(minutes=30):
                if self._direction == 'long':
                    if close > entry_price:
                        self._stop_price = max(self._stop_price, entry_price)
                    elif self._target_price > entry_price:
                        self._target_price = entry_price
                else:
                    if close < entry_price:
                        self._stop_price = min(self._stop_price, entry_price)
                    elif self._target_price < entry_price:
                        self._target_price = entry_price

            # 1R trail
            if self._direction == 'long':
                if high >= entry_price + entry_risk:
                    self._stop_price = max(self._stop_price, entry_price - 0.5 * entry_risk)
            else:
                if low <= entry_price - entry_risk:
                    self._stop_price = min(self._stop_price, entry_price + 0.5 * entry_risk)

            # Session end force-close (12:00 ET)
            if ts.hour >= SESSION_CLOSE_H and ts.minute >= SESSION_CLOSE_M:
                self._close_trade(trade, ts, close, 'forced_close', 'forced_close')
                self._prev_1m_bar = b
                return

            if self._direction == 'long':
                # Stop: close below stop
                if close <= self._stop_price:
                    self._close_trade(trade, ts, close, 'stop', 'loss')
                # Target hit
                elif high >= self._target_price:
                    self._handle_target_hit(trade, ts, 'long')
            else:
                # Stop: close above stop
                if close >= self._stop_price:
                    self._close_trade(trade, ts, close, 'stop', 'loss')
                # Target hit
                elif low <= self._target_price:
                    self._handle_target_hit(trade, ts, 'short')

        self._prev_1m_bar = b

    def _handle_target_hit(self, trade: LiveTrade, ts: datetime, direction: str):
        """Handle target hit — scale out or final exit."""
        if self._scale_out_active and self._scale_out_stage < len(self._scale_out_plan) - 1:
            # Intermediate scale-out leg
            leg_q, _ = self._scale_out_plan[self._scale_out_stage]
            next_q, next_r = self._scale_out_plan[self._scale_out_stage + 1]
            leg_price = self._target_price

            leg_pnl = ((leg_price - trade.entry_price) if direction == 'long'
                       else (trade.entry_price - leg_price)) * leg_q
            leg_pnl_dollars = leg_pnl * MNQ_DOLLARS_PER_POINT
            self._daily_pnl += leg_pnl_dollars

            # Create a completed leg record
            leg = LiveTrade(
                day=trade.day, direction=trade.direction,
                entry_time=trade.entry_time, entry_price=trade.entry_price,
                exit_time=ts, exit_price=leg_price,
                pnl=leg_pnl, contracts=leg_q,
                risk_dollars=float(trade.risk * MNQ_DOLLARS_PER_POINT * leg_q),
                outcome='win',
                exit_reason=f'target_scale{self._scale_out_stage + 1}',
                pivot=trade.pivot, flem=trade.flem,
                reentry_time=trade.reentry_time, pivot_time=trade.pivot_time,
                risk=trade.risk, target=trade.target,
                stop_price=self._stop_price,
                retrace_at_entry=trade.retrace_at_entry,
                displacement=trade.displacement,
            )
            self._trades[-1] = leg

            print(f"[BNR] SCALE OUT leg {self._scale_out_stage + 1} @ {leg_price:.2f} "
                  f"({leg_q} contracts)  pnl={leg_pnl_dollars:+.0f}$  daily={self._daily_pnl:+.0f}$")

            self._scale_out_stage += 1
            if direction == 'long':
                self._target_price = trade.entry_price + next_r * trade.risk
            else:
                self._target_price = trade.entry_price - next_r * trade.risk

            # Append open remaining-leg record
            remaining = LiveTrade(
                day=trade.day, direction=trade.direction,
                entry_time=trade.entry_time, entry_price=trade.entry_price,
                contracts=next_q,
                risk_dollars=float(trade.risk * MNQ_DOLLARS_PER_POINT * next_q),
                pivot=trade.pivot, flem=trade.flem,
                reentry_time=trade.reentry_time, pivot_time=trade.pivot_time,
                risk=trade.risk, target=float(self._target_price),
                retrace_at_entry=trade.retrace_at_entry,
                displacement=trade.displacement,
            )
            self._trades.append(remaining)
            self._in_trade = True
        else:
            # Final exit
            exit_price = self._target_price
            trade.exit_time  = ts
            trade.exit_price = exit_price
            trade.exit_reason = 'target'
            trade.outcome = 'win'
            trade.stop_price = self._stop_price
            if direction == 'long':
                trade.pnl = (exit_price - trade.entry_price) * (trade.contracts or 0)
            else:
                trade.pnl = (trade.entry_price - exit_price) * (trade.contracts or 0)
            pnl_dollars = trade.pnl * MNQ_DOLLARS_PER_POINT
            self._daily_pnl += pnl_dollars
            self._in_trade = False
            self._scale_out_active = False
            self._scale_out_stage = 0
            print(f"[BNR] EXIT ✓ WIN  target @ {exit_price:.2f} | "
                  f"pnl={pnl_dollars:+.0f}$  daily={self._daily_pnl:+.0f}$")

    # ── persistence ───────────────────────────────────────────────────────────

    def save_trades_csv(self):
        if not self._trades:
            print("[BNR] No trades to save.")
            return
        day_str = str(self._current_date or datetime.now(tz=ET).date())
        path = os.path.join(self._out_dir, f"bnr_trades_{day_str}.csv")
        fields = [
            'day', 'direction', 'entry_time', 'entry_price', 'exit_time', 'exit_price',
            'pnl', 'contracts', 'risk_dollars', 'outcome', 'exit_reason',
            'pivot', 'flem', 'reentry_time', 'pivot_time', 'risk', 'target',
            'stop_price', 'retrace_at_entry', 'displacement',
        ]
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in self._trades:
                w.writerow({k: getattr(t, k, '') for k in fields})
        print(f"[BNR] Saved {len(self._trades)} trade records → {path}")

    def get_combined_trades(self) -> list:
        """Return completed trades as normalized dicts matching the shared combined-CSV schema."""
        out = []
        for t in self._trades:
            if t.outcome is None:
                continue  # skip open / incomplete trades
            out.append({
                "Engine":       "bnr_pwin",
                "Date":         str(t.day),
                "Open Time":    t.entry_time.strftime("%H:%M:%S") if t.entry_time else "",
                "Close Time":   t.exit_time.strftime("%H:%M:%S")  if t.exit_time  else "",
                "Side":         t.direction,
                "Entry Price":  t.entry_price,
                "Exit Price":   t.exit_price,
                "Qty":          t.contracts,
                "PnL ($)":      round((t.pnl or 0) * MNQ_DOLLARS_PER_POINT, 2),
                "Exit Reason":  t.exit_reason,
            })
        return out

    def print_summary(self):
        completed = [t for t in self._trades if t.outcome is not None]
        wins  = [t for t in completed if t.outcome == 'win']
        total_pnl = sum((t.pnl or 0) * MNQ_DOLLARS_PER_POINT for t in completed)
        wr = len(wins) / len(completed) * 100 if completed else 0.0
        print(f"\n[BNR] ── Daily Summary ──")
        print(f"  Trades: {len(completed)}  WR: {wr:.1f}%  P&L: ${total_pnl:+,.0f}")
        print(f"  (paper trading — no orders placed)")
