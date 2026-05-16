"""
15-minute bar builder and log-return buffer for mean reversion signal.

Responsibilities:
  - Aggregate a trade stream (price, timestamp) into 15-minute OHLC bars.
  - Compute log returns between consecutive bar closes.
  - Switch to logit-transformed returns near market extremes (p < 0.15 or p > 0.85)
    to avoid the blow-up in log(p/(1-p)) at the bounds.
  - Maintain a rolling deque of the last N log returns for the signal layer.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

EXTREME_LO = 0.15
EXTREME_HI = 0.85


@dataclass
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_start_ts: int
    bar_end_ts: int


def _logit(p: float) -> float:
    p = max(1e-9, min(1 - 1e-9, p))
    return math.log(p / (1.0 - p))


def log_return(p_prev: float, p_curr: float) -> float:
    """
    Log return between two probability prices.
    Uses logit transform when either price is in the extreme zone
    (< EXTREME_LO or > EXTREME_HI) to keep returns finite and symmetric.
    """
    if p_prev <= 0.0 or p_curr <= 0.0:
        return 0.0
    if p_prev >= 1.0 or p_curr >= 1.0:
        return 0.0
    if min(p_prev, p_curr) < EXTREME_LO or max(p_prev, p_curr) > EXTREME_HI:
        return _logit(p_curr) - _logit(p_prev)
    return math.log(p_curr / p_prev)


class BarBuilder:
    """
    Aggregates a stream of (price, timestamp) trades into fixed-width bars
    and maintains a rolling buffer of log returns used for the lag-1 signal.

    Thread-safety: not thread-safe; call from a single thread.
    """

    def __init__(self, bar_seconds: int = 900, buffer_size: int = 4):
        self._bar_sec = bar_seconds
        self._buf: deque = deque(maxlen=buffer_size)
        self._prev_close: Optional[float] = None
        self._pending: List[tuple] = []          # [(price, ts), ...]
        self._cur_bar_start: Optional[int] = None

    # ------------------------------------------------------------------
    # Feed

    def feed_trade(self, price: float, timestamp: float) -> Optional[Bar]:
        """
        Process one trade. Returns a completed Bar when a window closes,
        None otherwise.
        """
        bar_start = int(timestamp // self._bar_sec) * self._bar_sec

        if self._cur_bar_start is None:
            self._cur_bar_start = bar_start

        if bar_start > self._cur_bar_start:
            bar = self._close_current_bar()
            self._cur_bar_start = bar_start
            self._pending = [(price, timestamp)]
            return bar

        self._pending.append((price, timestamp))
        return None

    def flush(self) -> Optional[Bar]:
        """Force-close the in-progress bar (call at bot shutdown / bar boundary)."""
        return self._close_current_bar()

    # ------------------------------------------------------------------
    # Signal accessors

    def lag_1(self) -> Optional[float]:
        """Previous bar's log return (the only feature the strategy uses)."""
        if len(self._buf) < 2:
            return None
        return self._buf[-2]

    def direction(self) -> Optional[int]:
        """Sign of lag-1: +1 if previous bar moved up, -1 if down."""
        lag = self.lag_1()
        if lag is None:
            return None
        return 1 if lag > 0 else -1

    def signal(self) -> Optional[int]:
        """Mean reversion signal: -direction. +1 → buy YES, -1 → buy NO."""
        d = self.direction()
        return None if d is None else -d

    def buffer(self) -> List[float]:
        return list(self._buf)

    # ------------------------------------------------------------------
    # Bulk load for calibration

    def load_historical(self, bars: List[Bar]):
        """
        Pre-populate the log-return buffer from a list of already-built Bars.
        Call before starting the live loop so the first signal is immediately available.
        """
        self._buf.clear()
        self._prev_close = None
        for bar in bars:
            if self._prev_close is not None:
                lr = log_return(self._prev_close, bar.close)
                self._buf.append(lr)
            self._prev_close = bar.close

    # ------------------------------------------------------------------
    # Internal

    def _close_current_bar(self) -> Optional[Bar]:
        if not self._pending:
            return None
        prices = [p for p, _ in self._pending]
        bar = Bar(
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=float(len(prices)),
            bar_start_ts=self._cur_bar_start or 0,
            bar_end_ts=(self._cur_bar_start or 0) + self._bar_sec,
        )
        if self._prev_close is not None:
            lr = log_return(self._prev_close, bar.close)
            self._buf.append(lr)
        self._prev_close = bar.close
        self._pending = []
        return bar
