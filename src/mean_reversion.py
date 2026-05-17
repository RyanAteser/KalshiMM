"""
Mean reversion validation pipeline — steps 2-8 from the HFT article.

The pipeline takes a chronological list of Bars and:
  1. Computes log returns between consecutive closes.
  2. Creates a lag-1 column (yesterday's log return).
  3. Encodes direction as +1 / -1.
  4. Groups today's returns by the previous bar's direction (bucket analysis).
  5. Splits the data 75/25 by time and repeats analysis on each half.
  6. Declares the edge "live" only when both OOS buckets clear the minimum.
  7. Computes the signal equity curve, win rate, and Sharpe for reporting.

Usage:
    validator = MeanReversionValidator()
    result    = validator.validate(bars)          # List[Bar]
    if result.edge_is_live:
        signal = bar_builder.signal()             # +1 or -1
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .bar_builder import Bar, log_return

log = logging.getLogger(__name__)

# Minimum absolute mean log return per OOS bucket to call the edge live.
_DEFAULT_MIN_EDGE = 0.002       # 0.2%
# Minimum samples in each OOS bucket — below this we have no statistical power.
_MIN_BUCKET_N = 30
# Number of 15-min bars per calendar year (96/day × 365).
_BARS_PER_YEAR = 35_040


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BucketStats:
    direction: int        # -1 (prev bar down) or +1 (prev bar up)
    count: int
    mean_return: float    # mean of TODAY's log returns in this bucket
    sum_return: float
    std_return: float


@dataclass
class ValidationResult:
    in_sample: Tuple[BucketStats, BucketStats]       # (down-bucket, up-bucket)
    out_of_sample: Tuple[BucketStats, BucketStats]
    win_rate: float
    total_log_return: float
    annualized_sharpe: float
    edge_is_live: bool
    n_bars: int


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _bucket_analysis(
    curr_returns: np.ndarray,
    lag_returns: np.ndarray,
) -> Tuple[BucketStats, BucketStats]:
    """
    Group today's log returns by the sign of yesterday's return.
    Returns (down_bucket, up_bucket) where down = prev bar was negative.
    """
    directions = np.where(lag_returns > 0, 1, -1)
    result = []
    for d in (-1, 1):
        mask = directions == d
        bucket = curr_returns[mask]
        if len(bucket) == 0:
            result.append(BucketStats(
                direction=d, count=0,
                mean_return=0.0, sum_return=0.0, std_return=0.0,
            ))
        else:
            result.append(BucketStats(
                direction=d,
                count=int(mask.sum()),
                mean_return=float(np.mean(bucket)),
                sum_return=float(np.sum(bucket)),
                std_return=float(np.std(bucket)) if len(bucket) > 1 else 0.0,
            ))
    return tuple(result)  # type: ignore[return-value]


def _check_edge(
    oos: Tuple[BucketStats, BucketStats],
    min_edge: float,
) -> bool:
    """
    Edge is live when:
      - prev-down bucket has positive mean (we bet YES up after a down bar)
      - prev-up   bucket has negative mean (we bet YES down after an up bar)
      - both buckets have enough samples
    """
    down_b, up_b = oos

    # When prev bar was down (d=-1) we bet +1 (YES up) → positive mean expected
    down_ev = down_b.mean_return
    # When prev bar was up (d=+1) we bet -1 (YES down) → mean should be negative
    up_ev = -up_b.mean_return   # negate to express as "edge in trade direction"

    down_ok = down_b.count >= _MIN_BUCKET_N and down_ev >= min_edge
    up_ok   = up_b.count   >= _MIN_BUCKET_N and up_ev   >= min_edge

    if not down_ok:
        log.info(
            "OOS prev-DOWN bucket: mean=%.4f  n=%d  (need mean>=%.4f, n>=%d)  → %s",
            down_ev, down_b.count, min_edge, _MIN_BUCKET_N,
            "OK" if down_ok else "FAIL",
        )
    if not up_ok:
        log.info(
            "OOS prev-UP   bucket: mean=%.4f  n=%d  (need mean>=%.4f, n>=%d)  → %s",
            up_ev, up_b.count, min_edge, _MIN_BUCKET_N,
            "OK" if up_ok else "FAIL",
        )

    return down_ok and up_ok


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class MeanReversionValidator:
    """
    Runs the full validation pipeline on a list of Bar objects.

    Parameters
    ----------
    min_edge : float
        Minimum mean log return per OOS bucket to declare the edge live.
    min_bars : int
        Minimum number of bars required to run any analysis.
    bars_per_year : int
        Number of bars per calendar year, used for Sharpe annualisation.
    """

    def __init__(
        self,
        min_edge: float = _DEFAULT_MIN_EDGE,
        min_bars: int = 200,
        bars_per_year: int = _BARS_PER_YEAR,
    ):
        self._min_edge = min_edge
        self._min_bars = min_bars
        self._bpy = bars_per_year

    def validate(self, bars: List[Bar]) -> ValidationResult:
        n = len(bars)
        if n < self._min_bars:
            log.warning("Only %d bars — need %d for validation", n, self._min_bars)
            return self._empty(n)

        # Step 2: log returns
        log_rets = self._log_returns(bars)   # length n-1

        # Step 3: lag-1 pairing
        curr = log_rets[1:]    # today's return
        lag  = log_rets[:-1]   # yesterday's return (lag-1)

        # Step 4+5: in-sample (oldest 75%), out-of-sample (newest 25%)
        split = int(len(curr) * 0.75)
        is_buckets  = _bucket_analysis(curr[:split], lag[:split])
        oos_buckets = _bucket_analysis(curr[split:], lag[split:])

        # Step 6: check OOS edge
        live = _check_edge(oos_buckets, self._min_edge)

        # Step 7-8: equity curve metrics on the full dataset
        signals       = -np.sign(lag)           # mean reversion: flip the sign
        trade_returns = signals * curr
        win_rate      = float(np.mean(trade_returns > 0))
        total_lr      = float(np.sum(trade_returns))
        mean_tr       = float(np.mean(trade_returns))
        std_tr        = float(np.std(trade_returns))
        sharpe = (mean_tr / std_tr * math.sqrt(self._bpy)) if std_tr > 0 else 0.0

        result = ValidationResult(
            in_sample=is_buckets,
            out_of_sample=oos_buckets,
            win_rate=win_rate,
            total_log_return=total_lr,
            annualized_sharpe=sharpe,
            edge_is_live=live,
            n_bars=n,
        )
        self._log(result)
        return result

    def oos_ev(self, result: ValidationResult, signal: int) -> float:
        """
        Return the estimated edge (mean log return in trade direction)
        for the given signal (+1 = bet YES up, -1 = bet YES down).
        Used by the live bot to gate each trade against spread cost.
        """
        down_b, up_b = result.out_of_sample
        if signal == 1:
            return down_b.mean_return     # prev-down bucket: expect positive
        return -up_b.mean_return          # prev-up   bucket: expect negative, flip sign

    # ------------------------------------------------------------------

    @staticmethod
    def _log_returns(bars: List[Bar]) -> np.ndarray:
        closes = [b.close for b in bars]
        return np.array([
            log_return(closes[i - 1], closes[i])
            for i in range(1, len(closes))
        ])

    @staticmethod
    def _empty(n: int) -> ValidationResult:
        z = BucketStats(direction=0, count=0, mean_return=0.0, sum_return=0.0, std_return=0.0)
        return ValidationResult(
            in_sample=(z, z), out_of_sample=(z, z),
            win_rate=0.0, total_log_return=0.0,
            annualized_sharpe=0.0, edge_is_live=False, n_bars=n,
        )

    def _log(self, r: ValidationResult):
        log.info("=== Mean Reversion Validation ===")
        log.info(
            "bars=%d  win_rate=%.1f%%  total_lr=%.3f (%.0f%%)  sharpe=%.2f  edge_live=%s",
            r.n_bars, r.win_rate * 100,
            r.total_log_return, (math.exp(r.total_log_return) - 1) * 100,
            r.annualized_sharpe, r.edge_is_live,
        )
        for label, (down_b, up_b) in [
            ("IS ", r.in_sample),
            ("OOS", r.out_of_sample),
        ]:
            log.info(
                "  %s prev=DOWN → mean=%+.4f n=%4d  |  prev=UP → mean=%+.4f n=%4d",
                label,
                down_b.mean_return, down_b.count,
                up_b.mean_return,   up_b.count,
            )
