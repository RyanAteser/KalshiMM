#!/usr/bin/env python3
"""
Kalshi KXBTC15M Mean Reversion — Validation & Backtest Script.

Runs the full 8-step pipeline from the HFT article on BTC price data:

  Step 1  Fetch 90 days of BTC/USDT 15-min OHLC from Binance (free, no auth).
          BTC log returns are a reliable proxy for near-the-money KXBTC15M
          YES price movements (delta ≈ 0.5 near the money).
  Step 2  Convert BTC closes to log returns.
  Step 3  Add lag-1 (previous bar's log return).
  Step 4  Encode direction as +1 / -1.
  Step 5  Bucket analysis — mean next-bar return by previous direction.
  Step 6  75/25 time-split out-of-sample validation.
  Step 7  Signal equity curve.
  Step 8  Win rate, compound return, annualised Sharpe.

No orders are placed. Safe to run at any time as a weekly health check.

Usage
-----
  python mean_reversion_backtest.py                  # 90-day default
  python mean_reversion_backtest.py --days 60
  python mean_reversion_backtest.py --min-edge 0.003  # stricter gate
"""

import argparse
import logging
import math

import numpy as np

from src.bar_builder import log_return
from src.kalshi_mean_reversion_bot import fetch_btc_15min_bars
from src.mean_reversion import MeanReversionValidator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("backtest")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _hr(char="=", n=64):
    print(char * n)


def _equity_curve(trade_returns: np.ndarray, width: int = 60, height: int = 12):
    cum = np.cumsum(trade_returns)
    if not len(cum):
        return
    lo, hi = float(np.min(cum)), float(np.max(cum))
    if hi == lo:
        print("  (flat equity curve)")
        return

    step = max(1, len(cum) // width)
    pts  = cum[::step][:width]

    print()
    print("  Equity curve (cumulative log-return of signal strategy):")
    print(f"  Peak  {hi:+.3f}  ({(math.exp(hi) - 1) * 100:+.1f}%)")
    print(f"  Final {cum[-1]:+.3f}  ({(math.exp(cum[-1]) - 1) * 100:+.1f}%)")
    print()

    for row in range(height, -1, -1):
        thresh = lo + (hi - lo) * row / height
        line   = "".join("*" if v >= thresh else " " for v in pts)
        if row % 3 == 0:
            print(f"  {thresh:+.4f} | {line}")
        else:
            print(f"          | {line}")

    n_bars = len(cum)
    print(f"          +-{'-' * len(pts)}")
    print(f"           0{' ' * (len(pts) // 2 - 5)}bars → {n_bars:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest(days: int = 90, min_edge: float = 0.002):
    _hr()
    print("KXBTC15M MEAN REVERSION — VALIDATION PIPELINE")
    print(f"  Source : Binance BTC/USDT 15-min OHLC")
    print(f"  Window : {days} days  |  Min OOS edge : {min_edge:.4f}")
    _hr()

    bars = fetch_btc_15min_bars(days=days)
    if len(bars) < 50:
        log.error("Only %d bars — check internet access and retry.", len(bars))
        return

    # ── Steps 2-8: validation pipeline ──────────────────────────────
    validator = MeanReversionValidator(min_edge=min_edge)
    result    = validator.validate(bars)

    _hr("-")
    print("RESULTS")
    _hr("-")
    print(f"  Total bars           : {result.n_bars:,}")
    print(f"  Win rate             : {result.win_rate:.1%}")
    pct = (math.exp(result.total_log_return) - 1) * 100
    print(f"  Total log return     : {result.total_log_return:+.4f}  ({pct:+.1f}%)")
    print(f"  Annualised Sharpe    : {result.annualized_sharpe:.2f}")
    print(f"  Edge is LIVE         : {'YES ✓' if result.edge_is_live else 'NO  ✗'}")
    print()

    for label, (down_b, up_b) in [
        ("IN-SAMPLE  (oldest 75%)", result.in_sample),
        ("OUT-OF-SAMPLE (25%)",     result.out_of_sample),
    ]:
        print(f"  {label}")
        print(
            f"    prev=DOWN → next mean {down_b.mean_return:+.5f}  "
            f"n={down_b.count:,}  → signal +1 (buy YES)"
        )
        print(
            f"    prev=UP   → next mean {up_b.mean_return:+.5f}  "
            f"n={up_b.count:,}  → signal -1 (buy NO)"
        )
        print()

    # ── Reconstruct trade returns for equity curve ───────────────────
    closes    = np.array([b.close for b in bars])
    log_rets  = np.array([log_return(closes[i - 1], closes[i]) for i in range(1, len(closes))])
    lag_rets  = log_rets[:-1]
    curr_rets = log_rets[1:]
    signals       = -np.sign(lag_rets)
    trade_returns = signals * curr_rets

    _equity_curve(trade_returns)

    # ── Per-bar economics (illustrative) ────────────────────────────
    mean_tr  = float(np.mean(trade_returns))
    net_lr   = mean_tr - min_edge   # approximate spread cost in log-return units
    bpy      = 35_040               # 96 bars/day × 365
    print()
    _hr("-")
    print("ILLUSTRATIVE PER-BAR ECONOMICS")
    _hr("-")
    print(f"  Mean log-return per trade  : {mean_tr:+.5f}")
    print(f"  Approx spread cost         : {-min_edge:.5f}")
    print(f"  Net per bar (rough)        : {net_lr:+.5f}")
    print(f"  Bars/year (96/day × 365)   : {bpy:,}")
    ann_return = (math.exp(net_lr * bpy) - 1) * 100
    print(f"  Compounded annual return   : {ann_return:+.0f}%  (illustrative, no sizing)")
    print()
    print("  Note: BTC log-return ≠ KXBTC15M YES price log-return exactly,")
    print("  but near-the-money delta ≈ 0.5 makes them closely correlated.")
    print("  Validate on live YES price data once the bot starts collecting it.")

    _hr()
    if result.edge_is_live:
        print("✓ EDGE VALIDATED — run: python mean_reversion_main.py --paper")
    else:
        print("✗ EDGE NOT VALIDATED — do not trade until OOS buckets clear threshold.")
        print(f"  Required: both OOS means ≥ {min_edge:.4f}, n ≥ 30")
    _hr()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kalshi KXBTC15M Mean Reversion Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--days",     type=int,   default=90,
                        help="Days of BTC history to analyse (default: 90)")
    parser.add_argument("--min-edge", type=float, default=0.002,
                        help="Min OOS mean per bucket to call edge live (default: 0.002)")
    args = parser.parse_args()

    run_backtest(days=args.days, min_edge=args.min_edge)
