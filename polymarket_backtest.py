#!/usr/bin/env python3
"""
Polymarket BTC 15-Min Mean Reversion — Validation & Backtest Script.

Runs the full 8-step pipeline from the article on real Polymarket data:

  Step 1  Fetch 90 days of BTC 15-min trade history from Polymarket CLOB.
  Step 2  Convert YES prices to log returns (logit for extreme prices).
  Step 3  Add lag-1 column (previous bar's log return).
  Step 4  Encode direction as +1 / -1.
  Step 5  Bucket analysis — mean return by previous bar direction.
  Step 6  75/25 time-split out-of-sample validation.
  Step 7  Generate signal equity curve.
  Step 8  Win rate, total compound return, Sharpe ratio.

Usage
-----
  python polymarket_backtest.py                     # 90-day default
  python polymarket_backtest.py --days 60
  python polymarket_backtest.py --min-edge 0.003    # stricter gate

The script does NOT place any orders. Safe to run at any time.
"""

import argparse
import logging
import math
import time
from typing import List

import numpy as np

from src.bar_builder import Bar, BarBuilder, log_return
from src.mean_reversion import MeanReversionValidator
from src.polymarket_client import PolymarketCLOBClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("backtest")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_bars(days: int = 90) -> List[Bar]:
    client   = PolymarketCLOBClient(paper_trade=True)
    start_ts = int(time.time()) - days * 86_400

    log.info("Searching for BTC 15-min markets (last %d days)…", days)
    markets = client.find_btc_15min_markets(active_only=False, limit=500)
    log.info("Found %d candidate markets", len(markets))

    builder = BarBuilder(bar_seconds=900)
    bars: List[Bar] = []

    for i, market in enumerate(markets):
        if market.end_date_ts > 0 and market.end_date_ts < start_ts:
            continue   # older than our window

        trades = client.get_trades(
            token_id=market.yes_token_id,
            start_ts=start_ts,
            limit=5_000,
        )
        for trade in trades:
            bar = builder.feed_trade(trade.price, trade.timestamp)
            if bar:
                bars.append(bar)

        if (i + 1) % 25 == 0:
            log.info(
                "  %d / %d markets → %d bars so far",
                i + 1, len(markets), len(bars),
            )

    bars.sort(key=lambda b: b.bar_start_ts)
    log.info("Total bars assembled: %d", len(bars))
    return bars


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _equity_curve_ascii(trade_returns: np.ndarray, width: int = 60, height: int = 12):
    cum = np.cumsum(trade_returns)
    if len(cum) == 0:
        return

    lo, hi = float(np.min(cum)), float(np.max(cum))
    if hi == lo:
        print("  (flat equity curve)")
        return

    step    = max(1, len(cum) // width)
    sampled = cum[::step][:width]

    print()
    print(f"  Equity curve (cumulative log-return):")
    print(f"  Peak  : {hi:+.3f}  ({(math.exp(hi) - 1) * 100:.0f}%)")
    print(f"  Final : {cum[-1]:+.3f}  ({(math.exp(cum[-1]) - 1) * 100:.0f}%)")
    print()

    for row in range(height, -1, -1):
        threshold = lo + (hi - lo) * row / height
        line = "".join("*" if v >= threshold else " " for v in sampled)
        if row % 3 == 0:
            print(f"  {threshold:+.3f} | {line}")
        else:
            print(f"         | {line}")

    bar_count = len(cum)
    print(f"         +-{'-' * len(sampled)}")
    print(f"          0{' ' * (len(sampled) // 2 - 6)}bars → {bar_count:,}")


def _hr():
    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest(days: int = 90, min_edge: float = 0.002):
    bars = fetch_bars(days=days)

    if len(bars) < 50:
        log.error(
            "Only %d bars available — need at least 50. "
            "Check internet access and Polymarket API availability.",
            len(bars),
        )
        return

    _hr()
    print("STEP 2-8 · MEAN REVERSION VALIDATION PIPELINE")
    _hr()

    validator = MeanReversionValidator(min_edge=min_edge)
    result    = validator.validate(bars)

    _hr()
    print("RESULTS")
    _hr()
    print(f"  Total bars           : {result.n_bars:,}")
    print(f"  Win rate             : {result.win_rate:.1%}")
    total_pct = (math.exp(result.total_log_return) - 1) * 100
    print(f"  Total log return     : {result.total_log_return:+.4f}  ({total_pct:+.1f}%)")
    print(f"  Annualized Sharpe    : {result.annualized_sharpe:.2f}")
    print(f"  Edge is LIVE         : {'YES ✓' if result.edge_is_live else 'NO  ✗'}")
    print()

    for label, (down_b, up_b) in [
        ("IN-SAMPLE  (oldest 75%)", result.in_sample),
        ("OOS        (newest 25%)", result.out_of_sample),
    ]:
        print(f"  {label}")
        print(
            f"    prev=DOWN → next mean {down_b.mean_return:+.4f}  "
            f"n={down_b.count:,}  (signal=+1, buy YES)"
        )
        print(
            f"    prev=UP   → next mean {up_b.mean_return:+.4f}  "
            f"n={up_b.count:,}  (signal=-1, buy NO)"
        )
        print()

    # Reconstruct per-trade log returns for the equity curve
    closes   = np.array([b.close for b in bars])
    log_rets = np.array([log_return(closes[i - 1], closes[i]) for i in range(1, len(closes))])
    lag_rets = log_rets[:-1]
    curr_rets = log_rets[1:]
    signals       = -np.sign(lag_rets)
    trade_returns = signals * curr_rets

    _equity_curve_ascii(trade_returns)

    _hr()
    if result.edge_is_live:
        print("✓ EDGE VALIDATED — bot is cleared to trade this week.")
        print("  Run weekly: python polymarket_backtest.py")
    else:
        print("✗ EDGE NOT VALIDATED — do not trade until OOS buckets clear threshold.")
        print(f"  Required: both OOS bucket means ≥ {min_edge:.4f}, n ≥ 30")
    _hr()
    print()

    # Per-bar economics summary (illustrative, no real position sizing)
    mean_bar_return = float(np.mean(trade_returns)) if len(trade_returns) else 0.0
    spread_cost = 0.015
    net_per_bar = mean_bar_return - spread_cost
    print("ILLUSTRATIVE PER-BAR ECONOMICS  ($1,000 position size)")
    print(f"  Raw edge (mean log-return)  : {mean_bar_return:+.4f}  (${mean_bar_return * 1000:.2f})")
    print(f"  Spread + gas cost           : {-spread_cost:.4f}  (-${spread_cost * 1000:.2f})")
    print(f"  Net per bar                 : {net_per_bar:+.4f}  (${net_per_bar * 1000:.2f})")
    bars_per_year = 35_040
    print(f"  Bars/year                   : {bars_per_year:,}  (96/day × 365)")
    ann_net = net_per_bar * bars_per_year * 1_000
    print(f"  Gross annual (no compound)  : ${ann_net:,.0f}  per $1,000 capital")
    print()
    print("Note: illustrative only. Real PnL depends on edge persistence,")
    print("      execution quality, and regime stability.")
    _hr()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Polymarket BTC 15-Min Mean Reversion Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--days",     type=int,   default=90,    help="Days of history")
    parser.add_argument("--min-edge", type=float, default=0.002, help="Min OOS mean per bucket")
    args = parser.parse_args()

    run_backtest(days=args.days, min_edge=args.min_edge)
