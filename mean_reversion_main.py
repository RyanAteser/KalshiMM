#!/usr/bin/env python3
"""
Kalshi KXBTC15M Mean Reversion Bot — entry point.

Strategy summary
----------------
At each 15-minute UTC boundary (:00, :15, :30, :45):
  1. Sample live BTC price → compute 15-min log return.
  2. Signal = -sign(lag-1): fade the previous bar's direction.
     prev bar UP   → signal -1 → buy NO  (bet BTC retraces)
     prev bar DOWN → signal +1 → buy YES (bet BTC rebounds)
  3. Gate: OOS expected value must exceed spread + gas cost.
  4. Enter the near-the-money KXBTC15M contract (YES mid ≈ 0.50).
  5. Contract settles automatically at resolution — no manual exit.

Edge validation
---------------
On startup and weekly: pulls 90 days of BTC/USDT 15-min OHLC from
Binance (free, no auth) and runs a 75/25 OOS bucket analysis.
The bot pauses until both direction buckets show positive EV.

Run first:  python mean_reversion_backtest.py
Then:       python mean_reversion_main.py --paper
Live:       python mean_reversion_main.py   (needs Kalshi creds in .env)
"""

import argparse
import logging
import signal
import sys

import yaml
from dotenv import load_dotenv

from src.kalshi_mean_reversion_bot import KalshiMeanReversionBot

load_dotenv()


def _setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Kalshi KXBTC15M Mean Reversion Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",    default="config.yaml",
                        help="YAML config path (default: config.yaml)")
    parser.add_argument("--paper",     action="store_true",
                        help="Paper-trade mode — no real orders sent")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if args.paper:
        config["paper_trade"] = True

    _setup_logging(args.log_level or config.get("log_level", "INFO"))

    log = logging.getLogger("mean_reversion_main")
    mode = "PAPER" if config.get("paper_trade") else "LIVE"
    log.info("Kalshi Mean Reversion Bot | mode=%s | series=%s",
             mode, config.get("trading", {}).get("series", "KXBTC15M"))

    bot = KalshiMeanReversionBot(config)

    def _shutdown(sig, _frame):
        log.info("Signal %s — shutting down", sig)
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    bot.start()


if __name__ == "__main__":
    main()
