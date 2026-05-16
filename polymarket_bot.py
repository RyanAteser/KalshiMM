#!/usr/bin/env python3
"""
Polymarket BTC 15-Minute Mean Reversion Bot — CLI entry point.

Strategy summary
----------------
At each 15-minute UTC boundary (00, 15, 30, 45):
  1. Observe the log return of the bar that just closed.
  2. Generate a mean reversion signal: -sign(lag-1).
     - Previous bar went up  → bet YES price goes down (buy NO).
     - Previous bar went down → bet YES price goes up  (buy YES).
  3. Gate the trade: OOS expected value must clear the spread cost.
  4. Enter the next BTC 15-min market with a marketable limit order.
  5. Hold to resolution (Polymarket markets settle automatically).

Edge is re-validated weekly on 90 days of history using a 75/25
time-split (in-sample / out-of-sample). The bot pauses if the OOS
bucket means fall below the configured threshold.

Usage
-----
  python polymarket_bot.py --paper          # paper trade (default safe mode)
  python polymarket_bot.py                  # live (needs POLYMARKET_PRIVATE_KEY)
  python polymarket_bot.py --config foo.yaml --log-level DEBUG

Environment
-----------
  POLYMARKET_PRIVATE_KEY  Polygon EOA private key for live order signing.
"""

import argparse
import logging
import signal
import sys

import yaml
from dotenv import load_dotenv

from src.mean_reversion_bot import MeanReversionBot

load_dotenv()


def _setup_logging(level: str = "INFO"):
    fmt = "%(asctime)s  %(levelname)-8s  %(name)-25s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket BTC 15-Min Mean Reversion Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",    default="config.yaml", help="YAML config path")
    parser.add_argument("--paper",     action="store_true",   help="Paper-trade mode (no real orders)")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if args.paper:
        config["paper_trade"] = True

    log_level = args.log_level or config.get("log_level", "INFO")
    _setup_logging(log_level)

    log = logging.getLogger("polymarket_bot")
    mode = "PAPER" if config.get("paper_trade") else "LIVE"
    log.info("Polymarket Mean Reversion Bot | mode=%s", mode)

    bot = MeanReversionBot(config)

    def _shutdown(sig, _frame):
        log.info("Signal %s received — shutting down", sig)
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    bot.start()


if __name__ == "__main__":
    main()
