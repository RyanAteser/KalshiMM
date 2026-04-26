#!/usr/bin/env python3
import argparse
import logging
import signal
import sys

import yaml
from dotenv import load_dotenv

from src.market_maker import MarketMaker

load_dotenv()


def setup_logging(level: str = "INFO"):
    fmt = "%(asctime)s  %(levelname)-8s  %(name)-25s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="KalshiMM — Avellaneda-Stoikov Market Maker + Bonding Bot for KXBTC15M"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--paper", action="store_true", help="Paper-trade mode (no real orders)")
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if args.paper:
        config["paper_trade"] = True

    log_level = args.log_level or config.get("log_level", "INFO")
    setup_logging(log_level)

    log = logging.getLogger("main")
    mode = "PAPER" if config.get("paper_trade") else "LIVE"
    log.info("KalshiMM starting in %s mode", mode)

    mm = MarketMaker(config)

    def _handle_signal(sig, frame):
        log.info("Shutdown signal (%s) received", sig)
        mm.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    mm.start()


if __name__ == "__main__":
    main()
