#!/usr/bin/env python3
"""
Kalshi Weather Bot entry point.

Paper trade:   python weather_main.py --paper
Live:          python weather_main.py
Custom config: python weather_main.py --config my_config.yaml
"""

import argparse
import logging
import signal
import sys

import yaml
from dotenv import load_dotenv

from src.weather_bot import WeatherBot

load_dotenv()


def _setup_logging(level: str):
    fmt = "%(asctime)s  %(levelname)-8s  %(name)-25s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="Kalshi Weather Arbitrage Bot")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--paper", action="store_true", help="Paper-trade mode")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if args.paper:
        config["paper_trade"] = True

    level = args.log_level or config.get("log_level", "INFO")
    _setup_logging(level)

    bot = WeatherBot(config)

    def _shutdown(sig, frame):
        logging.getLogger("main").info("Shutdown signal received")
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    bot.start()


if __name__ == "__main__":
    main()
