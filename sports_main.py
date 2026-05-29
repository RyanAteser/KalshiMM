#!/usr/bin/env python3
"""
Kalshi Sports Arbitrage Bot entry point.

Paper trade:  python sports_main.py --paper
Live:         python sports_main.py
Backtest:     python sports_main.py --backtest
"""

import argparse
import logging
import signal
import sys

import yaml
from dotenv import load_dotenv

load_dotenv()


def _setup_logging(level: str):
    fmt = "%(asctime)s  %(levelname)-8s  %(name)-25s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _run_backtest(config: dict):
    """
    Replay the 10 built-in presets against all 20 synthetic game scenarios
    and print a results table — no Kalshi connection needed.
    """
    from src.sports_arb import ArbParams, SportsArbBot
    import math, random

    PRESETS = [
        {"label": "Featherweight",    "C":0.95,"entry":0.54,"patience":5, "edge":0.025,"clip":30, "budget":250,"max_imb":150},
        {"label": "Conservative",     "C":0.96,"entry":0.52,"patience":5, "edge":0.025,"clip":30, "budget":250,"max_imb":150},
        {"label": "Patient Tight",    "C":0.97,"entry":0.52,"patience":8, "edge":0.025,"clip":30, "budget":250,"max_imb":150},
        {"label": "High-Gate Tight",  "C":0.99,"entry":0.52,"patience":12,"edge":0.025,"clip":40, "budget":250,"max_imb":150},
        {"label": "Balanced",         "C":0.97,"entry":0.50,"patience":12,"edge":0.025,"clip":30, "budget":400,"max_imb":300},
        {"label": "Balanced Wide",    "C":0.96,"entry":0.52,"patience":12,"edge":0.025,"clip":40, "budget":400,"max_imb":300},
        {"label": "Eager Entry",      "C":0.96,"entry":0.50,"patience":3, "edge":0.025,"clip":20, "budget":250,"max_imb":300},
        {"label": "Deep Average-Down","C":0.99,"entry":0.50,"patience":12,"edge":0.015,"clip":20, "budget":250,"max_imb":300},
        {"label": "Throughput",       "C":0.98,"entry":0.52,"patience":12,"edge":0.025,"clip":70, "budget":700,"max_imb":1200},
        {"label": "Throughput Max",   "C":0.97,"entry":0.52,"patience":8, "edge":0.025,"clip":70, "budget":700,"max_imb":1200},
    ]

    GAMES = [
        {"n":"Knicks @ Pacers",   "seed":101,"drift":0,    "mr":0.075,"ep":0.07,"es":0.42,"hs":0.020},
        {"n":"Arsenal v Chelsea", "seed":102,"drift":0,    "mr":0.085,"ep":0.06,"es":0.40,"hs":0.018},
        {"n":"Dodgers @ Padres",  "seed":103,"drift":0,    "mr":0.070,"ep":0.075,"es":0.45,"hs":0.020},
        {"n":"Djokovic v Alcaraz","seed":104,"drift":0,    "mr":0.080,"ep":0.08,"es":0.38,"hs":0.019},
        {"n":"Celtics @ Heat",    "seed":105,"drift":0,    "mr":0.072,"ep":0.07,"es":0.44,"hs":0.021},
        {"n":"Inter v Juventus",  "seed":106,"drift":0,    "mr":0.088,"ep":0.058,"es":0.40,"hs":0.018},
        {"n":"Yankees @ Red Sox", "seed":107,"drift":0,    "mr":0.078,"ep":0.072,"es":0.43,"hs":0.020},
        {"n":"Chiefs @ Bills",    "seed":201,"drift":0.012,"mr":0.035,"ep":0.085,"es":0.60,"hs":0.023},
        {"n":"Man City v Spurs",  "seed":202,"drift":0.016,"mr":0.030,"ep":0.08,"es":0.62,"hs":0.024},
        {"n":"Nuggets @ Warriors","seed":203,"drift":-0.010,"mr":0.038,"ep":0.09,"es":0.58,"hs":0.022},
        {"n":"Bayern v Dortmund", "seed":204,"drift":0.018,"mr":0.028,"ep":0.075,"es":0.66,"hs":0.025},
        {"n":"Astros @ Mariners", "seed":205,"drift":-0.008,"mr":0.040,"ep":0.09,"es":0.55,"hs":0.022},
        {"n":"Sinner v Medvedev", "seed":206,"drift":0.014,"mr":0.032,"ep":0.10,"es":0.52,"hs":0.021},
        {"n":"Eagles @ Cowboys",  "seed":207,"drift":0.020,"mr":0.025,"ep":0.08,"es":0.70,"hs":0.026},
        {"n":"Lakers @ Grizzlies","seed":301,"drift":-0.045,"mr":0.008,"ep":0.07,"es":0.70,"hs":0.024,"shiftAt":120,"shiftDrift":0.06},
        {"n":"PSG v Marseille",   "seed":302,"drift":0.0,  "mr":0.005,"ep":0.12,"es":0.95,"hs":0.028},
        {"n":"Packers @ Bears",   "seed":303,"drift":0.055,"mr":0.0,  "ep":0.06,"es":0.55,"hs":0.022},
        {"n":"Raptors @ Hornets", "seed":304,"drift":0.0,  "mr":0.012,"ep":0.11,"es":0.90,"hs":0.027},
        {"n":"Liverpool v Everton","seed":305,"drift":0.03,"mr":0.006,"ep":0.08,"es":0.80,"hs":0.026,"shiftAt":150,"shiftDrift":-0.09},
        {"n":"Mets @ Braves",     "seed":306,"drift":-0.02,"mr":0.004,"ep":0.10,"es":0.85,"hs":0.026,"shiftAt":100,"shiftDrift":0.05},
    ]

    def mulberry32(seed):
        a = seed
        def rng():
            nonlocal a
            a = (a + 0x6D2B79F5) & 0xFFFFFFFF
            t = ((a ^ (a >> 15)) * (1 | a)) & 0xFFFFFFFF
            t = (t + ((t ^ (t >> 7)) * (61 | t))) & 0xFFFFFFFF
            return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
        return rng

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def build_path(g, T=200):
        r = mulberry32(g["seed"])
        x = 0.0
        asks_a, asks_b = [], []
        for t in range(T):
            drift = g["drift"]
            if g.get("shiftAt") and t >= g["shiftAt"]:
                drift = g["shiftDrift"]
            if r() < g["ep"]:
                sign = 1 if r() < 0.5 else -1
                x += sign * g["es"] * (0.6 + r())
            x += drift + (r() - 0.5) * 0.35
            x -= g["mr"] * x
            tf = 1 + 3.0 * (t / T)
            p  = max(0.02, min(0.98, sigmoid(x * tf * 0.55)))
            hs = g["hs"]
            aA = max(0.01, min(0.99, p + hs + (r() - 0.5) * 0.01))
            bidA = max(0.01, min(0.99, p - hs + (r() - 0.5) * 0.01))
            aB = max(0.01, min(0.99, 1 - bidA + (r() - 0.5) * 0.01))
            asks_a.append(aA)
            asks_b.append(aB)
        return asks_a, asks_b, x >= 0

    print(f"\n{'Preset':<22} {'Total P&L':>10} {'Wins':>6} {'Avg/game':>10}")
    print("─" * 55)

    for pr in PRESETS:
        params = ArbParams(
            C=pr["C"], entry=pr["entry"], patience=pr["patience"],
            edge=pr["edge"], clip=pr["clip"], budget=pr["budget"],
            max_imb=pr["max_imb"],
        )
        total_pnl, wins = 0.0, 0
        for g in GAMES:
            asks_a, asks_b, a_won = build_path(g)
            bot = SportsArbBot(params)
            for aA, aB in zip(asks_a, asks_b):
                bot.tick(aA, aB)
            result = bot.settle(a_won)
            total_pnl += result.net_pnl
            if result.net_pnl >= 0:
                wins += 1

        print(f"{pr['label']:<22} ${total_pnl:>9.0f} {wins:>5}/20 ${total_pnl/20:>9.2f}/game")

    print()


def main():
    parser = argparse.ArgumentParser(description="Kalshi Sports Arbitrage Bot")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--paper",     action="store_true")
    parser.add_argument("--backtest",  action="store_true", help="Replay synthetic games, no API needed")
    parser.add_argument("--log-level", default=None, choices=["DEBUG","INFO","WARNING","ERROR"])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if args.paper:
        config["paper_trade"] = True

    level = args.log_level or config.get("log_level", "INFO")
    _setup_logging(level)

    if args.backtest:
        _run_backtest(config)
        return

    from src.sports_bot import SportsArbTradingBot
    bot = SportsArbTradingBot(config)

    def _shutdown(sig, frame):
        logging.getLogger("main").info("Shutdown received")
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    bot.start()


if __name__ == "__main__":
    main()
