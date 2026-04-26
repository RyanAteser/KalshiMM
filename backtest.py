#!/usr/bin/env python3
"""
KalshiMM Backtester — Monte Carlo simulation of the Avellaneda-Stoikov
strategy on synthetic KXBTC15M market sessions.

Usage examples
--------------
# Single run with default params
python backtest.py

# Single run with custom params
python backtest.py --gamma 0.15 --k 1.2 --sigma 0.05 --sims 2000

# Grid search over gamma / k / sigma
python backtest.py --grid

Simulation model
----------------
Each simulation represents one 15-minute BTC contract session.
- YES price follows a bounded random walk (reflecting BTC-price uncertainty).
- Order arrivals follow a Poisson process with rate k * exp(-k * spread / 2).
- The A-S engine recalculates bid/ask every tick (default: every 1 second).
- Fees are deducted on each fill (7 bps trading + 3.5 bps regulatory ~ 1%).
- Final inventory is marked to the terminal YES price (1 or 0 in reality;
  here we use the last simulated price as an approximation).
"""

import argparse
import json
import logging
import math
import random

import numpy as np

from src.avellaneda_stoikov import ASParams, AvellanedaStoikov

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("backtest")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class BacktestConfig:
    def __init__(
        self,
        gamma: float = 0.10,
        k: float = 1.50,
        sigma: float = 0.04,
        order_size: int = 10,
        max_position: int = 50,
        session_duration: float = 900.0,
        n_simulations: int = 1_000,
        initial_price: float = 0.50,
        price_drift: float = 0.0,
        tick_interval: float = 1.0,
        fee_rate: float = 0.010,    # ~1% round-trip (7bps trading + 3.5bps reg)
    ):
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.order_size = order_size
        self.max_position = max_position
        self.session_duration = session_duration
        self.n_simulations = n_simulations
        self.initial_price = initial_price
        self.price_drift = price_drift
        self.tick_interval = tick_interval
        self.fee_rate = fee_rate


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------

def simulate(cfg: BacktestConfig, seed: int = 0) -> dict:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    engine = AvellanedaStoikov(ASParams(
        gamma=cfg.gamma,
        k=cfg.k,
        sigma_min=0.005,
        sigma_max=0.50,
    ))

    price = cfg.initial_price
    inventory = 0      # net YES contracts
    cash = 0.0
    trades = 0
    spreads: list = []
    peak_cash = 0.0
    min_cash = 0.0

    steps = int(cfg.session_duration / cfg.tick_interval)
    for i in range(steps):
        tte = cfg.session_duration - i * cfg.tick_interval

        # Random walk with Gaussian noise
        noise = np_rng.normal(
            cfg.price_drift * cfg.tick_interval,
            cfg.sigma * math.sqrt(cfg.tick_interval),
        )
        price = max(0.01, min(0.99, price + noise))

        quote = engine.quote(
            mid=price,
            inventory=inventory,
            sigma=cfg.sigma,
            time_remaining=tte,
            session_duration=cfg.session_duration,
        )
        if quote is None:
            continue

        # Poisson arrival probability per tick
        arrival_p = cfg.k * math.exp(-cfg.k * quote.spread / 2.0) * cfg.tick_interval

        if abs(inventory) < cfg.max_position:
            # Bid fill: market sell hits our bid
            if price <= quote.bid and rng.random() < arrival_p:
                cost = cfg.order_size * quote.bid * (1.0 + cfg.fee_rate)
                cash -= cost
                inventory += cfg.order_size
                trades += 1
                spreads.append(quote.spread)

            # Ask fill: market buy hits our ask (BUY NO = equivalent)
            if price >= quote.ask and rng.random() < arrival_p:
                revenue = cfg.order_size * quote.ask * (1.0 - cfg.fee_rate)
                cash += revenue
                inventory -= cfg.order_size
                trades += 1
                spreads.append(quote.spread)

        peak_cash = max(peak_cash, cash)
        min_cash = min(min_cash, cash)

    # Mark final inventory at terminal price
    final_pnl = cash + inventory * price

    return {
        "pnl": final_pnl,
        "trades": trades,
        "avg_spread": float(np.mean(spreads)) if spreads else 0.0,
        "final_inventory": inventory,
        "max_drawdown_cash": peak_cash - min_cash,
    }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def run_backtest(cfg: BacktestConfig) -> dict:
    results = [simulate(cfg, seed=i) for i in range(cfg.n_simulations)]

    pnls = [r["pnl"] for r in results]
    trades = [r["trades"] for r in results]
    spreads = [r["avg_spread"] for r in results]

    win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls))
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

    return {
        "n_simulations": cfg.n_simulations,
        "win_rate": round(win_rate, 3),
        "sharpe": round(sharpe, 3),
        "pnl": {
            "mean":   round(mean_pnl, 5),
            "median": round(float(np.median(pnls)), 5),
            "std":    round(std_pnl, 5),
            "p5":     round(float(np.percentile(pnls,  5)), 5),
            "p95":    round(float(np.percentile(pnls, 95)), 5),
            "min":    round(min(pnls), 5),
            "max":    round(max(pnls), 5),
        },
        "avg_trades_per_session": round(float(np.mean(trades)), 1),
        "avg_spread_captured": round(float(np.mean(spreads)), 4),
        "params": {
            "gamma":      cfg.gamma,
            "k":          cfg.k,
            "sigma":      cfg.sigma,
            "order_size": cfg.order_size,
            "fee_rate":   cfg.fee_rate,
        },
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search():
    gammas = [0.05, 0.10, 0.20, 0.40]
    ks     = [1.00, 1.50, 2.00]
    sigmas = [0.02, 0.04, 0.08]

    best_score = -float("inf")
    best_params = None

    for gamma in gammas:
        for k in ks:
            for sigma in sigmas:
                cfg = BacktestConfig(gamma=gamma, k=k, sigma=sigma, n_simulations=500)
                res = run_backtest(cfg)
                score = res["sharpe"] * res["win_rate"]
                indicator = " ←" if score > best_score else ""
                if score > best_score:
                    best_score = score
                    best_params = (gamma, k, sigma, res)
                log.info(
                    "gamma=%.2f  k=%.1f  sigma=%.3f | win=%.1f%%  pnl_mean=%.5f  sharpe=%.3f%s",
                    gamma, k, sigma,
                    res["win_rate"] * 100,
                    res["pnl"]["mean"],
                    res["sharpe"],
                    indicator,
                )

    if best_params:
        gamma, k, sigma, res = best_params
        log.info("\n=== Best ===  gamma=%.2f  k=%.1f  sigma=%.3f  score=%.3f",
                 gamma, k, sigma, best_score)
        print(json.dumps(res, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KalshiMM Backtester")
    parser.add_argument("--grid",  action="store_true", help="Grid search over parameters")
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--k",     type=float, default=1.50)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--sims",  type=int,   default=1_000)
    parser.add_argument("--size",  type=int,   default=10)
    args = parser.parse_args()

    if args.grid:
        grid_search()
    else:
        cfg = BacktestConfig(
            gamma=args.gamma,
            k=args.k,
            sigma=args.sigma,
            n_simulations=args.sims,
            order_size=args.size,
        )
        result = run_backtest(cfg)
        print(json.dumps(result, indent=2))
