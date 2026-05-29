"""
Kalshi sports arbitrage bot — live trading loop.

For each configured sports series, it:
  1. Fetches active markets (one per game).
  2. Runs SportsArbBot tick-by-tick as prices update.
  3. Places BUY YES or BUY NO orders when the bot signals.
  4. Settles positions when a market resolves.

Kalshi binary sports markets:
  YES ask  = price to buy YES (Team A wins)
  NO ask   = 1.0 - YES bid  (price to buy NO = Team B wins)
"""

import logging
import time
from typing import Dict, List, Optional

from .kalshi_client import KalshiMMClient
from .models import Action, MarketInfo, OrderResult, Side
from .risk_manager import RiskManager
from .sports_arb import ArbParams, GameResult, SportsArbBot, TickSnapshot

log = logging.getLogger(__name__)


class SportsArbTradingBot:
    def __init__(self, config: dict):
        sc = config.get("sports", {})
        rc = config.get("risk", {})

        self._paper  = config.get("paper_trade", True)
        self._series: List[str] = sc.get("series", [])
        self._loop_s: float     = sc.get("loop_interval", 30.0)
        self._running           = False

        # Bot parameters from config
        self._arb_params = ArbParams(
            C        = sc.get("C", 0.97),
            entry    = sc.get("entry", 0.50),
            patience = sc.get("patience", 12),
            edge     = sc.get("edge", 0.025),
            clip     = sc.get("clip", 30),
            budget   = sc.get("budget", 400.0),
            max_imb  = sc.get("max_imb", 300),
        )

        self._client = KalshiMMClient(paper_trade=self._paper)
        self._risk   = RiskManager(
            max_total_position  = rc.get("max_total_position", 200),
            max_drawdown        = rc.get("max_drawdown", 0.15),
            max_loss_per_market = rc.get("max_loss_per_market", 50.0),
            daily_loss_limit    = rc.get("daily_loss_limit", 200.0),
        )

        # Per-market bot instances: ticker → SportsArbBot
        self._bots: Dict[str, SportsArbBot] = {}

    # ------------------------------------------------------------------

    def start(self):
        balance = self._client.get_balance()
        self._risk.initialise(balance)
        mode = "PAPER" if self._paper else "LIVE"
        log.info("SportsArbBot starting [%s] | balance=$%.2f | series=%s",
                 mode, balance, self._series)

        self._running = True
        while self._running:
            try:
                self._tick()
            except Exception as exc:
                log.error("Tick error: %s", exc, exc_info=True)
            time.sleep(self._loop_s)

    def stop(self):
        self._running = False
        log.info("SportsArbBot stopped.")

    # ------------------------------------------------------------------

    def _tick(self):
        balance = self._client.get_balance()
        total_exposure = sum(
            int(bot._qty_a + bot._qty_b) for bot in self._bots.values()
        )

        if not self._risk.check(balance, total_exposure):
            log.warning("Risk check failed — skipping tick")
            return

        for series in self._series:
            markets = self._client.get_markets(series)
            for market in markets:
                self._process_market(market)

    def _process_market(self, market: MarketInfo):
        ticker = market.ticker

        # Compute YES ask and NO ask
        ask_yes = market.yes_ask
        if ask_yes is None:
            return

        # NO ask = 1.0 - YES bid.  If no bid, fall back to (1 - yes_ask + spread_est)
        yes_bid = market.yes_bid
        if yes_bid is not None:
            ask_no = round(1.0 - yes_bid, 4)
        else:
            ask_no = round(1.0 - ask_yes + 0.02, 4)  # rough fallback

        ask_no  = max(0.01, min(0.99, ask_no))
        ask_yes = max(0.01, min(0.99, ask_yes))

        log.debug("%s  yes_ask=%.3f  no_ask=%.3f  combined=%.3f",
                  ticker, ask_yes, ask_no, ask_yes + ask_no)

        # Create or fetch bot for this market
        if ticker not in self._bots:
            self._bots[ticker] = SportsArbBot(self._arb_params)
            log.info("New market: %s", ticker)

        bot = self._bots[ticker]
        snap: TickSnapshot = bot.tick(ask_a=ask_yes, ask_b=ask_no)

        log.info(
            "%s | pairs=%.0f sumAvg=%.3f locked=$%.2f | %s",
            ticker, snap.pairs, snap.sum_avg, snap.locked, snap.status,
        )

        # Execute orders based on the most recent buy record
        if bot.buys and bot.buys[-1].tick == snap.tick:
            buy = bot.buys[-1]
            side   = Side.YES if buy.side == 'A' else Side.NO
            price  = buy.price
            count  = max(1, int(buy.size))

            if not self._risk.market_allowed(ticker):
                return

            log.info(
                "ORDER %s %s %d sh @ $%.3f [%s]",
                ticker, side.value, count, price,
                "PAPER" if self._paper else "LIVE",
            )

            result: OrderResult = self._client.place_order(
                ticker=ticker,
                side=side,
                action=Action.BUY,
                price=price,
                count=count,
            )

            if result.success:
                log.info("Order OK: %s", result.order_id)
                self._risk.record_market_pnl(ticker, -(price * count))
            else:
                log.warning("Order failed for %s: %s", ticker, result.error)
