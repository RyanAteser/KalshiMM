"""
Kalshi Weather Bot — main trading loop.

Strategy:
  1. Fetch NOAA temperature forecasts for configured cities.
  2. Fetch active Kalshi weather markets for those cities.
  3. For each market, compare NOAA's implied probability to Kalshi's ask price.
  4. Buy YES contracts where edge = P(NOAA) - P(market) exceeds the threshold.
  5. Repeat every `loop_interval` seconds.

Run paper trade first:
  python weather_main.py --paper

Run live:
  python weather_main.py
"""

import logging
import time
from typing import Dict, List, Optional

from .edge_calculator import EdgeCalculator, TradeSignal
from .kalshi_client import KalshiMMClient
from .models import Action, MarketInfo, Side
from .noaa_feed import NOAAFeed
from .risk_manager import RiskManager

log = logging.getLogger(__name__)


class WeatherBot:
    def __init__(self, config: dict):
        wc = config.get("weather", {})
        rc = config.get("risk", {})

        self._paper         = config.get("paper_trade", True)
        self._loop_interval = wc.get("loop_interval", 120)
        self._cities        = wc.get("cities", ["new_york"])
        self._series        = wc.get("series", [])
        self._max_contracts = wc.get("max_contracts_per_signal", 1)
        self._min_edge      = wc.get("min_edge", 0.15)
        self._min_tte       = wc.get("min_time_to_expiry", 3600)  # 1 hour
        self._max_ask       = wc.get("max_ask", 0.80)             # never buy overpriced
        self._running       = False

        self._client = KalshiMMClient(paper_trade=self._paper)
        self._noaa   = NOAAFeed(cities=self._cities, cache_ttl=1800)
        self._edge   = EdgeCalculator(min_edge=self._min_edge)
        self._risk   = RiskManager(
            max_total_position=rc.get("max_total_position", 20),
            max_drawdown=rc.get("max_drawdown", 0.15),
            max_loss_per_market=rc.get("max_loss_per_market", 2.0),
            daily_loss_limit=rc.get("daily_loss_limit", 5.0),
        )

        # Track open positions: ticker → contracts bought
        self._positions: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        balance = self._client.get_balance()
        self._risk.initialise(balance)
        mode = "PAPER" if self._paper else "LIVE"
        log.info("WeatherBot starting [%s] | balance=$%.2f | cities=%s",
                 mode, balance, self._cities)

        self._running = True
        while self._running:
            try:
                self._tick()
            except Exception as exc:
                log.error("Tick error: %s", exc, exc_info=True)
            time.sleep(self._loop_interval)

    def stop(self):
        self._running = False
        log.info("WeatherBot stopped.")

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def _tick(self):
        balance = self._client.get_balance()
        total_exposure = sum(self._positions.values())

        if not self._risk.check(balance, total_exposure):
            log.warning("Risk check failed — skipping tick")
            return

        # 1. Fetch NOAA forecasts
        forecasts = self._noaa.fetch_all()
        live_cities = [c for c, f in forecasts.items() if f is not None]
        if not live_cities:
            log.warning("No NOAA forecasts available — skipping tick")
            return

        # 2. Fetch Kalshi markets for each configured series
        markets: List[MarketInfo] = []
        for series in self._series:
            batch = self._client.get_markets(series)
            markets.extend(batch)

        markets = self._filter_markets(markets)
        log.info("Eligible markets: %d", len(markets))

        # 3. Find mispricings
        signals = self._edge.find_signals(markets, forecasts)

        # 4. Execute trades
        for signal in signals:
            self._maybe_trade(signal, balance)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_markets(self, markets: List[MarketInfo]) -> List[MarketInfo]:
        now = time.time()
        out = []
        for m in markets:
            tte = m.close_ts - now
            if tte < self._min_tte:
                log.debug("SKIP %s | tte=%.0fs < min=%ds", m.ticker, tte, self._min_tte)
                continue
            if m.yes_ask is not None and m.yes_ask > self._max_ask:
                log.debug("SKIP %s | ask=%.3f > max=%.3f", m.ticker, m.yes_ask, self._max_ask)
                continue
            out.append(m)
        return out

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _maybe_trade(self, signal: TradeSignal, balance: float):
        ticker = signal.ticker

        # Don't add to an existing position
        if self._positions.get(ticker, 0) > 0:
            log.debug("Already holding %s — skip", ticker)
            return

        if not self._risk.market_allowed(ticker):
            return

        contracts = self._size(signal, balance)
        if contracts < 1:
            log.info("Signal %s skipped — sizing returned 0 contracts", ticker)
            return

        cost = signal.buy_price * contracts
        log.info(
            "BUY %d × %s @ $%.3f | edge=%+.3f | cost=$%.2f [%s]",
            contracts, ticker, signal.buy_price, signal.edge, cost,
            "PAPER" if self._paper else "LIVE",
        )

        result = self._client.place_order(
            ticker=ticker,
            side=Side.YES,
            action=Action.BUY,
            price=signal.buy_price,
            count=contracts,
        )

        if result.success:
            self._positions[ticker] = self._positions.get(ticker, 0) + contracts
            self._risk.record_market_pnl(ticker, -cost)  # record outlay as negative PnL
            log.info("Order placed: %s", result.order_id)
        else:
            log.warning("Order failed for %s: %s", ticker, result.error)

    def _size(self, signal: TradeSignal, balance: float) -> int:
        """
        Position sizing: bet at most 2% of balance per signal,
        capped at max_contracts_per_signal.
        """
        if balance <= 0 or signal.buy_price <= 0:
            return 0
        max_spend     = balance * 0.02
        by_kelly      = int(max_spend / signal.buy_price)
        return max(0, min(by_kelly, self._max_contracts))
