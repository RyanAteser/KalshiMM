"""
MarketMaker — the main event loop that ties every component together.

Each tick:
  1. The BondingBot (re)selects which KXBTC15M markets to quote.
  2. For each bonded market the A-S engine calculates optimal bid / ask.
  3. Stale resting quotes are cancelled and fresh ones are posted.
  4. Risk checks gate every action; a hard halt stops the bot entirely.

Architecture notes
------------------
- Single-threaded synchronous loop.  The BTC price feed runs in its own
  daemon thread (BTCPriceFeed) and communicates via thread-safe shared state.
- Order lifecycle is cancel-then-replace on every tick.  This is intentionally
  simple: Kalshi does not have a native amend, and the API latency is low
  enough that cancel + new order keeps quotes fresh.
- The inventory manager is the source of truth for positions in-process.
  It is re-synced from the Kalshi API at startup and every 5 minutes.
"""

import logging
import time
from typing import Optional

from .avellaneda_stoikov import ASParams, AvellanedaStoikov
from .bonding_bot import BondingBot
from .btc_feed import BTCPriceFeed
from .inventory_manager import InventoryManager
from .kalshi_client import KalshiMMClient
from .models import Action, Side
from .order_manager import OrderManager
from .risk_manager import RiskManager

log = logging.getLogger(__name__)

_BALANCE_REFRESH_EVERY = 10    # ticks
_POSITION_SYNC_EVERY = 60      # ticks  (~5 min at 5s interval)
_STATUS_LOG_EVERY = 12         # ticks  (~1 min)


class MarketMaker:
    def __init__(self, config: dict):
        self._cfg = config
        paper = config.get("paper_trade", False)

        # -- API client --
        self._client = KalshiMMClient(paper_trade=paper)

        # -- BTC price feed --
        as_cfg = config.get("avellaneda_stoikov", {})
        self._btc = BTCPriceFeed(window=as_cfg.get("sigma_window", 100))

        # -- A-S pricing engine --
        self._as = AvellanedaStoikov(ASParams(
            gamma=as_cfg.get("gamma", 0.10),
            k=as_cfg.get("k", 1.50),
            min_spread=as_cfg.get("min_spread", 0.02),
            max_spread=as_cfg.get("max_spread", 0.20),
            sigma_min=as_cfg.get("sigma_min", 0.005),
            sigma_max=as_cfg.get("sigma_max", 0.40),
        ))

        # -- Trading params --
        tr = config.get("trading", {})
        self._order_size: int = tr.get("order_size", 10)
        self._max_position: int = tr.get("max_position", 50)
        self._loop_interval: float = tr.get("loop_interval", 5.0)
        self._min_tte: float = tr.get("min_time_to_expiry", 300.0)
        self._session_duration: float = tr.get("session_duration", 900.0)

        # -- Sub-systems --
        risk_cfg = config.get("risk", {})
        self._inventory = InventoryManager(max_position=self._max_position)
        self._risk = RiskManager(
            max_total_position=risk_cfg.get("max_total_position", 200),
            max_drawdown=risk_cfg.get("max_drawdown", 0.15),
            max_loss_per_market=risk_cfg.get("max_loss_per_market", 50.0),
            daily_loss_limit=risk_cfg.get("daily_loss_limit", 200.0),
        )
        self._orders = OrderManager(self._client)

        bond_cfg = config.get("bonding", {})
        self._bonding = BondingBot(
            client=self._client,
            series=tr.get("series", "KXBTC15M"),
            target_markets=bond_cfg.get("target_markets", 3),
            min_spread=bond_cfg.get("min_spread_to_bond", 0.04),
            min_volume=bond_cfg.get("min_volume", 100),
            min_time_to_expiry=self._min_tte,
            rebalance_interval=bond_cfg.get("rebalance_interval", 60.0),
        )

        self._running = False
        self._stopped = False
        self._cycle = 0
        self._last_balance: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        log.info(
            "KalshiMM starting | paper=%s  series=%s",
            self._cfg.get("paper_trade"), self._cfg.get("trading", {}).get("series"),
        )
        self._btc.start()
        time.sleep(2)  # warm-up the price feed

        balance = self._client.get_balance()
        self._last_balance = balance
        self._risk.initialise(balance)

        positions = self._client.get_positions()
        self._inventory.sync_from_api(positions)

        self._running = True
        self._stopped = False
        try:
            self._loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        log.info("Shutting down — cancelling all open orders")
        self._running = False
        self._orders.cancel_all()
        self._btc.stop()
        self._print_summary()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self):
        while self._running:
            try:
                self._cycle += 1
                self._tick()
            except Exception as exc:
                log.error("Unhandled error in tick %d: %s", self._cycle, exc, exc_info=True)
            time.sleep(self._loop_interval)

    def _tick(self):
        # Refresh market selection
        if self._bonding.should_rebalance():
            self._bonding.rebalance()

        # Refresh balance periodically
        if self._cycle % _BALANCE_REFRESH_EVERY == 0:
            self._last_balance = self._client.get_balance()

        # Re-sync positions from API periodically
        if self._cycle % _POSITION_SYNC_EVERY == 0:
            positions = self._client.get_positions()
            self._inventory.sync_from_api(positions)

        # Global risk gate
        if not self._risk.check(self._last_balance, self._inventory.total_exposure()):
            if self._risk.is_halted():
                log.error("Risk halt — stopping bot")
                self.stop()
            return

        sigma = self._btc.volatility()

        if self._cycle % _STATUS_LOG_EVERY == 0:
            self._bonding.log_status()
            log.info(
                "sigma=%.4f  balance=$%.2f  exposure=%d  pnl=$%.4f",
                sigma, self._last_balance,
                self._inventory.total_exposure(),
                self._inventory.total_realized_pnl(),
            )

        for ticker in self._bonding.bonded_markets():
            self._make_market(ticker, sigma)

    # ------------------------------------------------------------------
    # Per-market quoting
    # ------------------------------------------------------------------

    def _make_market(self, ticker: str, sigma: float):
        if not self._risk.market_allowed(ticker):
            return

        # Always fetch a fresh snapshot so quotes track live bid/ask movement.
        # Fall back to bonding-bot cache only if the API call fails.
        market = self._client.get_market(ticker) or self._bonding.get_market_info(ticker)
        if market is None:
            return

        mid = market.mid_price
        if mid is None:
            log.debug("%s: no mid price, skipping", ticker)
            return

        tte = market.time_to_expiry

        # Near-expiry regime: digital gamma explodes as 1/tau.
        # Pull all quotes in final 10s. Flatten-only in last 30s.
        # Widen spread 2x in last 2 min to compensate for rising gamma.
        if tte <= 10:
            log.info("%s: tte=%.0fs — pulling all quotes", ticker, tte)
            self._orders.cancel_market_orders(ticker)
            return
        if tte < self._min_tte:
            log.info("%s: tte=%.0fs — below min_tte, pulling quotes", ticker, tte)
            self._orders.cancel_market_orders(ticker)
            return

        # Spread multiplier increases as expiry approaches
        if tte < 120:
            spread_mult = 3.0
        elif tte < 300:
            spread_mult = 2.0
        else:
            spread_mult = 1.0

        # BTC feed staleness check — pull quotes if feed is >30s stale
        if self._btc.is_stale(max_age=30.0):
            log.warning("%s: BTC feed stale — pulling quotes", ticker)
            self._orders.cancel_market_orders(ticker)
            return

        net_inv = self._inventory.net_inventory(ticker)
        quote = self._as.quote(
            mid=mid,
            inventory=net_inv,
            sigma=sigma * spread_mult,   # widen near expiry via effective sigma
            time_remaining=tte,
            session_duration=self._session_duration,
        )
        if quote is None:
            return

        quote = self._as.skew_for_inventory(quote, net_inv, self._max_position)

        log.info(
            "%s | mid=%.2f  bid=%.2f  ask=%.2f  spread=%.2f  inv=%+d  tte=%.0fs  mult=%.0fx",
            ticker, mid, quote.bid, quote.ask, quote.spread, net_inv, tte, spread_mult,
        )

        # Cancel stale quotes before placing fresh ones
        self._orders.cancel_market_orders(ticker)

        # Bid: BUY YES at bid price
        if self._inventory.can_buy_yes(ticker):
            self._orders.place_quote(
                ticker=ticker,
                side=Side.YES,
                action=Action.BUY,
                price=quote.bid,
                count=self._order_size,
            )

        # Ask: BUY NO at (1 - ask_price)
        # Selling YES = buying NO on Kalshi; quoting NO at (1 - ask) fills when
        # someone buys YES at ask price.
        if self._inventory.can_buy_no(ticker):
            no_price = round(1.0 - quote.ask, 2)
            self._orders.place_quote(
                ticker=ticker,
                side=Side.NO,
                action=Action.BUY,
                price=no_price,
                count=self._order_size,
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self):
        log.info("==================== Session Summary ====================")
        log.info("Cycles run      : %d", self._cycle)
        log.info("Realized PnL    : $%.4f", self._inventory.total_realized_pnl())
        positions = self._inventory.position_summary()
        if positions:
            for ticker, info in positions.items():
                log.info("  %s  yes=%d  no=%d  net=%+d  pnl=$%.4f",
                         ticker, info["yes"], info["no"], info["net"], info["realized_pnl"])
        else:
            log.info("  (no open positions)")
        log.info("=========================================================")
