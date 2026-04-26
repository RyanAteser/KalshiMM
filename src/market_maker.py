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
from .toxicity import is_toxic

log = logging.getLogger(__name__)

_BALANCE_REFRESH_EVERY = 10
_POSITION_SYNC_EVERY = 60
_STATUS_LOG_EVERY = 12


class MarketMaker:
    def __init__(self, config: dict):
        self._cfg = config
        paper = config.get("paper_trade", False)

        self._client = KalshiMMClient(paper_trade=paper)

        as_cfg = config.get("avellaneda_stoikov", {})
        self._btc = BTCPriceFeed(window=as_cfg.get("sigma_window", 100))

        self._as = AvellanedaStoikov(ASParams(
            gamma=as_cfg.get("gamma", 0.10),
            k=as_cfg.get("k", 1.50),
            min_spread=as_cfg.get("min_spread", 0.02),
            max_spread=as_cfg.get("max_spread", 0.20),
            sigma_min=as_cfg.get("sigma_min", 0.005),
            sigma_max=as_cfg.get("sigma_max", 0.40),
        ))

        tr = config.get("trading", {})
        self._order_size: int = tr.get("order_size", 10)
        self._max_position: int = tr.get("max_position", 50)
        self._loop_interval: float = tr.get("loop_interval", 5.0)
        self._min_tte: float = tr.get("min_time_to_expiry", 300.0)
        self._session_duration: float = tr.get("session_duration", 900.0)

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
            target_markets=1,  # 🔥 reduced from 3+
            min_spread=bond_cfg.get("min_spread_to_bond", 0.04),
            min_volume=bond_cfg.get("min_volume", 100),
            min_time_to_expiry=self._min_tte,
            rebalance_interval=bond_cfg.get("rebalance_interval", 60.0),
        )

        self._running = False
        self._stopped = False
        self._cycle = 0
        self._last_balance: float = 0.0

    # ---------------------- LIFECYCLE ----------------------

    def start(self):
        log.info("Starting Kalshi MM")

        self._btc.start()
        time.sleep(2)

        balance = self._client.get_balance()
        self._last_balance = balance
        self._risk.initialise(balance)

        positions = self._client.get_positions()
        self._inventory.sync_from_api(positions)

        self._running = True
        self._stopped = False

        try:
            self._loop()
        finally:
            self.stop()

    def stop(self):
        if self._stopped:
            return
        self._stopped = True

        log.info("Stopping bot — cancelling all orders")
        self._running = False
        self._orders.cancel_all()
        self._btc.stop()
        self._print_summary()

    # ---------------------- LOOP ----------------------

    def _loop(self):
        while self._running:
            try:
                self._cycle += 1
                self._tick()
            except Exception as e:
                log.error("Tick error: %s", e, exc_info=True)
            time.sleep(self._loop_interval)

    def _tick(self):
        if self._bonding.should_rebalance():
            self._bonding.rebalance()

        if self._cycle % _BALANCE_REFRESH_EVERY == 0:
            self._last_balance = self._client.get_balance()

        if self._cycle % _POSITION_SYNC_EVERY == 0:
            positions = self._client.get_positions()
            self._inventory.sync_from_api(positions)

        # 🔥 HARD KILL SWITCH
        if self._inventory.total_realized_pnl() < -50:
            log.error("KILL SWITCH TRIGGERED (-$50)")
            self.stop()
            return

        if not self._risk.check(self._last_balance, self._inventory.total_exposure()):
            if self._risk.is_halted():
                self.stop()
            return

        sigma = self._btc.volatility()

        if self._cycle % _STATUS_LOG_EVERY == 0:
            log.info(
                "sigma=%.4f balance=%.2f pnl=%.2f exposure=%d",
                sigma,
                self._last_balance,
                self._inventory.total_realized_pnl(),
                self._inventory.total_exposure(),
            )

        for ticker in self._bonding.bonded_markets():
            self._make_market(ticker, sigma)

    # ---------------------- CORE ----------------------

    def _make_market(self, ticker: str, sigma: float):
        # 🚫 HARD FILTERS
        if ticker.startswith("KXMVE"):
            return

        if not self._risk.market_allowed(ticker):
            return

        market = self._client.get_market(ticker) or self._bonding.get_market_info(ticker)
        if market is None:
            return

        # --- Orderbook ---
        best_bid = getattr(market, "yes_bid", None)
        best_ask = getattr(market, "yes_ask", None)

        if best_bid is None or best_ask is None:
            return

        # --- Mid anchored to real market ---
        mid = (best_bid + best_ask) / 2

        # --- Toxic filter ---
        if hasattr(market, "orderbook") and is_toxic(market.orderbook):
            return

        # --- Time to expiry ---
        tte = market.time_to_expiry
        if tte < self._min_tte:
            self._orders.cancel_market_orders(ticker)
            return

        net_inv = self._inventory.net_inventory(ticker)

        # ---------------- ALPHA ----------------
        alpha = 0.0
        try:
            prices = self._btc.prices()
            if prices and len(prices) > 5:
                alpha = (prices[-1] - prices[-5]) * 0.1
        except Exception:
            alpha = 0.0

        # ---------------- QUOTE ----------------
        quote = self._as.quote(
            mid=mid,
            inventory=net_inv,
            sigma=sigma,
            time_remaining=tte,
            session_duration=self._session_duration,
            alpha=alpha,
        )

        if quote is None:
            return

        # ---------------- SPREAD CONTROL ----------------
        # Stay INSIDE spread (true MM behavior)
        bid_price = min(quote.bid, best_ask - 0.01)
        ask_price = max(quote.ask, best_bid + 0.01)

        # Clamp
        bid_price = max(0.01, min(0.99, bid_price))
        ask_price = max(0.01, min(0.99, ask_price))

        # 🚫 Prevent crossing (VERY IMPORTANT)
        if bid_price >= best_ask:
            return

        # ---------------- EV FILTER ----------------
        edge = quote.reservation_price - mid
        spread = ask_price - bid_price
        ev = edge - spread

        if ev < 0.002:  # slightly relaxed
            return

        log.debug(
            "%s | bid=%.3f ask=%.3f mid=%.3f ev=%.4f inv=%d",
            ticker, bid_price, ask_price, mid, ev, net_inv
        )

        # ---------------- CANCEL OLD ----------------
        self._orders.cancel_market_orders(ticker)

        # ---------------- INVENTORY LOGIC ----------------

        # NORMAL MM (balanced)
        if abs(net_inv) < 5:
            # Buy YES (bid)
            if self._inventory.can_buy_yes(ticker):
                self._orders.place_quote(
                    ticker=ticker,
                    side=Side.YES,
                    action=Action.BUY,
                    price=bid_price,
                    count=self._order_size,
                )

            # Sell YES (via NO)
            if self._inventory.can_buy_no(ticker):
                no_price = round(1.0 - ask_price, 4)

                if no_price > 0.01:
                    self._orders.place_quote(
                        ticker=ticker,
                        side=Side.NO,
                        action=Action.BUY,
                        price=no_price,
                        count=self._order_size,
                    )

        # ---------------- FORCE EXIT (THIS FIXES YOUR ISSUE) ----------------

        elif net_inv > 5:
            # Too long → aggressively sell
            log.info("%s: reducing long inventory (%d)", ticker, net_inv)

            no_price = round(1.0 - best_ask, 4)

            self._orders.place_quote(
                ticker=ticker,
                side=Side.NO,
                action=Action.BUY,
                price=no_price,
                count=self._order_size,
            )

        elif net_inv < -5:
            # Too short → aggressively buy
            log.info("%s: reducing short inventory (%d)", ticker, net_inv)

            self._orders.place_quote(
                ticker=ticker,
                side=Side.YES,
                action=Action.BUY,
                price=best_bid,
                count=self._order_size,
            )
    # ---------------------- SUMMARY ----------------------

    def _print_summary(self):
        log.info("====== SESSION SUMMARY ======")
        log.info("Cycles: %d", self._cycle)
        log.info("PnL: $%.4f", self._inventory.total_realized_pnl())

        for ticker, info in self._inventory.position_summary().items():
            log.info(
                "%s yes=%d no=%d net=%d pnl=%.2f",
                ticker,
                info["yes"],
                info["no"],
                info["net"],
                info["realized_pnl"],
            )

        log.info("=============================")