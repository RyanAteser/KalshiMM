import logging
from typing import Dict, List, Optional

from .kalshi_client import KalshiMMClient
from .models import Action, Order, OrderResult, OrderStatus, Side

log = logging.getLogger(__name__)


class OrderManager:
    """
    Places and cancels resting quotes on Kalshi.

    Each call to place_quote / cancel_market_orders is synchronous (the
    pykalshi SDK is blocking).  The market maker calls cancel then re-quote
    on every tick, so we keep a per-ticker list of open order IDs for
    efficient bulk cancellation.
    """

    def __init__(self, client: KalshiMMClient):
        self._client = client
        self._open: Dict[str, Order] = {}               # order_id → Order
        self._by_ticker: Dict[str, List[str]] = {}      # ticker → [order_ids]

    # ------------------------------------------------------------------
    # Quoting
    # ------------------------------------------------------------------

    def place_quote(
        self,
        ticker: str,
        side: Side,
        action: Action,
        price: float,
        count: int,
    ) -> Optional[Order]:
        """Place a single resting limit order and register it locally."""
        result: OrderResult = self._client.place_order(
            ticker=ticker,
            side=side,
            action=action,
            price=price,
            count=count,
        )
        if not result.success:
            log.warning("place_quote failed [%s %s %s]: %s", ticker, side, action, result.error)
            return None

        order = Order(
            order_id=result.order_id,
            ticker=ticker,
            side=side,
            action=action,
            price=price,
            count=count,
            status=OrderStatus.OPEN,
            filled_count=result.filled_count or 0,
        )
        self._open[order.order_id] = order
        self._by_ticker.setdefault(ticker, []).append(order.order_id)

        log.info(
            "quote  %s %s %s x%d @%.4f → %s",
            ticker, action.value, side.value, count, price, order.order_id,
        )
        return order

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel_market_orders(self, ticker: str):
        """Cancel all open orders for a specific ticker."""
        ids = list(self._by_ticker.get(ticker, []))
        for oid in ids:
            order = self._open.get(oid)
            if order and order.status == OrderStatus.OPEN:
                if self._client.cancel_order(oid):
                    order.status = OrderStatus.CANCELLED
                    log.debug("cancelled %s", oid)

        # Prune cancelled IDs from the ticker list
        self._by_ticker[ticker] = [
            oid for oid in self._by_ticker.get(ticker, [])
            if self._open.get(oid, Order("", "", None, None, 0, 0)).status == OrderStatus.OPEN
        ]

    def cancel_all(self):
        for ticker in list(self._by_ticker.keys()):
            self.cancel_market_orders(ticker)
        log.info("All open orders cancelled")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def open_count(self, ticker: str) -> int:
        return sum(
            1
            for oid in self._by_ticker.get(ticker, [])
            if self._open.get(oid, Order("", "", None, None, 0, 0)).status == OrderStatus.OPEN
        )
