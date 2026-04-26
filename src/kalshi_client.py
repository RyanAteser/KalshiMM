import logging
import time
from typing import List, Optional

from .models import Action, MarketInfo, OrderResult, Side

log = logging.getLogger(__name__)


def _extract(obj, *fields, default=None):
    """Pull first matching field from a dict or object."""
    for f in fields:
        val = obj.get(f) if isinstance(obj, dict) else getattr(obj, f, None)
        if val is not None:
            return val
    return default


def _opt_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


class KalshiMMClient:
    """
    Thin wrapper around pykalshi that normalises field names across API
    response shapes and handles retry / paper-trade mode.
    """

    def __init__(self, paper_trade: bool = False):
        from pykalshi import KalshiClient  # type: ignore
        self._client = KalshiClient.from_env()
        self._paper = paper_trade

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_markets(self, series: str, limit: int = 200) -> List[MarketInfo]:
        try:
            resp = self._client.get_markets(
                status="open",
                series_ticker=series,
                limit=limit,
            )
            raw = _extract(resp, "markets") or []
            results = []
            for m in raw:
                ticker = _extract(m, "ticker")
                if not ticker:
                    continue
                results.append(MarketInfo(
                    ticker=ticker,
                    strike_price=float(_extract(m, "strike_price", "floor_strike", default=0)),
                    close_ts=float(_extract(m, "close_ts", "close_time", default=0)),
                    yes_bid=_opt_float(_extract(m, "yes_bid_dollars", "yes_bid")),
                    yes_ask=_opt_float(_extract(m, "yes_ask_dollars", "yes_ask")),
                    last_price=_opt_float(_extract(m, "last_price_dollars", "last_price")),
                    volume=int(_extract(m, "volume", default=0)),
                ))
            return results
        except Exception as exc:
            log.error("get_markets failed: %s", exc)
            return []

    def get_market(self, ticker: str) -> Optional[MarketInfo]:
        try:
            resp = self._client.get_market(ticker)
            m = _extract(resp, "market") or resp
            return MarketInfo(
                ticker=ticker,
                strike_price=float(_extract(m, "strike_price", "floor_strike", default=0)),
                close_ts=float(_extract(m, "close_ts", "close_time", default=0)),
                yes_bid=_opt_float(_extract(m, "yes_bid_dollars", "yes_bid")),
                yes_ask=_opt_float(_extract(m, "yes_ask_dollars", "yes_ask")),
                last_price=_opt_float(_extract(m, "last_price_dollars", "last_price")),
                volume=int(_extract(m, "volume", default=0)),
            )
        except Exception as exc:
            log.error("get_market(%s) failed: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(
        self,
        ticker: str,
        side: Side,
        action: Action,
        price: float,
        count: int,
    ) -> OrderResult:
        if self._paper:
            return OrderResult(
                success=True,
                order_id=f"PAPER-{ticker}-{int(time.time() * 1000)}",
                filled_count=0,
                filled_price=price,
            )

        # Kalshi always takes yes_price_dollars regardless of side.
        # If we're placing a NO order at no_price, convert: yes_price = 1 - no_price.
        yes_price = round((1.0 - price) if side == Side.NO else price, 4)

        from pykalshi._sync.portfolio import Action as KA, Side as KS  # type: ignore
        kalshi_action = KA.BUY if action == Action.BUY else KA.SELL
        kalshi_side = KS.YES if side == Side.YES else KS.NO

        for attempt in range(3):
            try:
                resp = self._client.portfolio.place_order(
                    ticker=ticker,
                    action=kalshi_action,
                    side=kalshi_side,
                    count_fp=str(count),
                    yes_price_dollars=f"{yes_price:.4f}",
                )
                oid = _extract(resp, "order_id", "id")
                filled = int(_extract(resp, "count_filled", default=0))
                fp = _opt_float(_extract(resp, "yes_price_dollars", "price"))
                return OrderResult(
                    success=True,
                    order_id=str(oid),
                    filled_count=filled,
                    filled_price=fp,
                )
            except Exception as exc:
                msg = str(exc)
                if any(x in msg for x in ("insufficient_balance", "market_closed")):
                    return OrderResult(success=False, error=msg)
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    log.error("place_order failed after 3 attempts: %s", msg)
                    return OrderResult(success=False, error=msg)

        return OrderResult(success=False, error="exhausted retries")

    def cancel_order(self, order_id: str) -> bool:
        if self._paper:
            return True
        try:
            self._client.portfolio.cancel_order(order_id)
            return True
        except Exception as exc:
            log.warning("cancel_order(%s): %s", order_id, exc)
            return False

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        try:
            resp = self._client.portfolio.get_balance()
            return float(_extract(resp, "balance_dollars", "balance", default=0))
        except Exception as exc:
            log.error("get_balance failed: %s", exc)
            return 0.0

    def get_positions(self) -> dict:
        try:
            resp = self._client.portfolio.get_positions()
            raw = _extract(resp, "positions", "market_positions") or []
            out = {}
            for p in raw:
                ticker = _extract(p, "ticker")
                if ticker:
                    out[ticker] = {
                        "yes_count": int(_extract(p, "position", "yes_position", default=0)),
                        "no_count": int(_extract(p, "no_position", default=0)),
                    }
            return out
        except Exception as exc:
            log.error("get_positions failed: %s", exc)
            return {}
