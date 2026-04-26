import logging
import time
from datetime import datetime, timezone
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


def _to_probability(val) -> Optional[float]:
    """
    Normalise a Kalshi price to [0, 1].
    The API returns prices as integers (1–99 cents); pykalshi may expose
    them as floats already divided by 100.  Handle both.
    """
    if val is None:
        return None
    try:
        f = float(val)
        return f / 100.0 if f > 1.0 else f
    except (ValueError, TypeError):
        return None


def _parse_ts(val) -> float:
    """
    Parse a timestamp that may be a unix float, an integer, or an ISO-8601
    string (e.g. "2025-04-25T18:30:00Z") and return unix epoch seconds.
    """
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            cleaned = val.replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned).timestamp()
        except Exception:
            pass
    return 0.0


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
            try:
                from pykalshi import MarketStatus  # type: ignore
                status_arg = MarketStatus.OPEN
            except ImportError:
                status_arg = "open"

            resp = self._client.get_markets(
                status=status_arg,
                series_ticker=series,
                limit=limit,
            )
            raw = _extract(resp, "markets") or []

            # Log one raw market so we can verify field names in production
            if raw:
                sample = raw[0]
                keys = list(sample.keys()) if isinstance(sample, dict) else [
                    a for a in dir(sample) if not a.startswith("_")
                ]
                log.info("market fields (sample): %s", keys)

            results = []
            for m in raw:
                ticker = _extract(m, "ticker")
                if not ticker:
                    continue
                results.append(self._parse_market(m, ticker))

            log.info("Fetched %d open %s markets", len(results), series)
            return results
        except Exception as exc:
            log.error("get_markets failed: %s", exc, exc_info=True)
            return []

    def get_market(self, ticker: str) -> Optional[MarketInfo]:
        try:
            resp = self._client.get_market(ticker)
            m = _extract(resp, "market") or resp
            return self._parse_market(m, ticker)
        except Exception as exc:
            log.error("get_market(%s) failed: %s", ticker, exc)
            return None

    def _parse_market(self, m, ticker: str) -> MarketInfo:
        """Convert a raw pykalshi market object/dict to MarketInfo."""
        close_raw = _extract(m, "close_time", "close_ts", "expiration_time",
                             "expiry_time", "close_date")
        yes_bid_raw = _extract(m, "yes_bid", "yes_bid_dollars")
        yes_ask_raw = _extract(m, "yes_ask", "yes_ask_dollars")
        last_raw    = _extract(m, "last_price", "last_price_dollars")
        strike_raw  = _extract(m, "strike_price", "floor_strike", "cap_strike", default=0)
        volume_raw  = _extract(m, "volume", "dollar_volume", default=0)

        return MarketInfo(
            ticker=ticker,
            strike_price=float(strike_raw or 0),
            close_ts=_parse_ts(close_raw),
            yes_bid=_to_probability(yes_bid_raw),
            yes_ask=_to_probability(yes_ask_raw),
            last_price=_to_probability(last_raw),
            volume=int(float(volume_raw or 0)),
        )

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
