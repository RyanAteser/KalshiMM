"""
Kalshi API client wrapper — market data and order management.

Market discovery uses the two-step pattern from Kalshi98:
  1. get_markets()  → list of tickers (no price data in bulk response)
  2. get_market(ticker) → individual snapshot with bid/ask/last
Settled (≥0.99) and expired markets are filtered automatically.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import List, Optional

from .models import Action, MarketInfo, OrderResult, Side

log = logging.getLogger(__name__)

_SETTLED_THRESHOLD = 0.99
_MAX_CANDIDATES    = 50
_SNAP_RETRIES      = 3


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def _get(obj, *attrs):
    """Return first non-None value from dict keys or object attributes."""
    for a in attrs:
        v = obj.get(a) if isinstance(obj, dict) else getattr(obj, a, None)
        if v is not None:
            return v
    return None


def _safe_float(val) -> Optional[float]:
    """Return float only if strictly in (0, 1); None otherwise."""
    if val is None:
        return None
    try:
        v = float(val)
        return v if 0.0 < v < 1.0 else None
    except (TypeError, ValueError):
        return None


def _parse_ts(val) -> int:
    if val is None:
        return 0
    try:
        if isinstance(val, (int, float)):
            return int(val)
        s = str(val)
        for fmt in (
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S+00:00",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp())
            except ValueError:
                pass
    except Exception:
        pass
    return 0


def _get_close_ts(market) -> int:
    for attr in (
        "close_ts", "close_time", "close_time_ts", "close_timestamp",
        "closeTime", "close_time_seconds", "close",
    ):
        ts = _parse_ts(_get(market, attr))
        if ts > 0:
            return ts
    return 0


def _extract_strike(market) -> float:
    # 1. Direct numeric fields
    for attr in ("strike_price", "floor_price", "cap_price", "settlement_value"):
        val = _get(market, attr)
        if val is not None:
            try:
                v = float(val)
                if v > 1000:
                    return v
            except (TypeError, ValueError):
                pass

    # 2. Text subtitle — e.g. "$95,000"
    for attr in ("subtitle", "yes_sub_title", "no_sub_title", "title"):
        text = _get(market, attr)
        if text:
            m = re.search(r'\$?([\d,]+(?:\.\d+)?)', str(text))
            if m:
                try:
                    v = float(m.group(1).replace(",", ""))
                    if v > 1000:
                        return v
                except (TypeError, ValueError):
                    pass

    # 3. Ticker: KXBTC15M-25APR1400-T95000 → 95000
    ticker = _get(market, "ticker") or ""
    m = re.search(r'[T-](\d{4,6}(?:\.\d+)?)(?:[^0-9]|$)', str(ticker))
    if m:
        try:
            return float(m.group(1))
        except (TypeError, ValueError):
            pass

    return 0.0


def _snapshot_from_raw(market, ticker: str) -> Optional[dict]:
    """Build a normalised snapshot dict from a raw pykalshi market response."""
    yes_bid = _safe_float(_get(market,
        "yes_bid_dollars", "yes_bid", "bid", "best_bid_dollars"))
    yes_ask = _safe_float(_get(market,
        "yes_ask_dollars", "yes_ask", "ask", "best_ask_dollars"))
    last_px = _safe_float(_get(market,
        "last_price_dollars", "last_price", "last", "price_dollars"))
    volume  = _get(market, "volume_fp", "volume", "total_volume")
    close_ts = _get_close_ts(market)

    return {
        "ticker":     ticker,
        "yes_bid":    yes_bid,
        "yes_ask":    yes_ask,
        "last_price": last_px,
        "volume":     float(volume) if volume is not None else 0.0,
        "close_ts":   close_ts,
        "strike":     _extract_strike(market),
        "no_ask":     round(1.0 - yes_bid, 4) if yes_bid is not None else None,
        "no_bid":     round(1.0 - yes_ask, 4) if yes_ask is not None else None,
    }


def _snap_to_market_info(snap: dict) -> MarketInfo:
    yes_bid = snap.get("yes_bid")
    yes_ask = snap.get("yes_ask")
    mid = None
    if yes_bid is not None and yes_ask is not None:
        mid = (yes_bid + yes_ask) / 2.0
    elif yes_bid is not None:
        mid = yes_bid
    elif yes_ask is not None:
        mid = yes_ask
    elif snap.get("last_price") is not None:
        mid = snap["last_price"]

    return MarketInfo(
        ticker=snap["ticker"],
        strike_price=snap.get("strike") or 0.0,
        close_ts=float(snap.get("close_ts") or 0),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        last_price=snap.get("last_price"),
        volume=int(snap.get("volume") or 0),
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class KalshiMMClient:
    def __init__(self, paper_trade: bool = False):
        from pykalshi import KalshiClient  # type: ignore
        self._client = KalshiClient.from_env()
        self._paper = paper_trade

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    _logged_raw = False  # class-level flag — dump raw fields once per session

    def _get_snapshot(self, ticker: str) -> Optional[dict]:
        """Individual market snapshot with retry on rate-limit."""
        delay = 1.0
        for attempt in range(_SNAP_RETRIES):
            try:
                resp = self._client.get_market(ticker)
                market = getattr(resp, "market", resp)

                # One-time raw field dump so we can verify exact field names
                if not KalshiMMClient._logged_raw:
                    KalshiMMClient._logged_raw = True
                    if isinstance(market, dict):
                        log.info("RAW market keys: %s", sorted(market.keys()))
                        log.info("RAW market values: %s", {
                            k: v for k, v in market.items()
                            if k in ("ticker", "yes_bid", "yes_ask", "last_price",
                                     "close_time", "close_ts", "volume", "status")
                        })
                    else:
                        attrs = [a for a in dir(market) if not a.startswith("_")]
                        log.info("RAW market attrs: %s", attrs)
                        log.info("RAW market sample: %s", {
                            a: getattr(market, a, None)
                            for a in ("ticker", "yes_bid", "yes_ask", "last_price",
                                      "close_time", "close_ts", "volume", "status")
                        })

                snap = _snapshot_from_raw(market, ticker)
                return snap
            except Exception as exc:
                if attempt < _SNAP_RETRIES - 1 and "429" in str(exc):
                    time.sleep(delay)
                    delay *= 2
                elif attempt == _SNAP_RETRIES - 1:
                    log.warning("snapshot failed [%s]: %s", ticker, exc)
                    return None
        return None

    def get_markets(self, series: str, limit: int = 200) -> List[MarketInfo]:
        """
        Fetch active markets for a series.
        Uses two-step pattern: list tickers → individual snapshots.
        Settled (≥0.99) and already-expired contracts are excluded.
        """
        try:
            try:
                from pykalshi.models import MarketStatus  # type: ignore
                status_arg = MarketStatus.OPEN
            except ImportError:
                status_arg = "open"

            resp = self._client.get_markets(
                status=status_arg,
                series_ticker=series,
                limit=limit,
            )
        except Exception as exc:
            log.error("get_markets API call failed: %s", exc, exc_info=True)
            return []

        # Normalise response shape
        if hasattr(resp, "markets"):
            raw = resp.markets or []
        elif isinstance(resp, dict):
            raw = resp.get("markets") or []
        else:
            raw = list(resp) if resp else []

        log.info("get_markets returned %d tickers for %s", len(raw), series)

        now = time.time()
        results: List[MarketInfo] = []

        for m in raw[:_MAX_CANDIDATES]:
            ticker = _get(m, "ticker")
            if not ticker:
                continue

            snap = self._get_snapshot(ticker)
            if snap is None:
                continue

            bid       = snap.get("yes_bid")
            ask       = snap.get("yes_ask")
            last      = snap.get("last_price")
            close_ts  = snap.get("close_ts", 0)
            log.info(
                "snap  %-45s bid=%-6s ask=%-6s last=%-6s vol=%-6s tte=%.0fs",
                ticker,
                f"{bid:.4f}" if bid is not None else "None",
                f"{ask:.4f}" if ask is not None else "None",
                f"{last:.4f}" if last is not None else "None",
                snap.get("volume"),
                max(0.0, close_ts - now),
            )

            # Filter: no price data at all
            if bid is None and ask is None and last is None:
                log.info("SKIP %s | no price data", ticker)
                continue

            # Filter: settled contract
            if (bid is not None and bid >= _SETTLED_THRESHOLD) or \
               (ask is not None and ask >= _SETTLED_THRESHOLD):
                log.info("SKIP %s | settled (bid=%s ask=%s)", ticker, bid, ask)
                continue

            # Filter: already expired
            if close_ts > 0 and close_ts <= now:
                log.info("SKIP %s | expired (close_ts=%d now=%d)", ticker, close_ts, int(now))
                continue

            results.append(_snap_to_market_info(snap))

        results.sort(key=lambda m: m.close_ts)
        log.info("Active %s markets after filtering: %d", series, len(results))
        return results

    def get_market(self, ticker: str) -> Optional[MarketInfo]:
        snap = self._get_snapshot(ticker)
        if snap is None:
            return None
        return _snap_to_market_info(snap)

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

        # Kalshi always takes yes_price_dollars regardless of which side.
        yes_price = round((1.0 - price) if side == Side.NO else price, 4)

        from pykalshi._sync.portfolio import Action as KA, Side as KS  # type: ignore
        kalshi_action = KA.BUY if action == Action.BUY else KA.SELL
        kalshi_side   = KS.YES if side == Side.YES else KS.NO

        for attempt in range(3):
            try:
                resp = self._client.portfolio.place_order(
                    ticker=ticker,
                    action=kalshi_action,
                    side=kalshi_side,
                    count_fp=str(count),
                    yes_price_dollars=f"{yes_price:.4f}",
                )
                oid    = _get(resp, "order_id", "id")
                filled = int(_get(resp, "count_filled") or 0)
                fp     = _safe_float(_get(resp, "yes_price_dollars", "price"))
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
            return float(_get(resp, "balance_dollars", "balance") or 0)
        except Exception as exc:
            log.error("get_balance failed: %s", exc)
            return 0.0

    def get_positions(self) -> dict:
        try:
            resp = self._client.portfolio.get_positions()
            raw = _get(resp, "positions", "market_positions") or []
            out = {}
            for p in raw:
                ticker = _get(p, "ticker")
                if ticker:
                    out[ticker] = {
                        "yes_count": int(_get(p, "position", "yes_position") or 0),
                        "no_count":  int(_get(p, "no_position") or 0),
                    }
            return out
        except Exception as exc:
            log.error("get_positions failed: %s", exc)
            return {}
