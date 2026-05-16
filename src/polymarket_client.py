"""
Polymarket CLOB + Gamma API client.

Two API surfaces:
  Gamma  (https://gamma-api.polymarket.com)  — human-readable market search.
  CLOB   (https://clob.polymarket.com)        — order book, trade history, orders.

Read-only calls (market discovery, trade history, book) require no auth.
Order placement requires a Polygon wallet private key via POLYMARKET_PRIVATE_KEY
and the py-clob-client package (optional; paper mode works without it).

BTC 15-min market notes
-----------------------
Polymarket lists recurring "Will BTC be up in the next 15 minutes?" markets.
Each market has a unique condition_id and YES/NO ERC-1155 token IDs.
The YES token price (0-1) is what we track as the probability series.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

log = logging.getLogger(__name__)

_CLOB_BASE  = "https://clob.polymarket.com"
_GAMMA_BASE = "https://gamma-api.polymarket.com"
_TIMEOUT    = 15
_PAGE_LIMIT = 500


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PolyMarket:
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    end_date_ts: float
    active: bool = True

    @property
    def time_to_expiry(self) -> float:
        return max(0.0, self.end_date_ts - time.time())


@dataclass
class PolyTrade:
    price: float
    size: float
    timestamp: float
    side: str   # "BUY" or "SELL"


@dataclass
class PolyBook:
    bids: List[Dict]   # [{"price": float, "size": float}, ...]
    asks: List[Dict]

    @property
    def best_bid(self) -> Optional[float]:
        return max((b["price"] for b in self.bids), default=None)

    @property
    def best_ask(self) -> Optional[float]:
        return min((a["price"] for a in self.asks), default=None)

    @property
    def mid(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return bb or ba

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class PolymarketCLOBClient:
    """
    Thin wrapper around the Polymarket Gamma + CLOB REST APIs.

    paper_trade=True  → log orders but never send them to the exchange.
    private_key       → Polygon EOA key for live order signing; read from
                        POLYMARKET_PRIVATE_KEY env var if not supplied.
    """

    def __init__(
        self,
        paper_trade: bool = True,
        private_key: Optional[str] = None,
    ):
        self._paper = paper_trade
        self._pk = private_key or os.getenv("POLYMARKET_PRIVATE_KEY")
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "KalshiMM-MeanReversion/1.0"
        self._clob_client = None   # lazy-init on first live order

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    def find_btc_15min_markets(
        self,
        active_only: bool = True,
        limit: int = 100,
    ) -> List[PolyMarket]:
        """
        Search Gamma API for BTC 15-minute Up/Down markets.
        Returns markets sorted by end_date (soonest first).
        Falls back to CLOB /markets if Gamma returns nothing.
        """
        markets = self._gamma_search(active_only=active_only, limit=limit)
        if not markets:
            log.info("Gamma search empty — trying CLOB /markets fallback")
            markets = self._clob_search(active_only=active_only, limit=limit)

        markets.sort(key=lambda m: m.end_date_ts)
        log.info("Found %d BTC 15-min markets", len(markets))
        return markets

    def _gamma_search(self, active_only: bool, limit: int) -> List[PolyMarket]:
        params: Dict = {"limit": limit}
        if active_only:
            params["active"] = "true"
            params["closed"] = "false"

        # Gamma supports keyword search via `q` param.
        for query in ("BTC 15 minutes", "bitcoin 15 minutes", "BTC up 15"):
            try:
                resp = self._session.get(
                    f"{_GAMMA_BASE}/markets",
                    params={**params, "q": query},
                    timeout=_TIMEOUT,
                )
                resp.raise_for_status()
                raw = resp.json()
            except Exception as exc:
                log.warning("Gamma search (%r) failed: %s", query, exc)
                continue

            items = raw if isinstance(raw, list) else raw.get("markets", [])
            markets = [m for item in items for m in [self._parse_gamma(item)] if m]
            if markets:
                return markets

        return []

    def _clob_search(self, active_only: bool, limit: int) -> List[PolyMarket]:
        """Scan CLOB /markets for BTC 15-min questions."""
        params: Dict = {"limit": min(limit, _PAGE_LIMIT)}
        if active_only:
            params["active"] = "true"
        try:
            resp = self._session.get(
                f"{_CLOB_BASE}/markets", params=params, timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json()
        except Exception as exc:
            log.error("CLOB /markets failed: %s", exc)
            return []

        items = raw if isinstance(raw, list) else raw.get("data", [])
        markets = []
        for item in items:
            q = str(item.get("question") or "").lower()
            if ("btc" in q or "bitcoin" in q) and "15" in q:
                m = self._parse_clob_market(item)
                if m:
                    markets.append(m)
        return markets

    def _parse_gamma(self, m: dict) -> Optional[PolyMarket]:
        q = m.get("question", "")
        q_low = q.lower()
        if not (("btc" in q_low or "bitcoin" in q_low) and "15" in q_low):
            return None

        yes_tok, no_tok = self._extract_tokens(m)
        if not yes_tok or not no_tok:
            return None

        end_ts = self._parse_ts(m.get("endDate") or m.get("end_date_iso") or "")
        return PolyMarket(
            condition_id=m.get("conditionId") or m.get("id") or "",
            question=q,
            yes_token_id=yes_tok,
            no_token_id=no_tok,
            end_date_ts=end_ts,
            active=bool(m.get("active", True)),
        )

    def _parse_clob_market(self, m: dict) -> Optional[PolyMarket]:
        yes_tok, no_tok = self._extract_tokens(m)
        if not yes_tok or not no_tok:
            return None
        end_ts = self._parse_ts(
            m.get("end_date_iso") or m.get("endDate") or m.get("game_start_time") or ""
        )
        return PolyMarket(
            condition_id=m.get("condition_id") or m.get("id") or "",
            question=m.get("question", ""),
            yes_token_id=yes_tok,
            no_token_id=no_tok,
            end_date_ts=end_ts,
            active=bool(m.get("active", True)),
        )

    @staticmethod
    def _extract_tokens(m: dict):
        """Return (yes_token_id, no_token_id) from a Gamma or CLOB market dict."""
        tokens = m.get("tokens") or m.get("clobTokenIds") or []
        yes_tok = no_tok = None
        for t in tokens:
            if isinstance(t, str):
                # clobTokenIds is sometimes a list of two strings [yes, no]
                if yes_tok is None:
                    yes_tok = t
                elif no_tok is None:
                    no_tok = t
            elif isinstance(t, dict):
                outcome = (t.get("outcome") or "").strip().lower()
                tok_id  = t.get("token_id") or t.get("tokenId") or ""
                if outcome in ("yes", "up", "1"):
                    yes_tok = tok_id
                elif outcome in ("no", "down", "0"):
                    no_tok = tok_id
        return yes_tok, no_tok

    # ------------------------------------------------------------------
    # Trade history
    # ------------------------------------------------------------------

    def get_trades(
        self,
        token_id: str,
        start_ts: int = 0,
        limit: int = 10_000,
    ) -> List[PolyTrade]:
        """
        Fetch historical trades for a YES token, paginating until start_ts.
        Returns trades sorted chronologically (oldest first).
        """
        collected: List[PolyTrade] = []
        cursor: Optional[str] = None

        while len(collected) < limit:
            params: Dict = {"market": token_id, "limit": min(_PAGE_LIMIT, limit)}
            if cursor:
                params["before"] = cursor

            try:
                resp = self._session.get(
                    f"{_CLOB_BASE}/trades", params=params, timeout=_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                log.error("get_trades failed (token=%s…): %s", token_id[:8], exc)
                break

            raw = data.get("data", []) if isinstance(data, dict) else (data or [])
            if not raw:
                break

            stop = False
            for t in raw:
                try:
                    ts    = float(t.get("timestamp") or t.get("time") or 0)
                    price = float(t.get("price") or 0)
                    size  = float(t.get("size") or t.get("amount") or 0)
                    side  = str(t.get("side") or "BUY").upper()
                    if price <= 0:
                        continue
                    if ts > 0 and ts < start_ts:
                        stop = True
                        break
                    collected.append(PolyTrade(price=price, size=size, timestamp=ts, side=side))
                except Exception:
                    pass

            if stop:
                break

            cursor = data.get("next_cursor") if isinstance(data, dict) else None
            if not cursor:
                break

            time.sleep(0.1)   # be polite to rate limiter

        collected.sort(key=lambda t: t.timestamp)
        return collected

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    def get_book(self, token_id: str) -> Optional[PolyBook]:
        try:
            resp = self._session.get(
                f"{_CLOB_BASE}/book",
                params={"token_id": token_id},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.error("get_book failed: %s", exc)
            return None

        def _parse(levels) -> List[Dict]:
            out = []
            for e in (levels or []):
                try:
                    out.append({"price": float(e["price"]), "size": float(e["size"])})
                except Exception:
                    pass
            return out

        return PolyBook(bids=_parse(data.get("bids")), asks=_parse(data.get("asks")))

    def get_mid_price(self, token_id: str) -> Optional[float]:
        book = self.get_book(token_id)
        return book.mid if book else None

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_marketable_limit(
        self,
        token_id: str,
        side: str,           # "BUY" or "SELL"
        size_usd: float,
        slippage: float = 0.01,   # 1-cent extra aggressiveness
    ) -> dict:
        """
        Place a marketable limit at best-quote + slippage in the trade direction.
        Paper mode logs but does not send.
        Returns {"success": bool, "order_id": str, "price": float}.
        """
        book = self.get_book(token_id)
        if book is None:
            return {"success": False, "error": "no book data"}

        if side == "BUY":
            raw_price = (book.best_ask or 0.99) + slippage
        else:
            raw_price = (book.best_bid or 0.01) - slippage

        price = round(max(0.01, min(0.99, raw_price)), 2)

        if self._paper:
            log.info(
                "[PAPER] %s token=…%s  size=$%.2f  @%.2f",
                side, token_id[-6:], size_usd, price,
            )
            return {"success": True, "order_id": f"PAPER-{int(time.time())}", "price": price}

        return self._live_order(token_id, side, size_usd, price)

    def _live_order(self, token_id: str, side: str, size_usd: float, price: float) -> dict:
        if self._clob_client is None:
            self._init_clob_client()
        if self._clob_client is None:
            return {"success": False, "error": "CLOB client not initialised"}

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType  # type: ignore
            from py_clob_client.constants import BUY, SELL              # type: ignore

            args = OrderArgs(
                token_id=token_id,
                price=price,
                size=round(size_usd / price, 2),
                side=BUY if side == "BUY" else SELL,
                order_type=OrderType.GTC,
            )
            resp = self._clob_client.create_and_post_order(args)
            oid  = resp.get("orderID") or resp.get("order_id") or ""
            return {"success": True, "order_id": oid, "price": price}

        except Exception as exc:
            log.error("Live order failed: %s", exc)
            return {"success": False, "error": str(exc)}

    def _init_clob_client(self):
        if not self._pk:
            log.error("POLYMARKET_PRIVATE_KEY not set — live orders disabled")
            return
        try:
            from py_clob_client.client import ClobClient  # type: ignore
            chain_id = 137   # Polygon mainnet
            self._clob_client = ClobClient(_CLOB_BASE, key=self._pk, chain_id=chain_id)
            self._clob_client.set_api_creds(self._clob_client.create_or_derive_api_creds())
            log.info("CLOB client ready (Polygon chain_id=%d)", chain_id)
        except Exception as exc:
            log.error("Failed to init CLOB client: %s", exc)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ts(s: str) -> float:
        if not s:
            return 0.0
        from datetime import datetime, timezone
        for fmt in (
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S+00:00",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                pass
        return 0.0
