"""
Bonding Bot — selects and manages which KXBTC15M markets to quote in.

Strategy
--------
Every `rebalance_interval` seconds the bot:
  1. Fetches all open KXBTC15M markets from Kalshi.
  2. Filters out markets that are too close to expiry, too illiquid,
     or have a spread too tight to earn edge on.
  3. Scores the remaining candidates by a simple edge proxy:
       score = spread × log(volume + 1) × uncertainty
     where uncertainty = 1 - |mid - 0.5| * 2  (markets near 50 are most
     uncertain and therefore most profitable to make in).
  4. Selects the top `target_markets` by score and "bonds" to them,
     committing to continuously quote both sides.
  5. Markets that expire or drop below thresholds are dropped; new ones
     that score higher are added — the bot automatically rotates.
"""

import logging
import math
import time
from typing import List, Optional

from .kalshi_client import KalshiMMClient
from .models import MarketInfo

log = logging.getLogger(__name__)


class BondingBot:
    def __init__(
        self,
        client: KalshiMMClient,
        series: str = "KXBTC15M",
        target_markets: int = 3,
        min_spread: float = 0.04,
        min_volume: int = 100,
        min_time_to_expiry: float = 300.0,
        rebalance_interval: float = 60.0,
    ):
        self._client = client
        self._series = series
        self._target = target_markets
        self._min_spread = min_spread
        self._min_volume = min_volume
        self._min_tte = min_time_to_expiry
        self._interval = rebalance_interval

        self._bonded: List[str] = []
        self._market_cache: List[MarketInfo] = []
        self._last_rebalance: float = 0.0

    # ------------------------------------------------------------------
    # Public API used by MarketMaker
    # ------------------------------------------------------------------

    def bonded_markets(self) -> List[str]:
        return list(self._bonded)

    def should_rebalance(self) -> bool:
        return time.time() - self._last_rebalance >= self._interval

    def rebalance(self) -> List[str]:
        """
        Refresh market selection.
        Returns the list of newly added tickers (for the caller to act on).
        """
        self._market_cache = self._client.get_markets(self._series)
        log.debug("Fetched %d open %s markets", len(self._market_cache), self._series)

        candidates = self._filter(self._market_cache)
        ranked = self._rank(candidates)
        new_selection = [m.ticker for m in ranked[: self._target]]

        added = [t for t in new_selection if t not in self._bonded]
        removed = [t for t in self._bonded if t not in new_selection]

        if added or removed:
            log.info(
                "Bonding rebalance | +%s  -%s  (total bonded: %d)",
                added, removed, len(new_selection),
            )

        self._bonded = new_selection
        self._last_rebalance = time.time()
        return added

    def get_market_info(self, ticker: str) -> Optional[MarketInfo]:
        """Return cached market data, falling back to a live API call."""
        for m in self._market_cache:
            if m.ticker == ticker:
                return m
        return self._client.get_market(ticker)

    def active_count(self) -> int:
        return len(self._bonded)

    def log_status(self):
        log.info("=== Bonding Status | %d / %d markets ===", len(self._bonded), self._target)
        for m in self._market_cache:
            if m.ticker in self._bonded:
                log.info(
                    "  %-40s mid=%.3f  spread=%.3f  vol=%5d  tte=%4.0fs",
                    m.ticker,
                    m.mid_price or 0.0,
                    m.spread or 0.0,
                    m.volume,
                    m.time_to_expiry,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter(self, markets: List[MarketInfo]) -> List[MarketInfo]:
        out = []
        for m in markets:
            tte = m.time_to_expiry
            if tte < self._min_tte:
                log.debug("SKIP %s | tte=%.0fs < min=%.0fs", m.ticker, tte, self._min_tte)
                continue
            if m.volume < self._min_volume:
                log.debug("SKIP %s | volume=%d < min=%d", m.ticker, m.volume, self._min_volume)
                continue
            if m.spread is None or m.spread < self._min_spread:
                log.debug("SKIP %s | spread=%s < min=%.3f", m.ticker, m.spread, self._min_spread)
                continue
            if m.mid_price is None:
                log.debug("SKIP %s | no mid price", m.ticker)
                continue
            out.append(m)

        log.info(
            "Bonding filter: %d / %d pass  (min_tte=%.0f  min_vol=%d  min_spread=%.3f)",
            len(out), len(markets), self._min_tte, self._min_volume, self._min_spread,
        )
        # If nothing passes, log a summary of all markets to help tune thresholds
        if not out and markets:
            for m in markets[:5]:
                log.info(
                    "  candidate: %s | tte=%.0fs  vol=%d  spread=%s  mid=%s",
                    m.ticker, m.time_to_expiry, m.volume,
                    f"{m.spread:.3f}" if m.spread is not None else "None",
                    f"{m.mid_price:.3f}" if m.mid_price is not None else "None",
                )
        return out

    def _rank(self, markets: List[MarketInfo]) -> List[MarketInfo]:
        def score(m: MarketInfo) -> float:
            spread = m.spread or 0.0
            vol_factor = math.log(m.volume + 1)
            mid = m.mid_price or 0.5
            # Markets near 50 cents are hardest to call and thus best to MM
            uncertainty = 1.0 - abs(mid - 0.5) * 2.0
            return spread * vol_factor * uncertainty

        return sorted(markets, key=score, reverse=True)
