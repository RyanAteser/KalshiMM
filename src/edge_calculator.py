"""
Edge calculator for Kalshi weather markets.

Compares NOAA forecast probability distribution against Kalshi market prices
to find mispricings. Fires a signal when:

  edge = P(NOAA) - P(market) > min_edge

where P(market) is the YES ask price (what you pay to buy the contract).

Kalshi weather market tickers follow patterns like:
  HIGHNY-25MAY30-T50   → NYC high temp, resolves if high >= 50°F
  HIGHNY-25MAY30-B4550 → NYC high temp in range 45-50°F

This module maps NOAA bucket labels to Kalshi tickers and computes edge.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .noaa_feed import TempForecast
from .models import MarketInfo

log = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    ticker: str
    noaa_prob: float        # NOAA's probability for this outcome
    market_prob: float      # Kalshi ask price (cost to buy YES)
    edge: float             # noaa_prob - market_prob
    bucket_label: str       # e.g. "50-55"
    city: str
    buy_price: float        # the ask price we'd pay

    def __str__(self) -> str:
        return (
            f"{self.ticker} | city={self.city} bucket={self.bucket_label} "
            f"NOAA={self.noaa_prob:.3f} mkt={self.market_prob:.3f} "
            f"edge={self.edge:+.3f}"
        )


class EdgeCalculator:
    """
    Given a list of active Kalshi weather markets and a NOAA forecast,
    returns TradeSignal objects for every market where edge > min_edge.
    """

    def __init__(self, min_edge: float = 0.10, bucket_size: int = 5):
        self._min_edge   = min_edge
        self._bucket_size = bucket_size

    def find_signals(
        self,
        markets: List[MarketInfo],
        forecasts: Dict[str, Optional[TempForecast]],
    ) -> List[TradeSignal]:
        signals: List[TradeSignal] = []

        for market in markets:
            city = self._city_from_ticker(market.ticker)
            if city is None:
                continue

            fc = forecasts.get(city)
            if fc is None:
                continue

            bucket = self._bucket_from_ticker(market.ticker)
            if bucket is None:
                continue

            noaa_prob = fc.bucket_probs(self._bucket_size).get(bucket, 0.0)
            if noaa_prob == 0.0:
                continue

            # Use ask price — that's what we pay to buy YES
            market_prob = market.yes_ask
            if market_prob is None:
                market_prob = market.last_price
            if market_prob is None:
                continue

            edge = noaa_prob - market_prob

            log.debug(
                "%s  noaa=%.3f  mkt=%.3f  edge=%+.3f",
                market.ticker, noaa_prob, market_prob, edge,
            )

            if edge >= self._min_edge:
                signals.append(TradeSignal(
                    ticker=market.ticker,
                    noaa_prob=noaa_prob,
                    market_prob=market_prob,
                    edge=edge,
                    bucket_label=bucket,
                    city=city,
                    buy_price=market_prob,
                ))

        signals.sort(key=lambda s: s.edge, reverse=True)

        if signals:
            log.info("Found %d signal(s):", len(signals))
            for s in signals:
                log.info("  %s", s)
        else:
            log.info("No signals above min_edge=%.2f", self._min_edge)

        return signals

    # ------------------------------------------------------------------
    # Ticker parsing
    # Kalshi weather tickers:  HIGHNY-25MAY30-B5055  or  HIGHNY-25MAY30-T50
    # Series prefixes vary by city:
    #   HIGHNY  = NYC high temp
    #   HIGHCHI = Chicago high temp
    #   HIGHLA  = LA high temp
    #   HIGHMI  = Miami high temp
    #   HIGHBOS = Boston high temp
    # ------------------------------------------------------------------

    _CITY_MAP = {
        "NY":  "new_york",
        "CHI": "chicago",
        "LA":  "los_angeles",
        "MI":  "miami",
        "BOS": "boston",
    }

    def _city_from_ticker(self, ticker: str) -> Optional[str]:
        """Extract city from ticker prefix, e.g. HIGHNY → new_york."""
        m = re.match(r"HIGH([A-Z]+)-", ticker)
        if not m:
            return None
        code = m.group(1)
        return self._CITY_MAP.get(code)

    def _bucket_from_ticker(self, ticker: str) -> Optional[str]:
        """
        Extract temperature bucket from ticker suffix.
        B5055  → "50-55"   (range bucket)
        T50    → None      (threshold-style, handled separately)
        """
        # Range bucket: B{lo}{hi} where each is 2 digits
        m = re.search(r"-B(\d{2,3})(\d{2,3})$", ticker)
        if m:
            lo = int(m.group(1))
            hi = int(m.group(2))
            return f"{lo}-{hi}"
        return None
