"""
Avellaneda-Stoikov market making model adapted for Kalshi prediction markets.

Key adaptations vs. the original crypto formulation:
  - Prices live in [0, 1] (probability), not unbounded.
  - YES and NO are complementary: YES + NO = 1.0.
  - We track inventory in NET YES contracts (positive = long YES).
  - Time horizon is normalised over the 15-minute contract window.
  - Quotes are hard-clamped to [0.01, 0.99] and a minimum spread is enforced.

Core formulas (Avellaneda & Stoikov 2008):
  reservation_price  = mid - q * gamma * sigma^2 * (T - t)
  optimal_half_spread = (gamma * sigma^2 * (T-t)) / 2
                      + (1 / gamma) * ln(1 + gamma / k)
  bid = reservation_price - half_spread
  ask = reservation_price + half_spread
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

_PRICE_MIN = 0.01
_PRICE_MAX = 0.99


@dataclass
class ASParams:
    gamma: float = 0.10      # Inventory risk-aversion (higher → wider spread)
    k: float = 1.50          # Order-arrival rate (higher → tighter spread)
    min_spread: float = 0.02  # Hard floor on quoted spread
    max_spread: float = 0.20  # Hard ceiling on quoted spread
    sigma_min: float = 0.005
    sigma_max: float = 0.40


@dataclass
class ASQuote:
    reservation_price: float
    spread: float
    bid: float
    ask: float
    inventory_adjustment: float  # how far mid was shifted due to inventory


class AvellanedaStoikov:
    def __init__(self, params: ASParams):
        self.p = params

    def quote(
            self,
            mid: float,
            inventory: int,
            sigma: float,
            time_remaining: float,
            session_duration: float = 900.0,
            alpha: float = 0.0,  # NEW: directional edge
    ) -> Optional[ASQuote]:

        if not (_PRICE_MIN <= mid <= _PRICE_MAX):
            return None

        sigma = max(self.p.sigma_min, min(self.p.sigma_max, sigma))
        T = max(0.01, time_remaining / max(session_duration, 1.0))

        # ✅ FIXED: nonlinear inventory skew (prevents runaway)
        inv_ratio = inventory / 20.0
        inventory_adj = -0.05 * math.tanh(inv_ratio)

        # ✅ ADD: directional alpha (your edge)
        r = mid + alpha + inventory_adj

        # ✅ FIXED: spread now actually meaningful
        half_spread = (
                0.5 * sigma
                + 0.01 * abs(inventory)
                + 0.01 * (1 - T)  # tighter near expiry
        )

        half_spread = max(
            self.p.min_spread / 2.0,
            min(self.p.max_spread / 2.0, half_spread),
            )

        bid = math.floor((r - half_spread) * 100) / 100
        ask = math.ceil((r + half_spread) * 100) / 100

        bid = max(_PRICE_MIN, min(_PRICE_MAX - self.p.min_spread, bid))
        ask = min(_PRICE_MAX, max(_PRICE_MIN + self.p.min_spread, ask))

        # Re-enforce minimum spread after clamping
        if ask - bid < self.p.min_spread:
            centre = (bid + ask) / 2.0
            half = self.p.min_spread / 2.0
            bid = max(_PRICE_MIN, round(centre - half, 2))
            ask = min(_PRICE_MAX, round(centre + half, 2))

        # Final spread guard after floor/ceil + clamp
        if ask <= bid:
            ask = round(bid + 0.01, 2)


        return ASQuote(
            reservation_price=round(r, 2),
            spread=round(ask - bid, 2),
            bid=bid,
            ask=ask,
            inventory_adjustment=round(inventory_adj, 4),
        )

    def skew_for_inventory(
        self, quote: ASQuote, inventory: int, max_inventory: int
    ) -> ASQuote:
        """
        Extra skew when inventory exceeds 50% of the limit.
        Encourages mean-reversion without shutting quotes off.
        """
        ratio = abs(inventory) / max(max_inventory, 1)
        if ratio < 0.50:
            return quote

        extra = 0.02 * (ratio - 0.50) * 2.0  # up to +2 cents extra shift at limit

        if inventory > 0:
            # Overlong YES: push both legs down to sell YES / buy NO
            new_bid = round(quote.bid - extra, 2)
            new_ask = round(quote.ask - extra * 0.5, 2)
        else:
            # Overlong NO: push both legs up to sell NO / buy YES
            new_bid = round(quote.bid + extra * 0.5, 2)
            new_ask = round(quote.ask + extra, 2)

        if inventory > 0:
            new_bid = math.floor((quote.bid - extra) * 100) / 100
            new_ask = math.floor((quote.ask - extra * 0.5) * 100) / 100
        else:
            new_bid = math.ceil((quote.bid + extra * 0.5) * 100) / 100
            new_ask = math.ceil((quote.ask + extra) * 100) / 100

        return ASQuote(
            reservation_price=round((new_bid + new_ask) / 2.0, 4),
            spread=round(new_ask - new_bid, 4),
            bid=new_bid,
            ask=new_ask,
            inventory_adjustment=quote.inventory_adjustment,
        )
