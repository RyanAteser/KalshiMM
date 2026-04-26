import logging
from collections import defaultdict
from typing import Dict

from .models import Action, Position, Side

log = logging.getLogger(__name__)


class InventoryManager:
    """
    Tracks YES / NO positions and realised PnL for each market ticker.
    Provides net-inventory checks used by the A-S engine and risk layer.
    """

    def __init__(self, max_position: int = 50):
        self._max = max_position
        self._positions: Dict[str, Position] = {}
        self._realized_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_position(self, ticker: str) -> Position:
        if ticker not in self._positions:
            self._positions[ticker] = Position(ticker=ticker)
        return self._positions[ticker]

    def net_inventory(self, ticker: str) -> int:
        return self.get_position(ticker).net_position

    def can_buy_yes(self, ticker: str) -> bool:
        return self.net_inventory(ticker) < self._max

    def can_buy_no(self, ticker: str) -> bool:
        return self.net_inventory(ticker) > -self._max

    def total_exposure(self) -> int:
        return sum(abs(p.net_position) for p in self._positions.values())

    def total_realized_pnl(self) -> float:
        return self._realized_pnl

    # ------------------------------------------------------------------
    # State mutations
    # ------------------------------------------------------------------

    def record_fill(
        self,
        ticker: str,
        side: Side,
        action: Action,
        yes_price: float,
        count: int,
    ):
        """
        Update position and realised PnL when an order is filled.
        yes_price is always the YES-equivalent price (0-1), matching Kalshi's API.
        """
        pos = self.get_position(ticker)

        if side == Side.YES:
            if action == Action.BUY:
                total = pos.yes_count * pos.avg_yes_cost + count * yes_price
                pos.yes_count += count
                pos.avg_yes_cost = total / pos.yes_count if pos.yes_count else 0.0
            else:
                pnl = count * (yes_price - pos.avg_yes_cost)
                pos.realized_pnl += pnl
                self._realized_pnl += pnl
                pos.yes_count = max(0, pos.yes_count - count)
        else:
            no_price = 1.0 - yes_price
            if action == Action.BUY:
                total = pos.no_count * pos.avg_no_cost + count * no_price
                pos.no_count += count
                pos.avg_no_cost = total / pos.no_count if pos.no_count else 0.0
            else:
                pnl = count * (no_price - pos.avg_no_cost)
                pos.realized_pnl += pnl
                self._realized_pnl += pnl
                pos.no_count = max(0, pos.no_count - count)

        log.debug(
            "fill %s %s %s x%d @%.4f | net=%+d",
            ticker, action.value, side.value, count, yes_price,
            pos.net_position,
        )

    def sync_from_api(self, positions: dict):
        """Overwrite local state with positions fetched from the Kalshi API."""
        for ticker, data in positions.items():
            pos = self.get_position(ticker)
            pos.yes_count = data.get("yes_count", 0)
            pos.no_count = data.get("no_count", 0)
        log.info("Positions synced from API: %d markets", len(positions))

    def position_summary(self) -> Dict[str, dict]:
        return {
            ticker: {
                "yes": pos.yes_count,
                "no": pos.no_count,
                "net": pos.net_position,
                "realized_pnl": round(pos.realized_pnl, 4),
            }
            for ticker, pos in self._positions.items()
            if pos.yes_count > 0 or pos.no_count > 0
        }
