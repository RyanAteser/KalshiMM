from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class Side(str, Enum):
    YES = "yes"
    NO = "no"


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    PARTIAL = "partial_fill"


@dataclass
class MarketInfo:
    ticker: str
    strike_price: float
    close_ts: float
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    last_price: Optional[float] = None
    volume: int = 0

    @property
    def mid_price(self) -> Optional[float]:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2.0
        return self.yes_bid or self.yes_ask or self.last_price

    @property
    def spread(self) -> Optional[float]:
        if self.yes_bid is not None and self.yes_ask is not None:
            return self.yes_ask - self.yes_bid
        return None

    @property
    def time_to_expiry(self) -> float:
        return max(0.0, self.close_ts - time.time())


@dataclass
class Order:
    order_id: str
    ticker: str
    side: Side
    action: Action
    price: float
    count: int
    status: OrderStatus = OrderStatus.PENDING
    filled_count: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class Position:
    ticker: str
    yes_count: int = 0
    no_count: int = 0
    avg_yes_cost: float = 0.0
    avg_no_cost: float = 0.0
    realized_pnl: float = 0.0

    @property
    def net_position(self) -> int:
        """Positive = net long YES, negative = net long NO."""
        return self.yes_count - self.no_count


@dataclass
class Quote:
    ticker: str
    bid_price: float
    ask_price: float
    size: int
    reservation_price: float
    spread: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    filled_count: int = 0
    filled_price: Optional[float] = None
    error: Optional[str] = None
