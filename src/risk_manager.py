import logging
import time
from typing import Dict

log = logging.getLogger(__name__)


class RiskManager:
    """
    Portfolio-level kill switches and per-market loss guards.

    Checks performed on every tick:
      1. Peak drawdown from session high-water mark.
      2. Cumulative daily loss vs. hard limit.
      3. Total position size across all markets.

    Per-market checks:
      4. Per-market loss vs. individual limit.
    """

    def __init__(
        self,
        max_total_position: int = 200,
        max_drawdown: float = 0.15,
        max_loss_per_market: float = 50.0,
        daily_loss_limit: float = 200.0,
        initial_balance: float = 0.0,
    ):
        self._max_total_position = max_total_position
        self._max_drawdown = max_drawdown
        self._max_loss_per_market = max_loss_per_market
        self._daily_loss_limit = daily_loss_limit

        self._session_start_balance: float = initial_balance
        self._peak_balance: float = initial_balance
        self._market_pnl: Dict[str, float] = {}
        self._halted: bool = False
        self._session_start: float = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialise(self, balance: float):
        self._session_start_balance = balance
        self._peak_balance = balance
        log.info("Risk manager initialised | balance=%.2f", balance)

    def check(self, current_balance: float, total_exposure: int) -> bool:
        """
        Returns True when trading is permitted.
        Sets self._halted permanently if a kill condition fires.
        """
        if self._halted:
            return False

        # Update watermark
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance

        # Drawdown guard
        if self._peak_balance > 0:
            dd = (self._peak_balance - current_balance) / self._peak_balance
            if dd >= self._max_drawdown:
                log.error(
                    "RISK HALT — drawdown %.1f%% >= limit %.1f%%",
                    dd * 100, self._max_drawdown * 100,
                )
                self._halted = True
                return False

        # Daily loss guard
        session_pnl = current_balance - self._session_start_balance
        if session_pnl <= -self._daily_loss_limit:
            log.error(
                "RISK HALT — daily loss $%.2f >= limit $%.2f",
                abs(session_pnl), self._daily_loss_limit,
            )
            self._halted = True
            return False

        # Position-size throttle (warn but don't halt)
        if total_exposure > self._max_total_position:
            log.warning(
                "Total exposure %d > max %d — skipping new quotes",
                total_exposure, self._max_total_position,
            )
            return False

        return True

    def market_allowed(self, ticker: str) -> bool:
        loss = self._market_pnl.get(ticker, 0.0)
        if loss <= -self._max_loss_per_market:
            log.warning("Market %s suspended | loss=%.2f", ticker, abs(loss))
            return False
        return True

    def record_market_pnl(self, ticker: str, delta: float):
        self._market_pnl[ticker] = self._market_pnl.get(ticker, 0.0) + delta

    def is_halted(self) -> bool:
        return self._halted

    def reset_daily(self, current_balance: float):
        self._session_start = time.time()
        self._session_start_balance = current_balance
        self._peak_balance = current_balance
        self._market_pnl.clear()
        self._halted = False
        log.info("Daily risk reset | balance=%.2f", current_balance)
