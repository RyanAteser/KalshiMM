"""
Polymarket BTC 15-Minute Mean Reversion Bot.

Loop (one tick per 15-min UTC boundary):
  1. Fetch close price of the market that just ended → update bar buffer.
  2. Compute lag-1 direction → generate signal.
  3. Check OOS edge > spread cost threshold.
  4. If live: enter the next 15-min market (buy YES or NO).
  5. Previous position auto-settles on Polymarket when its market resolves.

Weekly recalibration:
  Pulls 90 days of trade history, rebuilds bars, reruns validation pipeline.
  If edge drops below threshold the bot pauses until next Sunday's check.

Position management:
  Each market is held until resolution (binary 0 or 1 payout).
  This is equivalent to "hold bar T → bar T+1" when the average per-bar
  duration equals one 15-min window.
  PnL per trade ≈ (resolution_price - entry_price) × shares
                = (1_if_correct - entry) × shares
"""

from __future__ import annotations

import logging
import math
import time
from typing import List, Optional

from .bar_builder import Bar, BarBuilder
from .mean_reversion import MeanReversionValidator, ValidationResult
from .polymarket_client import PolymarketCLOBClient, PolyMarket

log = logging.getLogger(__name__)

_BAR_SECONDS       = 900            # 15 minutes
_RECALIBRATE_SECS  = 7 * 86_400    # weekly
_DEFAULT_MIN_EDGE  = 0.002          # 0.2% OOS mean per bucket
_DEFAULT_SIZE_PCT  = 0.02           # 2% of capital per trade
_DEFAULT_SPREAD    = 0.015          # 1.5-cent round-trip cost
_MIN_HISTORY_BARS  = 200
_HISTORY_DAYS      = 90


class MeanReversionBot:
    """
    Live Polymarket mean reversion bot for BTC 15-minute Up/Down markets.
    Instantiate with the full config dict from config.yaml.
    """

    def __init__(self, config: dict):
        self._cfg = config
        paper = config.get("paper_trade", True)

        mr = config.get("mean_reversion", {})
        self._client    = PolymarketCLOBClient(paper_trade=paper)
        self._builder   = BarBuilder(bar_seconds=_BAR_SECONDS)
        self._validator = MeanReversionValidator(
            min_edge=mr.get("min_edge", _DEFAULT_MIN_EDGE),
            min_bars=mr.get("min_bars", _MIN_HISTORY_BARS),
        )

        self._spread_cost: float = mr.get("spread_cost",   _DEFAULT_SPREAD)
        self._size_pct:    float = mr.get("position_size_pct", _DEFAULT_SIZE_PCT)
        self._capital:     float = float(mr.get("initial_capital", 1_000.0))

        self._validation: Optional[ValidationResult] = None
        self._last_cal:   float = 0.0
        self._open_pos: Optional[dict] = None  # {market, token_id, side, size_usd, entry_price}
        self._total_pnl:  float = 0.0
        self._n_trades:   int   = 0
        self._running:    bool  = False

    # ------------------------------------------------------------------
    # Lifecycle

    def start(self):
        log.info("MeanReversionBot starting — paper=%s", self._client._paper)
        self._calibrate()
        self._running = True
        self._loop()

    def stop(self):
        self._running = False
        log.info(
            "Bot stopped | trades=%d  total_pnl=$%.2f  capital=$%.2f",
            self._n_trades, self._total_pnl, self._capital,
        )

    # ------------------------------------------------------------------
    # Main loop — sleeps to the next 15-min UTC boundary, then ticks

    def _loop(self):
        while self._running:
            now      = time.time()
            next_bar = math.ceil(now / _BAR_SECONDS) * _BAR_SECONDS
            sleep    = max(1.0, next_bar - now)
            log.info(
                "Sleeping %.0fs → next bar %s UTC",
                sleep, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(next_bar)),
            )
            time.sleep(sleep)
            if not self._running:
                break
            try:
                self._tick(int(next_bar))
            except Exception as exc:
                log.error("Tick error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # One 15-min cycle

    def _tick(self, bar_end_ts: int):
        # Weekly recalibration
        if time.time() - self._last_cal >= _RECALIBRATE_SECS:
            self._calibrate()

        markets = self._client.find_btc_15min_markets(active_only=True)
        if not markets:
            log.warning("No BTC 15-min markets found — skipping tick")
            return

        now = time.time()

        # Market that just closed (end_date closest to bar_end_ts - bar_seconds)
        prev_bar_ts = bar_end_ts - _BAR_SECONDS
        closed_market: Optional[PolyMarket] = min(
            markets,
            key=lambda m: abs(m.end_date_ts - prev_bar_ts),
            default=None,
        )

        # Market opening now (next one to trade)
        next_market: Optional[PolyMarket] = next(
            (m for m in markets if m.end_date_ts > now + 60),
            None,
        )

        # ---- Step 1: close price of the bar that just ended ----
        if closed_market:
            p = self._client.get_mid_price(closed_market.yes_token_id)
            if p:
                self._builder.feed_trade(p, bar_end_ts - 5)
                log.debug("Bar close price: %.4f (market %s)", p, closed_market.condition_id[:10])

        # ---- Step 2-3: signal + EV check ----
        if self._validation is None or not self._validation.edge_is_live:
            log.info("Edge not validated — monitoring only")
            return

        sig = self._builder.signal()
        if sig is None:
            log.info("Insufficient bar history for signal (need 2+ bars)")
            return

        ev = self._validator.oos_ev(self._validation, sig)
        log.info(
            "Signal: %+d  lag-1=%.4f  OOS_EV=%.4f  spread_cost=%.4f",
            sig, self._builder.lag_1() or 0.0, ev, self._spread_cost,
        )

        if ev < self._spread_cost:
            log.info("EV %.4f < spread cost %.4f — skipping trade", ev, self._spread_cost)
            return

        # ---- Step 4: enter next market ----
        if next_market is None:
            log.warning("No next market available — skipping entry")
            return

        self._enter(next_market, sig)

    # ------------------------------------------------------------------
    # Position management

    def _enter(self, market: PolyMarket, signal: int):
        """Open a position in the new market based on the mean reversion signal."""
        # signal +1 → bet YES goes up → buy YES token
        # signal -1 → bet YES goes down → buy NO token
        token_id = market.yes_token_id if signal == 1 else market.no_token_id
        side      = "BUY"
        size_usd  = round(self._capital * self._size_pct, 2)
        direction = "YES" if signal == 1 else "NO"

        result = self._client.place_marketable_limit(
            token_id=token_id, side=side, size_usd=size_usd,
        )

        if result.get("success"):
            entry_p = result.get("price", 0.5)
            self._open_pos = {
                "market":      market,
                "token_id":    token_id,
                "side":        direction,
                "size_usd":    size_usd,
                "entry_price": entry_p,
                "signal":      signal,
                "bar_ts":      int(time.time()),
            }
            log.info(
                "ENTERED %s  market=…%s  size=$%.2f  @%.2f  tte=%.0fs",
                direction, market.condition_id[-8:],
                size_usd, entry_p, market.time_to_expiry,
            )
        else:
            log.error("Entry failed: %s", result.get("error"))

    def _record_resolution(self, resolved_yes: bool):
        """
        Call this when the open position's market resolves.
        resolved_yes=True → YES token pays $1; False → pays $0.
        """
        if self._open_pos is None:
            return

        pos        = self._open_pos
        entry_p    = pos["entry_price"]
        shares     = pos["size_usd"] / entry_p
        direction  = pos["side"]

        if direction == "YES":
            resolution_p = 1.0 if resolved_yes else 0.0
        else:
            resolution_p = 0.0 if resolved_yes else 1.0   # NO pays 1 when YES fails

        pnl = (resolution_p - entry_p) * shares
        self._total_pnl += pnl
        self._capital   += pnl
        self._n_trades  += 1
        self._open_pos   = None

        log.info(
            "SETTLED %s  resolved=%s  entry=%.2f  pnl=$%.2f  total_pnl=$%.2f",
            direction, "YES" if resolved_yes else "NO",
            entry_p, pnl, self._total_pnl,
        )

    # ------------------------------------------------------------------
    # Calibration

    def _calibrate(self):
        """Pull 90 days of trade history, rebuild bars, run validation."""
        log.info("Calibrating on last %d days of BTC 15-min history…", _HISTORY_DAYS)
        start_ts = int(time.time()) - _HISTORY_DAYS * 86_400

        markets = self._client.find_btc_15min_markets(active_only=False, limit=500)
        if not markets:
            log.warning("No markets found for calibration")
            return

        scratch = BarBuilder(bar_seconds=_BAR_SECONDS)
        all_bars: List[Bar] = []

        processed = 0
        for market in markets:
            if market.end_date_ts > 0 and market.end_date_ts < start_ts:
                continue
            trades = self._client.get_trades(
                token_id=market.yes_token_id,
                start_ts=start_ts,
                limit=10_000,
            )
            for trade in trades:
                bar = scratch.feed_trade(trade.price, trade.timestamp)
                if bar:
                    all_bars.append(bar)
            processed += 1
            if processed % 50 == 0:
                log.info("Calibration: %d markets processed → %d bars", processed, len(all_bars))

        all_bars.sort(key=lambda b: b.bar_start_ts)
        log.info("Calibration bars assembled: %d", len(all_bars))

        if len(all_bars) < _MIN_HISTORY_BARS:
            log.warning("Too few bars (%d) for robust validation", len(all_bars))
            return

        self._validation = self._validator.validate(all_bars)
        self._last_cal   = time.time()

        # Seed the live bar buffer with the most recent bars
        self._builder.load_historical(all_bars[-10:])

        log.info(
            "Calibration done | edge_live=%s  sharpe=%.2f  win_rate=%.1f%%",
            self._validation.edge_is_live,
            self._validation.annualized_sharpe,
            self._validation.win_rate * 100,
        )
