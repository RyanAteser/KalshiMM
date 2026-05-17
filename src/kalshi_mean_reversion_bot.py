"""
Kalshi KXBTC15M Mean Reversion Bot.

Signal source
-------------
BTC/USDT 15-minute log returns from the existing BTCPriceFeed.  At each
15-minute UTC boundary the bot samples the live BTC price, appends it to
the BarBuilder, and reads the lag-1 signal.

Entry logic
-----------
  signal = -sign(lag-1 BTC log return)
  +1 → BTC expected to rebound up → buy YES on near-the-money KXBTC15M
  -1 → BTC expected to retrace down → buy NO on near-the-money KXBTC15M

"Near the money" = active market whose YES mid price is closest to 0.50.
That market has the highest sensitivity to BTC moves and the most trading
activity, giving the best liquidity for a directional position.

Position lifecycle
------------------
KXBTC15M contracts settle automatically when BTC price is fixed at the
close time.  We hold to resolution — no manual exit needed.  One trade
per 15-min bar, sized at `order_size` contracts from config.

Calibration
-----------
On startup and then weekly: pulls 90 days of BTC/USD 15-min OHLC from
Kraken (primary, no auth) with automatic fallback to CryptoCompare, then
runs the full MeanReversionValidator pipeline. Trading is enabled only when
both OOS bucket means clear `min_edge`.
"""

from __future__ import annotations

import logging
import math
import time
from typing import List, Optional

import requests

from .bar_builder import Bar, BarBuilder
from .btc_feed import BTCPriceFeed
from .kalshi_client import KalshiMMClient
from .mean_reversion import MeanReversionValidator, ValidationResult
from .models import Action, MarketInfo, Side

log = logging.getLogger(__name__)

_BAR_SECONDS       = 900            # 15 minutes
_RECALIBRATE_SECS  = 7 * 86_400    # weekly
_MIN_HISTORY_BARS  = 200
_DEFAULT_HISTORY   = 90             # days of BTC data for calibration
_DEFAULT_MIN_TTE   = 300            # skip markets with < 5 min to expiry


# ---------------------------------------------------------------------------
# BTC OHLC helpers (calibration only — no auth required)
#
# Primary:  Kraken — most relevant because Kalshi KXBTC15M settlement uses
#           a Kraken-based BTC index price.  Paginates via 'since'; returns
#           up to 720 bars per call (720 × 15 min = 7.5 days → ~12 calls).
#
# Fallback: CryptoCompare — cross-exchange aggregated BTC/USD price.
#           Paginates backward via 'toTs'; returns 2000 × 15-min bars per
#           call (~20 days → ~5 calls for 90 days).
# ---------------------------------------------------------------------------

def fetch_btc_15min_bars(days: int = 90) -> List[Bar]:
    """
    Fetch BTC/USD 15-min OHLC bars.  Tries Kraken first; falls back to
    CryptoCompare if Kraken is unreachable (e.g. some cloud environments
    block exchange IPs).  Returns bars sorted chronologically.
    """
    bars = _fetch_kraken(days)
    if len(bars) >= 100:
        return bars
    log.warning("Kraken returned %d bars — falling back to CryptoCompare", len(bars))
    return _fetch_cryptocompare(days)


def _fetch_kraken(days: int) -> List[Bar]:
    """
    Kraken public OHLC endpoint.
    Row format: [time, open, high, low, close, vwap, volume, count]
    Rate limit: ~1 public req/s.
    """
    end_ts   = int(time.time())
    start_ts = end_ts - days * 86_400
    url      = "https://api.kraken.com/0/public/OHLC"
    bars: List[Bar] = []
    since = start_ts

    while since < end_ts:
        try:
            resp = requests.get(
                url,
                params={"pair": "XBTUSD", "interval": 15, "since": since},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("Kraken request failed: %s", exc)
            break

        if data.get("error"):
            log.warning("Kraken API error: %s", data["error"])
            break

        result   = data.get("result", {})
        pair_key = next((k for k in result if k != "last"), None)
        if not pair_key:
            break

        candles = result[pair_key]
        if not candles:
            break

        for row in candles:
            try:
                open_ts = int(row[0])
                if open_ts < start_ts:
                    continue
                bars.append(Bar(
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[6]),
                    bar_start_ts=open_ts,
                    bar_end_ts=open_ts + 900,
                ))
            except Exception:
                pass

        last = result.get("last")
        if not last or int(last) <= since:
            break
        since = int(last)
        time.sleep(0.5)

    bars.sort(key=lambda b: b.bar_start_ts)
    log.info("Kraken: %d BTC/USD 15-min bars (%d days)", len(bars), days)
    return bars


def _fetch_cryptocompare(days: int) -> List[Bar]:
    """
    CryptoCompare histominute endpoint with aggregate=15.
    Each call returns up to 2000 fifteen-minute bars (~20.8 days).
    Paginates backward by decrementing 'toTs'.
    No API key required for the free tier.
    """
    end_ts   = int(time.time())
    start_ts = end_ts - days * 86_400
    url      = "https://min-api.cryptocompare.com/data/v2/histominute"
    bars: List[Bar] = []
    to_ts = end_ts

    while to_ts > start_ts:
        try:
            resp = requests.get(
                url,
                params={
                    "fsym":      "BTC",
                    "tsym":      "USD",
                    "limit":     2000,
                    "aggregate": 15,
                    "toTs":      to_ts,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.error("CryptoCompare request failed: %s", exc)
            break

        if data.get("Response") != "Success":
            log.error("CryptoCompare error: %s", data.get("Message"))
            break

        candles = data["Data"]["Data"]
        if not candles:
            break

        for row in candles:
            try:
                ts = int(row["time"])
                if ts < start_ts:
                    continue
                bars.append(Bar(
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volumefrom", 0)),
                    bar_start_ts=ts,
                    bar_end_ts=ts + 900,
                ))
            except Exception:
                pass

        oldest_ts = int(candles[0]["time"])
        if oldest_ts <= start_ts:
            break
        to_ts = oldest_ts - 1
        time.sleep(0.3)

    bars.sort(key=lambda b: b.bar_start_ts)
    log.info("CryptoCompare: %d BTC/USD 15-min bars (%d days)", len(bars), days)
    return bars


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class KalshiMeanReversionBot:
    """
    Mean reversion bot for Kalshi KXBTC15M (BTC 15-minute prediction markets).

    Integrates with the existing KalshiMMClient and BTCPriceFeed.
    Does not use the Avellaneda-Stoikov engine — this is a directional
    single-entry strategy, not a market maker.
    """

    def __init__(self, config: dict):
        self._cfg = config
        paper = config.get("paper_trade", True)

        tr = config.get("trading", {})
        mr = config.get("mean_reversion", {})

        self._client      = KalshiMMClient(paper_trade=paper)
        self._btc         = BTCPriceFeed(window=200)
        self._builder     = BarBuilder(bar_seconds=_BAR_SECONDS)
        self._validator   = MeanReversionValidator(
            min_edge=mr.get("min_edge", 0.002),
            min_bars=mr.get("min_bars", _MIN_HISTORY_BARS),
        )

        self._series:       str   = tr.get("series", "KXBTC15M")
        self._order_size:   int   = tr.get("order_size", 5)
        self._min_tte:      float = float(tr.get("min_time_to_expiry", _DEFAULT_MIN_TTE))
        self._spread_cost:  float = mr.get("spread_cost", 0.015)
        self._history_days: int   = mr.get("history_days", _DEFAULT_HISTORY)

        self._validation: Optional[ValidationResult] = None
        self._last_cal:   float = 0.0
        self._n_trades:   int   = 0
        self._running:    bool  = False

    # ------------------------------------------------------------------
    # Lifecycle

    def start(self):
        log.info(
            "KalshiMeanReversionBot starting | series=%s paper=%s",
            self._series, self._client._paper,
        )
        self._btc.start()
        time.sleep(3)   # let the BTC feed warm up
        self._calibrate()
        self._running = True
        self._loop()

    def stop(self):
        self._running = False
        self._btc.stop()
        log.info("Bot stopped | total_trades=%d", self._n_trades)

    # ------------------------------------------------------------------
    # Main loop — wakes at each 15-min UTC boundary

    def _loop(self):
        while self._running:
            now      = time.time()
            next_bar = math.ceil(now / _BAR_SECONDS) * _BAR_SECONDS
            sleep    = max(1.0, next_bar - now)
            log.info(
                "Sleeping %.0fs → boundary %s UTC",
                sleep, time.strftime("%H:%M:%S", time.gmtime(next_bar)),
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

        # ── 1. Sample BTC price at this boundary ──────────────────────
        btc_price = self._btc.last_price()
        if btc_price is None:
            log.warning("BTC feed has no price — skipping tick")
            return

        # Feed the boundary price into the bar builder.
        # Because bar_builder uses fixed-width 900-second windows keyed
        # on timestamp, two adjacent boundary prices close one bar.
        self._builder.feed_trade(btc_price, bar_end_ts - 1)

        # ── 2. Signal ─────────────────────────────────────────────────
        if self._validation is None or not self._validation.edge_is_live:
            log.info("Edge not validated — monitoring only (BTC=%.2f)", btc_price)
            return

        sig = self._builder.signal()
        lag = self._builder.lag_1()

        if sig is None or lag is None:
            log.info("Insufficient bar history for signal — need 2+ boundaries")
            return

        # ── 3. Gate on OOS EV vs spread cost ──────────────────────────
        ev = self._validator.oos_ev(self._validation, sig)
        log.info(
            "BTC=%.2f  lag-1=%.4f%%  signal=%+d  OOS_EV=%.4f  spread=%.4f",
            btc_price, lag * 100, sig, ev, self._spread_cost,
        )

        if ev < self._spread_cost:
            log.info("EV %.4f < spread %.4f — skipping trade", ev, self._spread_cost)
            return

        # ── 4. Find near-the-money KXBTC15M market ────────────────────
        market = self._find_target_market()
        if market is None:
            log.warning("No suitable %s market — skipping", self._series)
            return

        # ── 5. Enter ──────────────────────────────────────────────────
        self._enter(market, sig)

    # ------------------------------------------------------------------
    # Market selection

    def _find_target_market(self) -> Optional[MarketInfo]:
        """
        Return the active KXBTC15M market whose YES mid price is closest to 0.50.
        Excludes markets with < min_time_to_expiry seconds remaining.
        """
        markets = self._client.get_markets(self._series)
        candidates = [
            m for m in markets
            if m.mid_price is not None and m.time_to_expiry > self._min_tte
        ]
        if not candidates:
            # Credential-free paper mode: no live markets available — return a
            # synthetic near-the-money stub so signal + order logging still runs.
            if self._client._paper and self._client._client is None:
                mock = MarketInfo(
                    ticker=f"{self._series}-PAPER-MOCK",
                    strike_price=0.0,
                    close_ts=time.time() + _BAR_SECONDS,
                    yes_bid=0.49,
                    yes_ask=0.51,
                    last_price=0.50,
                )
                log.info("Paper mode (no creds): using mock market %s", mock.ticker)
                return mock
            log.warning(
                "No %s markets with tte>%.0fs and a mid price",
                self._series, self._min_tte,
            )
            return None

        target = min(candidates, key=lambda m: abs((m.mid_price or 0.5) - 0.5))
        log.info(
            "Target market: %s  mid=%.3f  tte=%.0fs",
            target.ticker, target.mid_price or 0.0, target.time_to_expiry,
        )
        return target

    # ------------------------------------------------------------------
    # Order entry

    def _enter(self, market: MarketInfo, signal: int):
        """
        Place a directional order.

        signal=+1: BTC expected to rebound up → buy YES
                   (bet BTC stays above the strike at resolution)
        signal=-1: BTC expected to retrace down → buy NO
                   (bet BTC falls below the strike)
        """
        ticker = market.ticker

        if signal == 1:
            # Buy YES — take the ask + 1 tick slippage
            yes_price = round(min(0.99, (market.yes_ask or (market.mid_price or 0.5) + 0.01) + 0.01), 2)
            side, action = Side.YES, Action.BUY
            direction = "YES"
        else:
            # Buy NO — equivalent to selling YES below current bid
            # Kalshi takes yes_price_dollars even for NO orders; place_order()
            # in KalshiMMClient converts internally.
            yes_bid   = market.yes_bid or (market.mid_price or 0.5) - 0.01
            no_price  = round(max(0.01, 1.0 - yes_bid + 0.01), 2)  # NO price + slippage
            yes_price = no_price   # KalshiMMClient maps this correctly via Side.NO
            side, action = Side.NO, Action.BUY
            direction = "NO"

        log.info(
            "ENTERING %s | %s  mid=%.3f  price=%.2f  size=%d  tte=%.0fs",
            direction, ticker,
            market.mid_price or 0.0, yes_price,
            self._order_size, market.time_to_expiry,
        )

        result = self._client.place_order(
            ticker=ticker,
            side=side,
            action=action,
            price=yes_price,
            count=self._order_size,
        )

        if result.success:
            self._n_trades += 1
            log.info(
                "Order placed | id=%s  filled=%d  price=%.2f",
                result.order_id, result.filled_count, yes_price,
            )
        else:
            log.error("Order failed: %s", result.error)

    # ------------------------------------------------------------------
    # Weekly calibration

    def _calibrate(self):
        """
        Fetch BTC 15-min OHLC from Binance and run the full validation pipeline.
        Enables or disables trading based on whether the OOS edge clears the threshold.
        Pre-seeds the live BarBuilder with recent BTC prices so the first signal
        is available immediately after calibration.
        """
        log.info("Calibrating on %d days of BTC 15-min data from Binance…", self._history_days)
        bars = fetch_btc_15min_bars(days=self._history_days)

        if len(bars) < _MIN_HISTORY_BARS:
            log.warning(
                "Only %d bars fetched — need %d. Calibration skipped.",
                len(bars), _MIN_HISTORY_BARS,
            )
            return

        self._validation = self._validator.validate(bars)
        self._last_cal   = time.time()

        # Seed the live bar builder with recent bars so lag-1 is ready at the first tick
        self._builder.load_historical(bars[-10:])

        log.info(
            "Calibration complete | bars=%d  edge_live=%s  win_rate=%.1f%%  sharpe=%.2f",
            len(bars),
            self._validation.edge_is_live,
            self._validation.win_rate * 100,
            self._validation.annualized_sharpe,
        )
