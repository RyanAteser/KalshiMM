import asyncio
import json
import logging
import time
from collections import deque
from threading import Lock, Thread
from typing import Optional

import websockets  # type: ignore

log = logging.getLogger(__name__)

_WS_URL = "wss://advanced-trade-ws.coinbase.com/"
_PRODUCT = "BTC-USD"
_SUBSCRIBE = json.dumps({
    "type": "subscribe",
    "product_ids": [_PRODUCT],
    "channel": "market_trades",
})


class BTCPriceFeed:
    """
    Streams BTC-USD trades from Coinbase Advanced Trade WebSocket.
    Exposes last price and a rolling window for volatility estimation.
    Reconnects automatically on failure.
    """

    def __init__(self, window: int = 100):
        self._window = window
        self._prices: deque = deque(maxlen=window)
        self._lock = Lock()
        self._last_price: Optional[float] = None
        self._last_update: float = 0.0
        self._running = False
        self._thread: Optional[Thread] = None

    def start(self):
        self._running = True
        self._thread = Thread(target=self._run, daemon=True, name="btc-feed")
        self._thread.start()
        log.info("BTC price feed started")

    def stop(self):
        self._running = False

    def last_price(self) -> Optional[float]:
        return self._last_price

    def is_stale(self, max_age: float = 30.0) -> bool:
        return time.time() - self._last_update > max_age

    def prices(self) -> list:
        with self._lock:
            return list(self._prices)

    def volatility(self, window: int = 20) -> float:
        """
        Rolling std of log-returns, used as sigma input for A-S model.
        Scaled to per-15-minute units to match KXBTC15M session length.
        Returns a fallback of 0.02 when insufficient data.
        """
        import numpy as np
        prices = self.prices()
        if len(prices) < 3:
            return 0.02
        arr = np.array(prices[-window:], dtype=float)
        log_returns = np.diff(np.log(arr))
        if len(log_returns) == 0:
            return 0.02
        raw_sigma = float(np.std(log_returns))
        # Scale BTC price vol to YES-probability vol (rough heuristic: /BTC_price * 50000)
        btc = self._last_price or 50000.0
        prob_sigma = raw_sigma * (btc / 50000.0) * 0.5
        return max(0.005, min(0.40, prob_sigma))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._stream())
        loop.close()

    async def _stream(self):
        backoff = 1
        while self._running:
            try:
                async with websockets.connect(_WS_URL, ping_interval=30) as ws:
                    await ws.send(_SUBSCRIBE)
                    backoff = 1
                    async for raw in ws:
                        if not self._running:
                            return
                        self._handle(raw)
            except Exception as exc:
                log.warning("BTC feed lost connection: %s — retry in %ds", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _handle(self, raw: str):
        try:
            msg = json.loads(raw)
            if msg.get("channel") != "market_trades":
                return
            for event in msg.get("events", []):
                for trade in event.get("trades", []):
                    price = float(trade["price"])
                    self._last_price = price
                    self._last_update = time.time()
                    with self._lock:
                        self._prices.append(price)
        except Exception:
            pass
