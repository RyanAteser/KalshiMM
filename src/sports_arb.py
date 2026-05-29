"""
Sports arbitrage bot — direct port of the corrected JS simulator.

Strategy:
  Buy a side only when its ask ≤ C − avg(other), and only to:
    (a) build toward balance (equal shares on both sides), or
    (b) average down a side already held (new price is edge below avg cost).

  A soft imbalance cap (max_imb shares) prevents runaway directional exposure.

On Kalshi a sports market has two sides:
  Side A = YES  ("Will Team A win?")
  Side B = NO   (same market — effectively "Will Team B win?")
  NO ask = 1.0 − YES bid

When avg_A + avg_B < 1.0 the bot has locked in a guaranteed profit on every
matched pair regardless of outcome. Excess (unmatched) shares resolve based on
the actual result — the imbalance cap limits how much unhedged risk we carry.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger(__name__)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Parameter bundle
# ---------------------------------------------------------------------------

@dataclass
class ArbParams:
    C: float = 0.97               # combined ceiling: gate is C − avg(other)
    entry: float = 0.50           # max ask to open a position from scratch
    patience: int = 12            # min ticks between balance-building buys
    edge: float = 0.025           # avg-down threshold: buy when px < avg − edge
    clip: int = 30                # base clip size (shares)
    budget: float = 400.0         # soft budget; hard cap = budget × 3
    max_imb: int = 300            # max excess shares on either side


# ---------------------------------------------------------------------------
# Per-tick state snapshot (for UI / logging)
# ---------------------------------------------------------------------------

@dataclass
class TickSnapshot:
    tick: int
    ask_a: float
    ask_b: float
    qty_a: float
    qty_b: float
    avg_a: float
    avg_b: float
    pairs: float
    sum_avg: float          # avg_a + avg_b (combined cost of one matched pair)
    deployed: float
    locked: float           # pairs × (1 − sum_avg)  — guaranteed profit
    status: str             # human-readable next-action note


# ---------------------------------------------------------------------------
# Buy record
# ---------------------------------------------------------------------------

@dataclass
class BuyRecord:
    tick: int
    side: str               # 'A' or 'B'
    price: float
    size: float


# ---------------------------------------------------------------------------
# Final outcome
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    pairs: float
    sum_avg: float
    locked: float           # guaranteed P&L from matched pairs
    excess: float           # unmatched shares
    excess_side: str
    excess_avg: float
    excess_pnl: float       # P&L on the unmatched tail (depends on outcome)
    net_pnl: float
    a_won: bool             # did Side A resolve YES?
    deployed: float
    buys: List[BuyRecord]
    snaps: List[TickSnapshot]


# ---------------------------------------------------------------------------
# Core bot
# ---------------------------------------------------------------------------

class SportsArbBot:
    """
    Stateful bot. Feed it ticks one at a time via `tick()`.
    Call `settle(a_won)` at resolution to compute final P&L.
    """

    def __init__(self, params: ArbParams):
        self.p = params
        self._qty_a: float = 0.0
        self._cost_a: float = 0.0
        self._qty_b: float = 0.0
        self._cost_b: float = 0.0
        self._low_a: float = 1.0
        self._low_b: float = 1.0
        self._last_buy_tick: int = -999
        self._tick: int = 0
        self.buys: List[BuyRecord] = []
        self.snaps: List[TickSnapshot] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def tick(self, ask_a: float, ask_b: float) -> TickSnapshot:
        """
        Process one price tick. Returns a TickSnapshot.
        ask_a  = YES ask price
        ask_b  = NO ask price  (= 1 − YES bid, passed in from caller)
        """
        t = self._tick
        self._low_a = min(self._low_a, ask_a)
        self._low_b = min(self._low_b, ask_b)

        avg_a = self._cost_a / self._qty_a if self._qty_a > 0 else 0.0
        avg_b = self._cost_b / self._qty_b if self._qty_b > 0 else 0.0
        deployed = self._cost_a + self._cost_b
        status = ""

        hard_cap = self.p.budget * 3

        if deployed < hard_cap:
            if self._qty_a == 0 and self._qty_b == 0:
                # No position yet — open on whichever side is cheaper
                if ask_a <= self.p.entry and ask_a <= ask_b:
                    self._buy('A', ask_a, self.p.clip)
                    status = f"opened A @ ${ask_a:.3f}"
                elif ask_b <= self.p.entry:
                    self._buy('B', ask_b, self.p.clip)
                    status = f"opened B @ ${ask_b:.3f}"
                else:
                    status = f"waiting: need a side ≤ ${self.p.entry:.2f}"
            else:
                cap_a = self.p.C - avg_b
                cap_b = self.p.C - avg_a

                pick = self._best_eligible(ask_a, ask_b, avg_a, avg_b, cap_a, cap_b)

                if pick:
                    fresh_trough = pick["px"] <= pick["low"] + 0.003
                    deep         = pick["edge"] >= 0.06
                    patient      = (
                        (t - self._last_buy_tick) >= self.p.patience
                        and pick["balance"]
                        and pick["bal"] > 0
                    )

                    if fresh_trough or deep or patient:
                        size = self.p.clip * _clamp(1 + 8 * pick["edge"], 1, 5)
                        if pick["balance"] and pick["bal"] > 0:
                            size = max(size, min(pick["bal"], self.p.clip * 4))
                        size *= _clamp(1 - deployed / self.p.budget, 0.35, 1)
                        size = max(self.p.clip * 0.5, round(size))
                        self._buy(pick["side"], pick["px"], size)
                        status = (f"bought {pick['side']} {size:.0f}sh @ ${pick['px']:.3f} "
                                  f"({'balance' if pick['balance'] else 'avg-down'})")
                    else:
                        status = (f"scoping {pick['side']} ≤ ${pick['cap']:.3f} "
                                  f"(now ${pick['px']:.3f}) — waiting for dip")
                else:
                    heavy = 'A' if self._qty_a >= self._qty_b else 'B'
                    light = 'B' if heavy == 'A' else 'A'
                    cap   = cap_a if light == 'A' else cap_b
                    px    = ask_a if light == 'A' else ask_b
                    h_avg = avg_a if heavy == 'A' else avg_b
                    status = (f"holding {heavy} avg ${h_avg:.3f} · "
                              f"waiting for {light} ≤ ${cap:.3f} (now ${px:.3f})")
        else:
            status = "soft budget reached — conserving"

        # Recompute derived values after any buy
        avg_a = self._cost_a / self._qty_a if self._qty_a > 0 else 0.0
        avg_b = self._cost_b / self._qty_b if self._qty_b > 0 else 0.0
        pairs   = min(self._qty_a, self._qty_b)
        sum_avg = (avg_a + avg_b) if (self._qty_a > 0 and self._qty_b > 0) else 0.0
        locked  = pairs * (1.0 - sum_avg) if sum_avg < 1.0 else 0.0
        deployed = self._cost_a + self._cost_b

        snap = TickSnapshot(
            tick=t, ask_a=ask_a, ask_b=ask_b,
            qty_a=self._qty_a, qty_b=self._qty_b,
            avg_a=avg_a, avg_b=avg_b,
            pairs=pairs, sum_avg=sum_avg,
            deployed=deployed, locked=locked,
            status=status,
        )
        self.snaps.append(snap)
        self._tick += 1
        return snap

    def settle(self, a_won: bool) -> GameResult:
        """Compute final P&L after the market resolves."""
        avg_a = self._cost_a / self._qty_a if self._qty_a > 0 else 0.0
        avg_b = self._cost_b / self._qty_b if self._qty_b > 0 else 0.0
        pairs   = min(self._qty_a, self._qty_b)
        sum_avg = (avg_a + avg_b) if (self._qty_a > 0 and self._qty_b > 0) else 0.0
        locked  = pairs * (1.0 - sum_avg) if math.isfinite(locked := pairs * (1.0 - sum_avg)) else 0.0

        excess      = abs(self._qty_a - self._qty_b)
        excess_side = 'A' if self._qty_a > self._qty_b else 'B'
        excess_avg  = avg_a if excess_side == 'A' else avg_b
        ex_wins     = a_won if excess_side == 'A' else not a_won
        excess_pnl  = excess * ((1.0 - excess_avg) if ex_wins else -excess_avg)
        if not math.isfinite(excess_pnl):
            excess_pnl = 0.0

        net = locked + excess_pnl
        deployed = self._cost_a + self._cost_b

        result = GameResult(
            pairs=pairs, sum_avg=sum_avg, locked=locked,
            excess=excess, excess_side=excess_side, excess_avg=excess_avg,
            excess_pnl=excess_pnl, net_pnl=net, a_won=a_won,
            deployed=deployed, buys=list(self.buys), snaps=list(self.snaps),
        )

        side_won = 'A' if a_won else 'B'
        log.info(
            "SETTLE | side %s won | pairs=%.0f sumAvg=%.3f locked=$%.2f "
            "excess=%.0f×%s pnl=$%.2f NET=$%.2f on $%.2f deployed",
            side_won, pairs, sum_avg, locked,
            excess, excess_side, excess_pnl, net, deployed,
        )
        return result

    def reset(self):
        self.__init__(self.p)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _buy(self, side: str, price: float, size: float):
        if side == 'A':
            self._qty_a  += size
            self._cost_a += size * price
            self._low_a   = 1.0
        else:
            self._qty_b  += size
            self._cost_b += size * price
            self._low_b   = 1.0
        self._last_buy_tick = self._tick
        self.buys.append(BuyRecord(tick=self._tick, side=side, price=price, size=size))
        log.debug("BUY %s %g sh @ $%.3f (tick %d)", side, size, price, self._tick)

    def _best_eligible(
        self,
        ask_a: float, ask_b: float,
        avg_a: float, avg_b: float,
        cap_a: float, cap_b: float,
    ) -> Optional[dict]:
        """Return the better-scoring eligible side, or None."""
        candidates = []
        for side, px, cap, q, qo, avg_s, low in [
            ('A', ask_a, cap_a, self._qty_a, self._qty_b, avg_a, self._low_a),
            ('B', ask_b, cap_b, self._qty_b, self._qty_a, avg_b, self._low_b),
        ]:
            if px > cap:
                continue
            balance = q <= qo
            avgdown = q > 0 and px < avg_s - self.p.edge
            if not (balance or avgdown):
                continue
            if not balance and (q - qo) >= self.p.max_imb:
                continue
            edge_val = cap - px
            candidates.append({
                "side": side, "px": px, "edge": edge_val, "cap": cap, "low": low,
                "score": edge_val + (0.05 if balance else 0) + (0.03 if avgdown else 0),
                "balance": balance, "bal": max(0.0, qo - q),
            })

        if not candidates:
            return None
        return max(candidates, key=lambda c: c["score"])
