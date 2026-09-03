"""Risk limits. The layer that decides whether an order is allowed at all.

Everything here exists to bound losses, not to find them. Of eight strategies
tested in this project, one survived and it died on transaction costs — so the
honest assumption while paper trading is that the edge is zero and the only
thing under real control is how much a bad run can take.

The ruin simulation in lottery.py is the reason these numbers are what they are:
a bet with a +50% edge per trade still ruins you 100% of the time at full size
and 90% of the time at quarter size. Sizing decides survival more than selection
does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

RISK_PER_TRADE = 0.01     # fraction of equity risked per position
MAX_POSITIONS = 3         # concurrent open positions
MAX_DAILY_LOSS = 0.03     # halt for the day past this drawdown
DAILY_PROFIT_TARGET = 20.0  # bank the day once realised P&L reaches this
MAX_TOTAL_DRAWDOWN = 0.15 # halt entirely past this, pending manual restart
MIN_ORDER_VALUE = 1.0     # brokers reject dust
MAX_CORRELATION = 0.70    # reject a candidate this correlated with something held
CORR_WINDOW = 90          # trading days of daily returns used for the estimate


@dataclass
class Decision:
    allowed: bool
    reason: str
    qty: float = 0.0
    value: float = 0.0
    risk_dollars: float = 0.0


def size(equity: float, entry: float, stop: float,
         risk_pct: float = RISK_PER_TRADE, fractional: bool = True,
         side: str = "long") -> Decision:
    """Position size falls out of the stop. It is never a preference.

    Whole shares break down at small accounts — risking $4.21 on a $267 stock
    with an $18.70 stop rounds to zero — so fractional is the default. Alpaca,
    Robinhood, Schwab and Fidelity all support it.
    """
    if entry <= 0 or stop <= 0:
        return Decision(False, "bad prices")
    # A short inverts only the direction check. Risk per share is the distance
    # to the stop either way, so the sizing maths below is unchanged.
    if side == "short":
        if stop <= entry:
            return Decision(False, "short stop is not above entry")
    elif stop >= entry:
        return Decision(False, "stop is not below entry")

    risk_dollars = equity * risk_pct
    per_share = abs(entry - stop)
    qty = risk_dollars / per_share
    if not fractional:
        qty = float(int(qty))

    value = qty * entry
    if qty <= 0 or value < MIN_ORDER_VALUE:
        return Decision(False, f"position too small (${value:.2f})")
    # A position worth more than the account means the stop is so tight that 1%
    # risk implies leverage. Refuse rather than silently truncate.
    if value > equity:
        return Decision(False, f"position ${value:.0f} exceeds equity ${equity:.0f}")

    return Decision(True, "ok", qty=round(qty, 4), value=value, risk_dollars=risk_dollars)


def gate(equity: float, start_equity: float, day_start_equity: float,
         open_positions: int, market_open: bool,
         max_positions: int = MAX_POSITIONS,
         day_pnl: float | None = None) -> Decision:
    """Pre-trade checks that have nothing to do with the candidate itself.

    Ordered cheapest-first, and each returns a reason string that lands in the
    journal — so a quiet week is explainable after the fact rather than a
    mystery.
    """
    if not market_open:
        return Decision(False, "market closed")

    if open_positions >= max_positions:
        return Decision(False, f"already holding {open_positions} positions (max {max_positions})")

    # Profit target: once the day has banked its number, stop opening. Checked
    # on REALISED P&L only - an open position showing +$20 has not made $20, and
    # treating it as though it had is how a good day becomes a flat one.
    # Existing positions keep running; this blocks new entries, not exits.
    if day_pnl is not None and day_pnl >= DAILY_PROFIT_TARGET:
        return Decision(False, f"day's target hit (${day_pnl:+,.2f}) - done trading today")

    day_loss = 1 - equity / day_start_equity if day_start_equity > 0 else 0
    if day_loss >= MAX_DAILY_LOSS:
        return Decision(False, f"daily loss {day_loss*100:.1f}% hit the {MAX_DAILY_LOSS*100:.0f}% limit")

    total_dd = 1 - equity / start_equity if start_equity > 0 else 0
    if total_dd >= MAX_TOTAL_DRAWDOWN:
        return Decision(False,
                        f"total drawdown {total_dd*100:.1f}% hit the {MAX_TOTAL_DRAWDOWN*100:.0f}% "
                        f"kill switch - restart manually after reviewing the journal")

    return Decision(True, "ok")


def correlation_veto(candidate: str, held: list[str], close, window: int = CORR_WINDOW,
                     limit: float = MAX_CORRELATION) -> Decision:
    """Reject a candidate that is really the position you already hold.

    Three positions in JNJ, BMY and ABBV is not diversification - it is one
    pharma bet at triple size, and the 1%-per-trade rule silently becomes 3% on
    a single factor. Position COUNT does not catch this; correlation does.

    Uses daily-return correlation over `window` sessions. Deliberately a blunt
    instrument: it will not catch a subtle factor exposure, but it reliably
    catches "these are the same trade".
    """
    if not held:
        return Decision(True, "no open positions")
    if candidate not in close.columns:
        return Decision(True, "no price history for candidate - cannot check")

    rets = close.pct_change().tail(window)
    worst_name, worst = None, 0.0
    for h in held:
        if h not in close.columns:
            continue
        pair = rets[[candidate, h]].dropna()
        if len(pair) < 30:
            continue
        c = float(pair[candidate].corr(pair[h]))
        if c == c and abs(c) > abs(worst):
            worst_name, worst = h, c

    if worst_name is None:
        return Decision(True, "no overlapping history to compare")
    if abs(worst) >= limit:
        return Decision(False,
                        f"{candidate} correlates {worst:.2f} with open position {worst_name} "
                        f"(limit {limit:.2f}) - same trade, not a second one")
    return Decision(True, f"max correlation {worst:.2f} vs {worst_name}")
