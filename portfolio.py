"""Simulated paper account: positions, exits, realised P&L, equity curve.

The gap this fills: runner.py logged entries and nothing else, so a position
opened on Monday was never resolved. A journal of unresolved entries cannot
answer "did this work" no matter how long it runs.

State lives in data_cache/portfolio.json and is the single source of truth for
what the bot "holds". It is rebuilt from nothing if deleted, so wiping the file
is a clean restart.

HONESTY ABOUT THE SIMULATION - what it does and does not model:

  models      entry at the price the decision was made on, exits at the stop or
              target, gap-through (an exit worse than the level when the day
              opens past it), a per-side cost, and mark-to-market on the rest.
  ignores     intraday path (a bar that hits both stop and target is scored as a
              STOP - the pessimistic assumption, because you cannot know), real
              fill quality, partial fills, and dividends.

The one thing simulation can never give you is real slippage. That needs a
broker. Everything here is "what the rules would have produced against the daily
bars", which is a fair test of the RULES and an optimistic one of EXECUTION.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

STATE = Path(__file__).parent / "data_cache" / "portfolio.json"

START_EQUITY = 421.0
COST_PER_SIDE = 0.0005   # 5bp each way: commission-free broker, spread + slippage
MAX_HOLD_DAYS = 40       # time stop - a position that has done nothing in 8 weeks
FORCED_HOLD_DAYS = 10    # forced trades cycle faster so their slots free up


def _blank() -> dict:
    return {"start_equity": START_EQUITY, "cash": START_EQUITY,
            "positions": [], "closed": [], "equity_curve": [],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def load() -> dict:
    if not STATE.exists():
        return _blank()
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A corrupt state file must not silently reset the account and lose the
        # record. Refuse loudly instead.
        raise RuntimeError(f"{STATE} is unreadable. Inspect it before continuing.")


def save(state: dict) -> None:
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _bars(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Recent daily OHLC per ticker.

    Branch on the COLUMN SHAPE, not on len(tickers): yfinance returns MultiIndex
    columns for a one-element list too, so counting tickers hands back a frame
    whose "Open" key does not exist.
    """
    raw = yf.download(tickers, period="10d", interval="1d",
                      auto_adjust=True, progress=False, group_by="ticker")
    if not isinstance(raw.columns, pd.MultiIndex):
        return {tickers[0]: raw.dropna()}
    lvl0 = set(raw.columns.get_level_values(0))
    return {t: raw[t].dropna() for t in tickers if t in lvl0}


def _intraday_bars(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Today's session so far, as a single synthetic bar per ticker.

    Built from 5-minute bars, so the high and low are the running extremes of the
    session rather than yesterday's finished ones. This is what lets a stop fire
    on the day it is hit instead of the next morning.

    Yahoo's free intraday feed runs about 15 minutes behind. A stop is therefore
    detected up to ~15 min late, and the fill is recorded at the stop level - so
    a fast breakdown is modelled better than it would really fill. Called out
    here because it flatters results and is invisible in the output.
    """
    raw = yf.download(tickers, period="1d", interval="5m",
                      auto_adjust=True, progress=False, group_by="ticker")
    if raw.empty:
        return {}
    out = {}
    for t in tickers:
        df = raw if not isinstance(raw.columns, pd.MultiIndex) else (
            raw[t] if t in set(raw.columns.get_level_values(0)) else None)
        if df is None:
            continue
        df = df.dropna()
        if df.empty:
            continue
        out[t] = pd.DataFrame([{
            "Open": float(df["Open"].iloc[0]), "High": float(df["High"].max()),
            "Low": float(df["Low"].min()), "Close": float(df["Close"].iloc[-1]),
        }])
    return out


def open_position(state: dict, ticker: str, qty: float, entry: float,
                  stop: float, target: float, side: str = "long",
                  forced: bool = False, note: str = "") -> dict:
    """Record a simulated entry.

    A SHORT is modelled as collateral rather than as borrowed stock: the same
    notional is set aside from cash and released at exit, and P&L is the mirror
    of the long case. That skips borrow fees and margin calls, which is a real
    simplification - but it keeps one cash accounting rule for both sides, and
    borrow on the large caps in this universe is pennies.

    `forced` marks a position the checklist did NOT approve, so the record can
    be split later and one arm cannot silently contaminate the other.
    """
    # Guard the direction here as well as in risk.size. An inverted short - stop
    # below entry - looks plausible in JSON and silently resolves on the wrong
    # side of the bar, producing a profit from what should have been a loss.
    if side == "short" and stop <= entry:
        raise ValueError(f"short {ticker}: stop {stop} must be above entry {entry}")
    if side == "long" and stop >= entry:
        raise ValueError(f"long {ticker}: stop {stop} must be below entry {entry}")

    cost = qty * entry * (1 + COST_PER_SIDE)
    pos = {"ticker": ticker, "side": side, "qty": round(qty, 6), "entry": round(entry, 4),
           "stop": round(stop, 4), "target": round(target, 4),
           "opened": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
           "cost_basis": round(cost, 2), "forced": forced, "note": note}
    state["positions"].append(pos)
    state["cash"] = round(state["cash"] - cost, 2)
    return pos


def _close(state: dict, pos: dict, price: float, reason: str) -> dict:
    side = pos.get("side", "long")
    gross = pos["qty"] * price
    fee = gross * COST_PER_SIDE
    if side == "short":
        # Collateral comes back, plus what the price fell (or minus what it rose).
        pnl = pos["qty"] * (pos["entry"] - price) - fee
        returned = pos["cost_basis"] + pnl
    else:
        returned = gross - fee
        pnl = returned - pos["cost_basis"]
    rec = {**pos, "exit": round(price, 4), "reason": reason,
           "closed": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
           "proceeds": round(returned, 2), "pnl": round(pnl, 2),
           "pnl_pct": round(pnl / pos["cost_basis"] * 100, 2) if pos["cost_basis"] else None}
    state["closed"].append(rec)
    state["positions"] = [p for p in state["positions"] if p is not pos]
    state["cash"] = round(state["cash"] + returned, 2)
    return rec


def update(state: dict, intraday: bool = False) -> tuple[dict, list[dict]]:
    """Mark open positions to market and close any that hit stop, target or the
    time stop. Returns (state, list of closes made this run).

    intraday=True scores against the CURRENT session's running high/low instead
    of the last completed daily bar, so exits land on the day they happen.
    """
    if not state["positions"]:
        state["market_value"] = 0.0
        state["equity"] = state["cash"]
        _stamp_equity(state)
        return state, []

    tickers = sorted({p["ticker"] for p in state["positions"]})
    try:
        bars = _intraday_bars(tickers) if intraday else _bars(tickers)
        if intraday and not bars:
            # Before the open, or a feed hiccup. Fall back to daily rather than
            # marking the book to nothing.
            bars = _bars(tickers)
    except Exception:
        # Data failure must not silently mark the book to stale prices.
        state["equity"] = state["cash"] + sum(p["cost_basis"] for p in state["positions"])
        state["data_error"] = True
        return state, []
    state.pop("data_error", None)

    closes, market_value = [], 0.0
    for pos in list(state["positions"]):
        df = bars.get(pos["ticker"])
        if df is None or df.empty:
            market_value += pos["cost_basis"]
            continue

        day = df.iloc[-1]
        o, h, l, c = (float(day["Open"]), float(day["High"]),
                      float(day["Low"]), float(day["Close"]))

        # Stop checked before target: a bar that touches both is scored as a
        # loss, because the intraday order is unknowable from daily data and
        # assuming the good one would flatter every result.
        #
        # A short mirrors this: its stop sits ABOVE entry and its target BELOW,
        # so the comparisons and the gap-through side both invert.
        short = pos.get("side") == "short"
        if short:
            hit_stop, stop_fill = h >= pos["stop"], max(o, pos["stop"])
            hit_tgt, tgt_fill = l <= pos["target"], min(o, pos["target"])
        else:
            hit_stop, stop_fill = l <= pos["stop"], min(o, pos["stop"])
            hit_tgt, tgt_fill = h >= pos["target"], max(o, pos["target"])

        if hit_stop:
            closes.append(_close(state, pos, stop_fill, "stop"))
            continue
        if hit_tgt:
            closes.append(_close(state, pos, tgt_fill, "target"))
            continue

        held = (datetime.now(timezone.utc).date()
                - datetime.strptime(pos["opened"], "%Y-%m-%d").date()).days
        # Forced positions time out sooner so their slots recycle during a short
        # test; leaving them on the 40-day clock would jam the budget in a week.
        limit = FORCED_HOLD_DAYS if pos.get("forced") else MAX_HOLD_DAYS
        if held >= limit:
            closes.append(_close(state, pos, c, "time stop"))
            continue

        pos["last"] = round(c, 4)
        if short:
            # Collateral plus P&L, so a winning short raises equity and a losing
            # one lowers it, exactly as the long case does.
            pos["unrealised"] = round(pos["qty"] * (pos["entry"] - c), 2)
            market_value += pos["cost_basis"] + pos["unrealised"]
        else:
            pos["unrealised"] = round(pos["qty"] * c - pos["cost_basis"], 2)
            market_value += pos["qty"] * c

    state["market_value"] = round(market_value, 2)
    state["equity"] = round(state["cash"] + market_value, 2)
    _stamp_equity(state)
    return state, closes


def _stamp_equity(state: dict) -> None:
    """One equity point per calendar day, overwritten on re-runs so the curve
    does not gain three points because the runner was triggered three times."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    curve = [p for p in state.get("equity_curve", []) if p["date"] != today]
    curve.append({"date": today, "equity": state["equity"]})
    state["equity_curve"] = curve[-260:]


def day_realised(state: dict) -> float:
    """Realised P&L booked today. Drives the daily profit target.

    Realised only - an open position showing a gain has not banked anything, and
    counting it would stop the bot on profit it does not yet have.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return round(sum(c["pnl"] for c in state.get("closed", [])
                     if c.get("closed") == today), 2)


def stats(state: dict) -> dict:
    closed = state.get("closed", [])
    wins = [c for c in closed if c["pnl"] > 0]
    equity = state.get("equity", state["cash"])
    start = state.get("start_equity", START_EQUITY)
    return {
        "equity": equity,
        "start_equity": start,
        "total_return": equity / start - 1 if start else 0.0,
        "cash": state["cash"],
        "open_positions": len(state.get("positions", [])),
        "closed_trades": len(closed),
        "wins": len(wins),
        "win_rate": len(wins) / len(closed) if closed else None,
        "realised_pnl": round(sum(c["pnl"] for c in closed), 2),
        "unrealised_pnl": round(sum(p.get("unrealised", 0.0)
                                    for p in state.get("positions", [])), 2),
        "best": max((c["pnl"] for c in closed), default=None),
        "worst": min((c["pnl"] for c in closed), default=None),
        "arms": arms(state),
        "day_realised": day_realised(state),
    }


def arms(state: dict) -> dict:
    """Qualified trades and forced trades, scored separately.

    Pooling them would destroy the only reason forcing trades is defensible: the
    forced arm exists to be COMPARED against the rules, and a blended win rate
    answers neither question.
    """
    out = {}
    for label, want in (("qualified", False), ("forced", True)):
        rows = [c for c in state.get("closed", []) if bool(c.get("forced")) is want]
        wins = [c for c in rows if c["pnl"] > 0]
        longs = [c for c in rows if c.get("side", "long") == "long"]
        out[label] = {
            "n": len(rows),
            "pnl": round(sum(c["pnl"] for c in rows), 2),
            "win_rate": len(wins) / len(rows) if rows else None,
            "avg_pct": round(sum(c["pnl_pct"] for c in rows) / len(rows), 2) if rows else None,
            "longs": len(longs), "shorts": len(rows) - len(longs),
            "open": sum(1 for p in state.get("positions", [])
                        if bool(p.get("forced")) is want),
        }
    return out


if __name__ == "__main__":
    st = load()
    st, closes = update(st)
    save(st)
    s = stats(st)

    print()
    print("  PAPER ACCOUNT")
    print(f"    equity        ${s['equity']:,.2f}   ({s['total_return']*100:+.2f}% from ${s['start_equity']:,.0f})")
    print(f"    cash          ${s['cash']:,.2f}")
    print(f"    open          {s['open_positions']}   unrealised ${s['unrealised_pnl']:+,.2f}")
    print(f"    closed        {s['closed_trades']}   realised   ${s['realised_pnl']:+,.2f}")
    if s["win_rate"] is not None:
        print(f"    win rate      {s['win_rate']*100:.0f}%  ({s['wins']}/{s['closed_trades']})")
        print(f"    best / worst  ${s['best']:+,.2f} / ${s['worst']:+,.2f}")
    for c in closes:
        print(f"    CLOSED {c['ticker']} at ${c['exit']:.2f} ({c['reason']}) "
              f"P&L ${c['pnl']:+.2f} ({c['pnl_pct']:+.1f}%)")
    print()
    print(f"  {STATE}")
