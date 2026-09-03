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


def open_position(state: dict, ticker: str, qty: float, entry: float,
                  stop: float, target: float) -> dict:
    """Record a simulated buy. Cost is charged on entry."""
    cost = qty * entry * (1 + COST_PER_SIDE)
    pos = {"ticker": ticker, "qty": round(qty, 6), "entry": round(entry, 4),
           "stop": round(stop, 4), "target": round(target, 4),
           "opened": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
           "cost_basis": round(cost, 2)}
    state["positions"].append(pos)
    state["cash"] = round(state["cash"] - cost, 2)
    return pos


def _close(state: dict, pos: dict, price: float, reason: str) -> dict:
    proceeds = pos["qty"] * price * (1 - COST_PER_SIDE)
    pnl = proceeds - pos["cost_basis"]
    rec = {**pos, "exit": round(price, 4), "reason": reason,
           "closed": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
           "proceeds": round(proceeds, 2), "pnl": round(pnl, 2),
           "pnl_pct": round(pnl / pos["cost_basis"] * 100, 2) if pos["cost_basis"] else None}
    state["closed"].append(rec)
    state["positions"] = [p for p in state["positions"] if p is not pos]
    state["cash"] = round(state["cash"] + proceeds, 2)
    return rec


def update(state: dict) -> tuple[dict, list[dict]]:
    """Mark open positions to market and close any that hit stop, target or the
    time stop. Returns (state, list of closes made this run)."""
    if not state["positions"]:
        state["market_value"] = 0.0
        state["equity"] = state["cash"]
        _stamp_equity(state)
        return state, []

    tickers = sorted({p["ticker"] for p in state["positions"]})
    try:
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
        if l <= pos["stop"]:
            fill = min(o, pos["stop"])   # gapped through -> you get the open
            closes.append(_close(state, pos, fill, "stop"))
            continue
        if h >= pos["target"]:
            fill = max(o, pos["target"])
            closes.append(_close(state, pos, fill, "target"))
            continue

        held = (datetime.now(timezone.utc).date()
                - datetime.strptime(pos["opened"], "%Y-%m-%d").date()).days
        if held >= MAX_HOLD_DAYS:
            closes.append(_close(state, pos, c, "time stop"))
            continue

        pos["last"] = round(c, 4)
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
    }


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
