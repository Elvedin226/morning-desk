"""Swing trading bot — mechanical decisions, no judgment calls.

    .venv/Scripts/python.exe bot.py
    .venv/Scripts/python.exe bot.py --account 421 --risk 0.01

Emits ONE decision: a named candidate with entry, stop, share count and target,
or an explicit NO TRADE with the reason. The rules below are fixed in advance and
applied identically every run, so the same inputs always give the same answer.

WHAT THIS IS: consistency and risk control. Every rule here either survived
testing or exists to bound losses.
WHAT THIS IS NOT: an edge. Mechanising rules makes them repeatable, not
profitable. Of eight strategies tested in this project, one survived, and it
died on transaction costs. Nothing here beats that record.

Rules kept (tested or structural):
  regime filter    only trade when SPY's averages are stacked and rising
  momentum         12-1 momentum, the one factor with real literature behind it
  trend structure  price above SMA20 > SMA50; no broken charts
  not extended     must be near the recent high, not vertical off it
  earnings block   no entry if earnings falls inside the hold window
  1% risk sizing   position size derived from the stop, never chosen

Rules deliberately EXCLUDED (tested and failed — see ablate.py):
  breakout entry   -0.76% edge, t = -2.97. Buying the break subtracts value.
  the "coil"       cut sample 10x, added nothing measurable
  ADR / vol taper  same
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import catalysts
import watchlist

LOG = Path(__file__).parent / "data_cache" / "bot_log.jsonl"

# --- Rules. Change these deliberately, not to make today's output nicer. ------
HOLD_DAYS = 21           # expected hold; earnings inside this window blocks entry
MAX_FROM_HIGH = -0.15    # must be within 15% of the 52-week high
MIN_MOMENTUM = 0.10      # 12-1 momentum floor
MAX_EXTENDED = 0.10      # more than 10% above the 20-day SMA = chasing
MAX_RISK_PCT = 0.08     # a stop wider than this cannot be sized at a small account
STOP_BUFFER = 0.02       # place the stop this far below the consolidation low
TARGET_R = 2.0           # first trim at 2R


def evaluate(close: pd.DataFrame, ticker: str) -> dict | None:
    """Apply the checklist to one name. Returns None if data is insufficient."""
    s = close[ticker].dropna()
    if len(s) < 252:
        return None

    price = float(s.iloc[-1])
    sma10, sma20, sma50 = (float(s.rolling(n).mean().iloc[-1]) for n in (10, 20, 50))
    high_52w = float(s.rolling(252).max().iloc[-1])
    # The consolidation low over the recent base — this is what defines the stop,
    # and therefore the position size. Everything downstream follows from it.
    base_low = float(s.iloc[-20:].min())

    stop = base_low * (1 - STOP_BUFFER)
    risk_pct = (price - stop) / price if price > stop else np.nan

    checks = {
        "above_sma20": price > sma20,
        "sma20_above_sma50": sma20 > sma50,
        "near_high": price / high_52w - 1 >= MAX_FROM_HIGH,
        "momentum": float(s.iloc[-22] / s.iloc[-252] - 1) >= MIN_MOMENTUM,
        "not_extended": price / sma20 - 1 <= MAX_EXTENDED,
        "stop_usable": np.isfinite(risk_pct) and 0 < risk_pct <= MAX_RISK_PCT,
    }
    return {
        "ticker": ticker,
        "price": price,
        "sma10": sma10, "sma20": sma20, "sma50": sma50,
        "from_high": price / high_52w - 1,
        "momentum": float(s.iloc[-22] / s.iloc[-252] - 1),
        "vs_sma20": price / sma20 - 1,
        "stop": stop,
        "risk_pct": risk_pct,
        "checks": checks,
        "passes": all(checks.values()),
    }


def size_position(c: dict, account: float, risk_budget: float,
                  fractional: bool = True) -> dict:
    """Position size falls out of the stop. It is never a preference.

    Whole shares break down at small accounts: risking $4.21 on a $267 stock with
    a $18.70 stop rounds to zero shares, so the rule cannot be expressed at all.
    Fractional shares (Robinhood, Schwab, Fidelity all support them) remove that
    floor and let the 1% rule hold at any account size.
    """
    risk_dollars = account * risk_budget
    risk_per_share = c["price"] - c["stop"]
    if risk_per_share <= 0:
        shares = 0.0
    elif fractional:
        shares = risk_dollars / risk_per_share
    else:
        shares = float(int(risk_dollars // risk_per_share))
    return {
        "risk_dollars": risk_dollars,
        "risk_per_share": risk_per_share,
        "shares": shares,
        "fractional": fractional,
        "position_value": shares * c["price"],
        "target": c["price"] + TARGET_R * risk_per_share,
    }


def earnings_days_away(ticker: str) -> int | None:
    nxt = catalysts.next_earnings(ticker)
    if nxt is None:
        return None
    return (nxt - pd.Timestamp.utcnow().tz_localize(None).normalize()).days


def run(account: float, risk_budget: float, shortlist: int, force: bool = False,
        fractional: bool = True) -> dict:
    tickers = sorted(set(list(watchlist.SECTORS) + list(watchlist.SECTOR_ETFS) + ["SPY"]))
    close = watchlist._download(tickers, "2023-01-01")

    reg = watchlist.regime(close["SPY"].dropna())
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    print()
    print("=" * 66)
    print(f"  SWING BOT   {stamp}   account ${account:,.0f}   risk {risk_budget*100:.0f}%")
    print("=" * 66)

    print()
    print(f"  REGIME: {'GREEN' if reg['green'] else 'RED'}")
    print(f"    SPY {reg['spy']:.2f}  |  10sma {reg['s10']:.2f}  20sma {reg['s20']:.2f}  50sma {reg['s50']:.2f}")
    print(f"    stacked 10>20>50: {reg['stacked']}     10sma rising: {reg['rising']}")

    if not reg["green"] and force:
        print()
        print("  !!! PREVIEW MODE - regime is RED and you overrode it with --force.")
        print("      Shown so you can see the output format. Not a live signal.")

    if not reg["green"] and not force:
        print()
        print("  >>> NO TRADE - regime filter is RED.")
        print("      The strategy is long-only and only tested in bullish conditions.")
        print("      Re-run when SPY's 10-day is above its 20-day and rising.")
        print()
        return {"date": stamp, "decision": "NO_TRADE", "reason": "regime_red", "regime": reg}

    # Rank everything, then apply the checklist.
    sectors = watchlist.sector_strength(close)
    leaders = set(sectors.head(4)["etf"])
    rows = []
    for t in watchlist.SECTORS:
        if t not in close.columns:
            continue
        c = evaluate(close, t)
        if c:
            c["leading_sector"] = watchlist.SECTORS[t] in leaders
            rows.append(c)

    passing = sorted([c for c in rows if c["passes"]],
                     key=lambda c: (-c["leading_sector"], -c["momentum"]))

    if not passing:
        print()
        print(f"  >>> NO TRADE - regime is green but 0 of {len(rows)} names pass the checklist.")
        print()
        return {"date": stamp, "decision": "NO_TRADE", "reason": "no_candidates", "regime": reg}

    # Earnings check only on the shortlist — it costs an API call per name.
    chosen, blocked = None, []
    for c in passing[:shortlist]:
        days = earnings_days_away(c["ticker"])
        c["earnings_days"] = days
        if days is not None and 0 <= days <= HOLD_DAYS:
            blocked.append((c["ticker"], days))
            continue
        chosen = c
        break

    if chosen is None:
        print()
        print("  >>> NO TRADE - every candidate has earnings inside the hold window.")
        for t, d in blocked:
            print(f"      {t}: earnings in {d} days")
        print()
        return {"date": stamp, "decision": "NO_TRADE", "reason": "earnings_blocked",
                "blocked": blocked, "regime": reg}

    pos = size_position(chosen, account, risk_budget, fractional)

    print()
    print(f"  >>> CANDIDATE: {chosen['ticker']} @ ${chosen['price']:,.2f}")
    print()
    print(f"    entry        ${chosen['price']:,.2f}   (at or below; do not chase)")
    print(f"    stop         ${chosen['stop']:,.2f}   ({chosen['risk_pct']*100:.1f}% below entry)")
    print(f"    target       ${pos['target']:,.2f}   (first trim at {TARGET_R:.0f}R)")
    print()
    unit = "shares" if not pos["fractional"] else "shares (fractional)"
    print(f"    buy          {pos['shares']:.4f} {unit}")
    print(f"    position     ${pos['position_value']:,.2f}   ({pos['position_value']/account*100:.0f}% of account)")
    print(f"    at risk      ${pos['risk_dollars']:,.2f}   ({risk_budget*100:.0f}% of account)")
    print()
    print(f"    momentum     {chosen['momentum']*100:+.1f}%      from 52w high  {chosen['from_high']*100:.1f}%")
    print(f"    vs 20sma     {chosen['vs_sma20']*100:+.1f}%      earnings       "
          f"{'in ' + str(chosen['earnings_days']) + ' days' if chosen['earnings_days'] is not None else 'unknown'}")

    if pos["shares"] <= 0 or pos["position_value"] < 1:
        print()
        print("    !!! POSITION TOO SMALL to place. Widen risk or pick a cheaper name.")
    elif not pos["fractional"] and pos["shares"] < 1:
        print()
        print("    !!! 0 WHOLE SHARES - use fractional, or this rule cannot be expressed.")

    if blocked:
        print()
        print("    skipped for earnings: " + ", ".join(f"{t} ({d}d)" for t, d in blocked))

    others = [c["ticker"] for c in passing[1:6] if c["ticker"] != chosen["ticker"]]
    if others:
        print(f"    runners-up: {', '.join(others)}")

    print()
    print("  This is the output of fixed rules, not a recommendation. Nothing here")
    print("  demonstrates an edge - it enforces consistency and caps the downside.")
    print()

    return {
        "date": stamp, "decision": "CANDIDATE", "ticker": chosen["ticker"],
        "price": chosen["price"], "stop": chosen["stop"], "target": pos["target"],
        "shares": pos["shares"], "position_value": pos["position_value"],
        "risk_pct": chosen["risk_pct"], "account": account, "regime": reg,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mechanical swing-trade decision.")
    ap.add_argument("--account", type=float, default=421.0)
    ap.add_argument("--risk", type=float, default=0.01, help="fraction risked per trade")
    ap.add_argument("--shortlist", type=int, default=8, help="how many to earnings-check")
    ap.add_argument("--whole-shares", action="store_true",
                    help="disable fractional sizing (most brokers support fractional)")
    ap.add_argument("--force", action="store_true",
                    help="PREVIEW ONLY: ignore a RED regime to see the output format")
    args = ap.parse_args()

    result = run(args.account, args.risk, args.shortlist, args.force,
                 fractional=not args.whole_shares)
    if args.force:
        result["forced"] = True

    # Append-only record. In three months this is what lets you check whether the
    # rules did anything, instead of relying on memory.
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, default=str) + "\n")
    print(f"  logged to {LOG}")
