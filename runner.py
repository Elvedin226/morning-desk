"""The paper-trading loop: decision -> risk gate -> order -> journal.

    .venv/Scripts/python.exe runner.py            # one pass, dry run
    .venv/Scripts/python.exe runner.py --status   # journal summary, no trading

The decision comes from bot.py, unchanged — the same rules that produce the
dashboard produce the orders, so there is one strategy rather than two that can
drift apart.

SAFETY, in the order it matters:
  * No credentials  -> DRY RUN. Nothing is sent anywhere.
  * Credentials     -> PAPER. Fake money at a real broker.
  * Live trading    -> requires ALPACA_LIVE=1 as a separate, deliberate act.
  * Every run is journalled, including the ones that do nothing.
  * Open positions are read from the BROKER at startup, never from memory, so a
    crash mid-position cannot cause a double-up on restart.

In DRY RUN the SIMULATED ACCOUNT in portfolio.py plays the broker's part: it
holds the cash, the open positions and the equity, and it closes positions on
stop, target or time stop each run. That is what makes a two-week dry run
readable - entries that resolve into wins and losses instead of a list of
suggestions that never end.
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import broker as broker_mod
import journal
import portfolio
import risk
import watchlist
import bot


def decide() -> dict | None:
    """Today's candidate, or None. Mirrors bot.run() without the printing."""
    tickers = sorted(set(list(watchlist.SECTORS) + list(watchlist.SECTOR_ETFS) + ["SPY"]))
    close = watchlist._download(tickers, "2023-01-01")
    reg = watchlist.regime(close["SPY"].dropna())
    if not reg["green"]:
        return {"regime_green": False, "candidate": None, "regime": reg, "close": close}

    sectors = watchlist.sector_strength(close)
    leaders = set(sectors.head(4)["etf"])
    rows = []
    for t in watchlist.SECTORS:
        if t not in close.columns:
            continue
        c = bot.evaluate(close, t)
        if c:
            c["leading_sector"] = watchlist.SECTORS[t] in leaders
            rows.append(c)
    passing = sorted([c for c in rows if c["passes"]],
                     key=lambda c: (-c["leading_sector"], -c["momentum"]))

    for c in passing[:8]:
        days = bot.earnings_days_away(c["ticker"])
        c["earnings_days"] = days
        if days is not None and 0 <= days <= bot.HOLD_DAYS:
            continue
        return {"regime_green": True, "candidate": c, "regime": reg, "close": close}

    return {"regime_green": True, "candidate": None, "regime": reg, "close": close}


def run_once(dry_override: bool = False) -> dict:
    bk = broker_mod.Broker()
    print()
    print("=" * 62)
    print(f"  RUNNER  |  mode: {bk.mode}")
    print("=" * 62)

    if bk.live:
        print()
        print("  *** LIVE MODE - THIS PLACES REAL ORDERS WITH REAL MONEY ***")

    # Mark the simulated book to market FIRST, so a position that hit its stop
    # overnight is closed and its cash freed before today's decision is sized
    # against equity. Doing this afterwards would size against money the account
    # no longer has.
    book = portfolio.load()
    book, closed_today = portfolio.update(book)
    portfolio.save(book)

    for c in closed_today:
        print(f"\n  CLOSED {c['ticker']} at ${c['exit']:,.2f} ({c['reason']})  "
              f"P&L ${c['pnl']:+,.2f} ({c['pnl_pct']:+.1f}%)")
        journal.record("close", ticker=c["ticker"], qty=c["qty"], entry=c["entry"],
                       exit=c["exit"], reason=c["reason"], pnl=c["pnl"],
                       pnl_pct=c["pnl_pct"], opened=c["opened"], mode=bk.mode)

    # In DRY RUN the simulated book IS the account. Once credentials exist the
    # broker is authoritative and the book is only a mirror - never the source
    # of truth for what is actually held.
    if bk.dry:
        equity = book["equity"]
        positions = [{"symbol": p["ticker"]} for p in book["positions"]]
    else:
        equity = bk.equity()
        positions = bk.positions()
    market_open = bk.is_open()

    st = portfolio.stats(book)
    print(f"\n  equity ${equity:,.2f}  ({st['total_return']*100:+.2f}% from "
          f"${st['start_equity']:,.0f})   cash ${book['cash']:,.2f}")
    print(f"  open {len(positions)}   closed {st['closed_trades']}   "
          f"realised ${st['realised_pnl']:+,.2f}   "
          f"market {'open' if market_open else 'closed'}")
    for p in book["positions"]:
        print(f"    {p['ticker']:<6} {p['qty']:>8.4f} sh   entry ${p['entry']:,.2f}"
              f"   last ${p.get('last', p['entry']):,.2f}"
              f"   stop ${p['stop']:,.2f}   target ${p['target']:,.2f}"
              f"   P&L ${p.get('unrealised', 0.0):+,.2f}")

    d = decide()
    reg = d["regime"]
    print(f"  regime {'GREEN' if d['regime_green'] else 'RED'}   "
          f"SPY {reg['spy']:,.2f}  10sma {reg['s10']:,.2f}  20sma {reg['s20']:,.2f}")

    if d["candidate"] is None:
        why = "regime red" if not d["regime_green"] else "nothing passed the checklist"
        print(f"\n  >>> NO TRADE - {why}\n")
        return journal.record("no_trade", reason=why, equity=equity,
                              open_positions=len(positions), mode=bk.mode)

    c = d["candidate"]
    print(f"\n  candidate {c['ticker']} @ ${c['price']:,.2f}  "
          f"stop ${c['stop']:,.2f} ({c['risk_pct']*100:.1f}%)")

    # Risk gate before sizing: cheap checks that can veto regardless of the name.
    # The drawdown kill switch measures against the book's own starting equity
    # and yesterday's close, which is what those limits were always meant to
    # mean. Before the book existed there was no such history and both fell back
    # to today's equity, which could never trip either switch.
    start_equity = st["start_equity"]
    curve = book.get("equity_curve", [])
    day_start = curve[-2]["equity"] if len(curve) >= 2 else equity
    gate = risk.gate(equity, start_equity, day_start, len(positions), market_open)
    if not gate.allowed:
        print(f"\n  >>> BLOCKED - {gate.reason}\n")
        return journal.record("blocked", reason=gate.reason, ticker=c["ticker"],
                              equity=equity, mode=bk.mode)

    # Correlation veto: a second position highly correlated with an open one is
    # the same bet at double size, which position COUNT alone cannot detect.
    held = [p.get("symbol") for p in positions if p.get("symbol")]
    corr = risk.correlation_veto(c["ticker"], held, d["close"])
    print(f"  correlation: {corr.reason}")
    if not corr.allowed:
        print(f"\n  >>> BLOCKED - {corr.reason}\n")
        return journal.record("blocked", reason=corr.reason, ticker=c["ticker"],
                              equity=equity, mode=bk.mode)

    sized = risk.size(equity, c["price"], c["stop"], cash=book["cash"])
    if not sized.allowed:
        print(f"\n  >>> BLOCKED - {sized.reason}\n")
        return journal.record("blocked", reason=sized.reason, ticker=c["ticker"],
                              equity=equity, mode=bk.mode)

    print(f"  size {sized.qty:.4f} sh  =  ${sized.value:,.2f}  "
          f"({sized.value/equity*100:.0f}% of equity)   risking ${sized.risk_dollars:,.2f}")

    if dry_override:
        print("\n  >>> --no-order given, stopping before submit\n")
        return journal.record("no_trade", reason="--no-order", ticker=c["ticker"],
                              qty=sized.qty, equity=equity, mode=bk.mode)

    target = c["price"] + bot.TARGET_R * (c["price"] - c["stop"])

    order = bk.submit(c["ticker"], sized.qty, "buy", c["price"])
    slip = order.slippage_pct
    print(f"\n  >>> ORDER {order.status}  {order.symbol} x{order.qty:.4f}")
    if order.filled:
        print(f"      intended ${order.intended:,.2f}  filled ${order.filled:,.2f}  "
              f"slippage {slip*100:+.3f}%")

    # Record the position in the simulated book so future runs can resolve it.
    # The real fill is used when there is one; in DRY RUN there isn't, and the
    # decision price is the honest stand-in.
    pos = portfolio.open_position(book, order.symbol, order.qty,
                                  order.filled or order.intended, c["stop"], target)
    portfolio.save(book)
    print(f"      book: {pos['ticker']} entry ${pos['entry']:,.2f}  "
          f"stop ${pos['stop']:,.2f}  target ${pos['target']:,.2f}  "
          f"cost ${pos['cost_basis']:,.2f}")
    print()

    return journal.record(
        "order", ticker=order.symbol, qty=order.qty, side=order.side,
        intended=order.intended, filled=order.filled, slippage_pct=slip,
        stop=c["stop"], target=target,
        risk_dollars=sized.risk_dollars, equity=equity,
        order_id=order.order_id, status=order.status, mode=bk.mode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Paper-trading loop.")
    ap.add_argument("--status", action="store_true", help="journal summary, no trading")
    ap.add_argument("--no-order", action="store_true",
                    help="run the full decision and sizing but stop before submitting")
    args = ap.parse_args()

    if args.status:
        import subprocess, sys
        subprocess.run([sys.executable, "journal.py"])
    else:
        try:
            run_once(dry_override=args.no_order)
        except Exception as err:
            journal.record("error", error=str(err), kind=type(err).__name__)
            print(f"\n  ERROR: {type(err).__name__}: {err}\n")
            raise
