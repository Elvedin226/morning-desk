"""The owner's REAL trades, logged beside the bot's simulated ones.

Point of this file: the bot's record and the owner's record answer different
questions, and pooling them would destroy both. The bot is testing whether a
fixed rule set has an edge. The owner is testing whether discretionary direction
calls have one. Same dashboard, same period, separate books.

Why it earns its place: the intraday 0DTE study found that taking a call AND a
put every session returned +9.3% over a week, while picking the correct side
each day returned +97.2%. Essentially all the value in 0DTE is direction
selection - which a backtest cannot measure, because there is no rule to
backtest. The only way to find out whether someone can do it is to record the
calls in advance and count.

    log a trade   python realbook.py add --sym "SPXW 7695P" --pnl 35.00 --pct 13.21
    a no-trade day  python realbook.py flat
    see the record  python realbook.py

HONESTY NOTE. These are self-reported. Nothing here verifies against a broker
statement, so this book is exactly as reliable as the entries put into it -
including the ones that lost. A record with only winners in it is not a record.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

STATE = Path(__file__).parent / "data_cache" / "real_trades.json"

# The stated challenge: $100 across the 4 sessions of the week of 7 Sep 2026.
CHALLENGE_START = "2026-09-07"
CHALLENGE_END = "2026-09-11"
CHALLENGE_TARGET = 100.0
CHALLENGE_SESSIONS = 4


def _blank() -> dict:
    return {"trades": [], "flat_days": [], "open": [],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def load() -> dict:
    if not STATE.exists():
        return _blank()
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise RuntimeError(f"{STATE} is unreadable. Inspect it before continuing.")


def save(st: dict) -> None:
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2, default=str), encoding="utf-8")


def add(sym: str, pnl: float, pct: float | None = None, date: str | None = None,
        note: str = "", opened: str | None = None) -> dict:
    st = load()
    row = {"sym": sym, "pnl": round(float(pnl), 2),
           "pct": None if pct is None else round(float(pct), 2),
           "date": date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
           "note": note, "opened": opened,
           "logged": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    st["trades"].append(row)
    save(st)
    return row


def add_open(sym: str, cost: float, mark: float, opened: str, note: str = "") -> dict:
    """An OPEN position. Kept out of realised P&L on purpose - an unrealised
    gain is not a result, and folding one into a realised book would flatter
    the record by whatever the position happens to be worth today."""
    st = load()
    row = {"sym": sym, "cost": float(cost), "mark": float(mark),
           "opened": opened, "note": note,
           "unrealised": round((float(mark) - float(cost)) * 100, 2),
           "pct": round((float(mark) / float(cost) - 1) * 100, 2)}
    st.setdefault("open", []).append(row)
    save(st)
    return row


def call(sym: str, side: str, price: float, date: str, taken: bool,
         outcome_pct: float | None = None, note: str = "") -> dict:
    """A stated directional READ, whether or not it was traded.

    Executed trades measure read + execution together. A read that was NOT
    taken measures the read alone - no fill, no stop, no sizing - which makes
    passed calls the cleanest evidence of directional skill in the whole book.
    They are also the ones that vanish if nobody writes them down, because a
    correct read you sat out never shows on a P&L screen.

    `price` is where it stood when the call was made; `outcome_pct` is what the
    call would have returned, filled in later. Log the call BEFORE the outcome
    is known wherever possible - that is what makes it pre-registered.
    """
    st = load()
    row = {"sym": sym, "side": side, "price": float(price), "date": date,
           "taken": bool(taken), "outcome_pct": outcome_pct, "note": note,
           "logged": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    st.setdefault("calls", []).append(row)
    save(st)
    return row


def flat(date: str | None = None) -> None:
    """Record a session with no trades. Without this, a quiet day is
    indistinguishable from a day that was never logged, and the per-session
    average silently only counts days something happened."""
    st = load()
    d = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if d not in st["flat_days"]:
        st["flat_days"].append(d)
        save(st)


def stats(st: dict | None = None) -> dict:
    st = st or load()
    tr = st.get("trades", [])
    wins = [t for t in tr if t["pnl"] > 0]
    # A session is "used" if a position was OPENED, CLOSED, or held through it -
    # not only the day it closed. Without this a two-day hold reads as one day
    # of activity and one of nothing, which is how Sep 8 2026 got mislabelled
    # flat while a CHPT short was open across it.
    days = set(st.get("flat_days", []))
    for t in tr:
        days.add(t["date"])
        if t.get("opened"):
            days.add(t["opened"])
    days = sorted(days)
    by_day = {}
    for t in tr:
        by_day[t["date"]] = by_day.get(t["date"], 0.0) + t["pnl"]
    for d in st.get("flat_days", []):
        by_day.setdefault(d, 0.0)

    op = st.get("open", [])
    ch = [t for t in tr if CHALLENGE_START <= t["date"] <= CHALLENGE_END]
    ch_days = {d for d in st.get("flat_days", []) if CHALLENGE_START <= d <= CHALLENGE_END}
    for t in ch:
        ch_days.add(t["date"])
        if t.get("opened") and CHALLENGE_START <= t["opened"] <= CHALLENGE_END:
            ch_days.add(t["opened"])
    ch_days = sorted(ch_days)
    return {
        "trades": len(tr),
        "pnl": round(sum(t["pnl"] for t in tr), 2),
        "wins": len(wins),
        "win_rate": len(wins) / len(tr) if tr else None,
        "best": max((t["pnl"] for t in tr), default=None),
        "worst": min((t["pnl"] for t in tr), default=None),
        "sessions": len(days),
        "calls": st.get("calls", []),
        "calls_right": sum(1 for c in st.get("calls", [])
                           if c.get("outcome_pct") is not None and c["outcome_pct"] > 0),
        "calls_scored": sum(1 for c in st.get("calls", []) if c.get("outcome_pct") is not None),
        "open_positions": len(op),
        "unrealised": round(sum(o["unrealised"] for o in op), 2),
        "open_list": op,
        "per_session": round(sum(t["pnl"] for t in tr) / len(days), 2) if days else None,
        "by_day": dict(sorted(by_day.items())),
        "challenge": {
            "target": CHALLENGE_TARGET,
            "window": f"{CHALLENGE_START} to {CHALLENGE_END}",
            "sessions_used": len(ch_days),
            "sessions_total": CHALLENGE_SESSIONS,
            "pnl": round(sum(t["pnl"] for t in ch), 2),
            "trades": len(ch),
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Log real trades beside the bot's book.")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("add")
    a.add_argument("--sym", required=True)
    a.add_argument("--pnl", type=float, required=True)
    a.add_argument("--pct", type=float)
    a.add_argument("--date")
    a.add_argument("--opened", help="entry date for a multi-session hold")
    a.add_argument("--note", default="")
    f = sub.add_parser("flat")
    f.add_argument("--date")
    c = sub.add_parser("call")
    c.add_argument("--sym", required=True); c.add_argument("--side", required=True, choices=["long", "short"])
    c.add_argument("--price", type=float, required=True); c.add_argument("--date", required=True)
    c.add_argument("--taken", action="store_true"); c.add_argument("--outcome", type=float)
    c.add_argument("--note", default="")
    args = ap.parse_args()

    if args.cmd == "add":
        r = add(args.sym, args.pnl, args.pct, args.date, args.note, args.opened)
        print(f"  logged {r['sym']}  ${r['pnl']:+.2f}  {r['date']}")
    elif args.cmd == "flat":
        flat(args.date)
        print("  no-trade day recorded")
    elif args.cmd == "call":
        r = call(args.sym, args.side, args.price, args.date, args.taken, args.outcome, args.note)
        print(f"  call logged: {r['side']} {r['sym']} @ {r['price']:.2f} on {r['date']}"
              f"  taken={r['taken']}  outcome={r['outcome_pct']}")

    s = stats()
    c = s["challenge"]
    print()
    print("  REAL TRADES")
    print(f"    trades        {s['trades']}   sessions {s['sessions']}")
    print(f"    net P&L       ${s['pnl']:+,.2f}")
    if s["win_rate"] is not None:
        print(f"    win rate      {s['win_rate']*100:.0f}%  ({s['wins']}/{s['trades']})")
        print(f"    best / worst  ${s['best']:+.2f} / ${s['worst']:+.2f}")
        print(f"    per session   ${s['per_session']:+.2f}")
    print()
    print(f"  CHALLENGE: ${c['target']:.0f} across {c['sessions_total']} sessions, {c['window']}")
    print(f"    booked        ${c['pnl']:+,.2f}   ({c['pnl']/c['target']*100:.0f}% of target)")
    print(f"    sessions used {c['sessions_used']}/{c['sessions_total']}")
    if s["calls_scored"]:
        print(f"    reads scored  {s['calls_right']}/{s['calls_scored']} right")
        for c in s["calls"]:
            print(f"      {c['date']}  {c['side']:<5} {c['sym']:<6} @ {c['price']:.2f}"
                  f"  {'taken' if c['taken'] else 'PASSED'}"
                  f"  {'' if c['outcome_pct'] is None else '%+.1f%%' % c['outcome_pct']}")
    if s["by_day"]:
        print()
        for d, v in s["by_day"].items():
            print(f"    {d}   ${v:+8.2f}")
