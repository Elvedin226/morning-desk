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
    return {"trades": [], "flat_days": [],
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
        note: str = "") -> dict:
    st = load()
    row = {"sym": sym, "pnl": round(float(pnl), 2),
           "pct": None if pct is None else round(float(pct), 2),
           "date": date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
           "note": note,
           "logged": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    st["trades"].append(row)
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
    days = sorted({t["date"] for t in tr} | set(st.get("flat_days", [])))
    by_day = {}
    for t in tr:
        by_day[t["date"]] = by_day.get(t["date"], 0.0) + t["pnl"]
    for d in st.get("flat_days", []):
        by_day.setdefault(d, 0.0)

    ch = [t for t in tr if CHALLENGE_START <= t["date"] <= CHALLENGE_END]
    ch_days = sorted({t["date"] for t in ch}
                     | {d for d in st.get("flat_days", [])
                        if CHALLENGE_START <= d <= CHALLENGE_END})
    return {
        "trades": len(tr),
        "pnl": round(sum(t["pnl"] for t in tr), 2),
        "wins": len(wins),
        "win_rate": len(wins) / len(tr) if tr else None,
        "best": max((t["pnl"] for t in tr), default=None),
        "worst": min((t["pnl"] for t in tr), default=None),
        "sessions": len(days),
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
    a.add_argument("--note", default="")
    f = sub.add_parser("flat")
    f.add_argument("--date")
    args = ap.parse_args()

    if args.cmd == "add":
        r = add(args.sym, args.pnl, args.pct, args.date, args.note)
        print(f"  logged {r['sym']}  ${r['pnl']:+.2f}  {r['date']}")
    elif args.cmd == "flat":
        flat(args.date)
        print("  no-trade day recorded")

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
    if s["by_day"]:
        print()
        for d, v in s["by_day"].items():
            print(f"    {d}   ${v:+8.2f}")
