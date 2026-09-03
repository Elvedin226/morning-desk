"""Append-only trade journal.

Every run writes a line whether or not it traded, because "the bot did nothing
for three weeks" is itself a finding and needs to be visible rather than
inferred from an absence of rows.

The column that justifies the whole file is `slippage_pct` — intended price at
decision time against actual fill. A backtest cannot produce it, and it is the
number that killed vol-compression (edge lived inside the spread) and nearly
killed ORB (break-even at 2.2 cents/share). After enough fills it says whether
this account's real execution can support the strategy at all.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

LOG = Path(__file__).parent / "data_cache" / "trades.jsonl"


def record(event: str, **fields) -> dict:
    """Write one line. `event` is one of: no_trade, blocked, order, error."""
    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "event": event, **fields}
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def load() -> list[dict]:
    if not LOG.exists():
        return []
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a truncated final line should not lose the history
    return rows


def summary() -> dict:
    """What the journal says so far. Deliberately plain — no equity curve, no
    win rate until there are enough closed trades for either to mean anything."""
    rows = load()
    orders = [r for r in rows if r.get("event") == "order"]
    slips = [r["slippage_pct"] for r in orders
             if isinstance(r.get("slippage_pct"), (int, float))]
    return {
        "runs": len(rows),
        "orders": len(orders),
        "no_trade": sum(1 for r in rows if r.get("event") == "no_trade"),
        "blocked": sum(1 for r in rows if r.get("event") == "blocked"),
        "errors": sum(1 for r in rows if r.get("event") == "error"),
        "fills_with_slippage": len(slips),
        "median_slippage_pct": (sorted(slips)[len(slips) // 2] * 100) if slips else None,
    }


if __name__ == "__main__":
    s = summary()
    print()
    print("  TRADE JOURNAL")
    print(f"    runs logged      {s['runs']}")
    print(f"    orders placed    {s['orders']}")
    print(f"    no-trade days    {s['no_trade']}")
    print(f"    blocked by risk  {s['blocked']}")
    print(f"    errors           {s['errors']}")
    if s["median_slippage_pct"] is not None:
        print(f"    median slippage  {s['median_slippage_pct']:+.3f}%  "
              f"({s['fills_with_slippage']} fills)")
    else:
        print("    median slippage  no fills yet")
    print()
    print(f"  {LOG}")
