"""Find the owner's setups and hand them over. The bot does not trade these.

The decision this replaces: after a week where the owner went four for four
on spike-fade shorts and the bot went 0 closed / -1.5% on 21-day swings, the
question was whether to make the bot trade the owner's pattern mechanically or
to make it FIND the pattern and let the owner decide. spike_fade_test.py settled
it: the mechanism shows at every threshold (11 of 12 cells positive) but never
clears t = 2, and it is strongest exactly where a bot cannot go - micro-floats
it cannot borrow, timed intraday. The read went 2 for 2. No rule has a read.

So this scans, ranks, and stops. Every 30 minutes it asks Yahoo's screener for
the day's biggest US movers, drops anything under $1 (Robinhood will not let
you short below that), and enriches the survivors with the fields the owner
asks for before a trade: relative volume, float, gap, RSI on both frames,
VWAP, distance off the high.

RANKED BY RELATIVE VOLUME, not by move. A 40% move on 3x volume is a thin pop;
a 40% move on 190x volume is a frenzy, and the frenzy is the setup. TNON on
Sep 10 2026: float 620,847, volume 118M.

What "shortable" means here: price >= $1. Borrow availability is broker-side
and unknowable from public data, so the column is honest about what it checks.
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
from yfinance import EquityQuery as Q

LATEST = Path(__file__).parent / "data_cache" / "scan_latest.json"

MIN_PRICE = 1.0        # Robinhood short floor
MIN_MOVE = 15.0        # % on the day; the owner trades the tail, not the middle
MIN_VOL = 500_000      # shares; below this the tape is too thin to matter
TOP = 20               # names to enrich


def _rsi(s: pd.Series, n: int = 14) -> float:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    r = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    v = float(r.iloc[-1])
    return v if np.isfinite(v) else float("nan")


def movers() -> list[dict]:
    """Today's biggest US gainers above the price and volume floors."""
    q = Q("and", [Q("gt", ["percentchange", MIN_MOVE]),
                  Q("gte", ["intradayprice", MIN_PRICE]),
                  Q("eq", ["region", "us"]),
                  Q("gt", ["dayvolume", MIN_VOL])])
    r = yf.screen(q, sortField="percentchange", sortAsc=False, size=60)
    out = []
    for x in r.get("quotes", []):
        px = x.get("regularMarketPrice")
        if not px or px < MIN_PRICE:
            continue
        out.append({"ticker": x.get("symbol"), "name": (x.get("shortName") or "")[:30],
                    "price": float(px), "move": float(x.get("regularMarketChangePercent") or 0),
                    "volume": int(x.get("regularMarketVolume") or 0),
                    "mktcap": int(x.get("marketCap") or 0)})
    return out


def enrich(row: dict) -> dict:
    """Daily context and today's intraday shape. Failures leave fields NaN
    rather than dropping the name - a missing float is not a reason to hide a
    190x volume day."""
    t = row["ticker"]
    try:
        d = yf.download(t, period="2mo", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.dropna()
        c = d["Close"]
        avgv = float(d["Volume"].iloc[-21:-1].mean()) if len(d) > 21 else float("nan")
        row["rel_vol"] = row["volume"] / avgv if avgv and avgv > 0 else float("nan")
        row["rsi_d"] = _rsi(c) if len(c) > 15 else float("nan")
        prev = float(c.iloc[-2]) if len(c) > 1 else float("nan")
        today_o = float(d["Open"].iloc[-1])
        row["gap"] = (today_o / prev - 1) * 100 if np.isfinite(prev) else float("nan")
        row["gap_lo"], row["gap_hi"] = (prev, today_o) if today_o > prev else (today_o, prev)
        row["lo20"] = float(c.iloc[-21:-1].min()) if len(c) > 21 else float("nan")
        row["from_lo20"] = (row["price"] / row["lo20"] - 1) * 100 if np.isfinite(row.get("lo20", np.nan)) else float("nan")
    except Exception:
        for k in ("rel_vol", "rsi_d", "gap", "from_lo20"):
            row[k] = float("nan")
    try:
        i = yf.download(t, period="1d", interval="5m", prepost=False, progress=False, auto_adjust=True)
        if isinstance(i.columns, pd.MultiIndex):
            i.columns = i.columns.get_level_values(0)
        i = i.dropna()
        hi = float(i["High"].max())
        row["day_high"] = hi
        row["off_high"] = (row["price"] / hi - 1) * 100
        row["vwap"] = float((i["Close"] * i["Volume"]).sum() / i["Volume"].sum())
        row["vs_vwap"] = (row["price"] / row["vwap"] - 1) * 100
        row["rsi_5m"] = _rsi(i["Close"]) if len(i) > 15 else float("nan")
        # Momentum of the last hour vs the prior hour: still climbing, or rolling?
        h1 = i.tail(12); h0 = i.iloc[-24:-12] if len(i) >= 24 else i.iloc[:max(1, len(i) - 12)]
        row["last_hr"] = (float(h1["Close"].iloc[-1]) / float(h1["Open"].iloc[0]) - 1) * 100
        row["prev_hr"] = (float(h0["Close"].iloc[-1]) / float(h0["Open"].iloc[0]) - 1) * 100
    except Exception:
        for k in ("off_high", "vs_vwap", "rsi_5m", "last_hr", "prev_hr"):
            row[k] = float("nan")
    try:
        info = yf.Ticker(t).info or {}
        row["float"] = int(info.get("floatShares") or 0)
        si = info.get("shortPercentOfFloat")
        row["short_pct"] = float(si) * 100 if si else float("nan")
    except Exception:
        row["float"], row["short_pct"] = 0, float("nan")
    row["float_turnover"] = row["volume"] / row["float"] if row.get("float") else float("nan")
    row["shortable_price"] = row["price"] >= MIN_PRICE
    return row


def scan() -> dict:
    rows = movers()[:TOP]
    rows = [enrich(r) for r in rows]
    # Float turnover first. TNON on Sep 10: 193x float, only 28x relative volume,
    # because its prior 20 days were already busy. A dormant name waking up can
    # post 800x relative volume on 12x float. Turnover says how much of the
    # company changed hands today; that is the frenzy. Rel-vol is the fallback
    # when float is unknown.
    def _key(r):
        ft, rv = r.get("float_turnover", np.nan), r.get("rel_vol", np.nan)
        return -(ft if np.isfinite(ft) else (rv / 100 if np.isfinite(rv) else 0))
    rows.sort(key=_key)
    out = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "filters": {"min_price": MIN_PRICE, "min_move_pct": MIN_MOVE, "min_volume": MIN_VOL},
           "rows": rows}
    LATEST.parent.mkdir(exist_ok=True)
    # NaN is not JSON. json.dumps writes float('nan') as a bare NaN token - which
    # JSON.parse rejects, blanking the whole panel - and never routes it through
    # `default`, so a hook there is useless. Walk the structure first.
    def _clean(v):
        if isinstance(v, dict):
            return {k: _clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_clean(x) for x in v]
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            return None
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v
    # Clean ONCE and return the same object that was written. The first version
    # cleaned only what it saved and returned the raw dict, so the file on disk
    # was valid and the dashboard payload carried eleven NaN tokens.
    out = _clean(out)
    LATEST.write_text(json.dumps(out), encoding="utf-8")
    return out


def load_latest() -> dict:
    if not LATEST.exists():
        return {"ts": None, "rows": []}
    try:
        return json.loads(LATEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ts": None, "rows": []}


def _f(v, fmt):
    return fmt % v if v is not None and isinstance(v, (int, float)) and np.isfinite(v) else "   -"


if __name__ == "__main__":
    s = scan()
    print(f"\n  SCAN {s['ts']}   {len(s['rows'])} names   "
          f"filters: >= ${MIN_PRICE}, > {MIN_MOVE:.0f}%, > {MIN_VOL:,} shares")
    print("  " + "=" * 112)
    print(f"  {'tkr':<6}{'price':>7}{'move':>8}{'relvol':>8}{'flt x':>7}{'mcap':>9}"
          f"{'gap':>7}{'offhi':>7}{'vVWAP':>7}{'RSI d':>7}{'RSI5m':>7}{'lasthr':>8}{'prevhr':>8}")
    print("  " + "-" * 112)
    for r in s["rows"]:
        mc = r["mktcap"] / 1e6
        print(f"  {r['ticker']:<6}{r['price']:>7.2f}{r['move']:>+7.1f}%{_f(r.get('rel_vol'), '%7.0fx')}"
              f"{_f(r.get('float_turnover'), '%6.1fx')}{mc:>8.0f}M{_f(r.get('gap'), '%+6.1f%%')}"
              f"{_f(r.get('off_high'), '%+6.1f%%')}{_f(r.get('vs_vwap'), '%+6.1f%%')}"
              f"{_f(r.get('rsi_d'), '%7.0f')}{_f(r.get('rsi_5m'), '%7.0f')}"
              f"{_f(r.get('last_hr'), '%+7.1f%%')}{_f(r.get('prev_hr'), '%+7.1f%%')}")
    print("\n  ranked by float turnover ('flt x' = today's volume / float); rel-vol when float is unknown.")
    print("  shortable = price >= $1 only; borrow is broker-side and not checkable here.")
