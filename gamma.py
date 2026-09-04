"""Dealer gamma exposure from the public options chain.

The claim GEX makes, and the reason it is worth measuring: option dealers hedge
delta continuously. When they hold NET LONG gamma they must buy weakness and
sell strength to stay hedged, which damps moves and pulls price toward the
strikes with the most open interest. When they are NET SHORT gamma the same
mechanism runs backwards and amplifies moves. The level where the aggregate
crosses zero - the "gamma flip" - is where the regime changes.

That mechanism is real and documented. What follows is the part every GEX
dashboard glosses over.

THE ASSUMPTION THAT CARRIES THE WHOLE CALCULATION. Open interest says how many
contracts exist at a strike. It does NOT say who is long and who is short. To
get a signed number you must assume a side, and the near-universal convention -
used here - is that dealers are LONG calls and SHORT puts. That is a guess. It
is roughly defensible because customer flow skews toward buying puts for
protection and selling covered calls, but on any given day it can be exactly
backwards, and no amount of arithmetic downstream repairs it.

So: treat the SHAPE of the profile (where the big strikes sit, where the sign
changes) as the signal, and the absolute dollar figure as decoration. Anyone
quoting "$4bn of gamma" to three significant figures is quoting an assumption.

WHY THIS FILE EXISTS. Snapshot the profile every day, then later ask whether
price actually did get pulled toward the peak strike and whether the flip level
held. Nobody selling a GEX subscription publishes that test. Until there are
enough snapshots the honest answer is that this is unvalidated - it is being
recorded so it CAN be validated, which is not the same as it working.
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

HISTORY = Path(__file__).parent / "data_cache" / "gex_history.jsonl"
# History is slimmed (no strike ladder) because only the levels get tested
# later. The chart needs the full ladder, so the latest profile is kept whole
# here - otherwise intraday runs, which read history, have nothing to draw.
LATEST = Path(__file__).parent / "data_cache" / "gex_latest.json"
CONTRACT = 100


def _gamma(spot, strike, iv, years):
    """Black-Scholes gamma. Identical for a call and a put at the same strike."""
    if years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1 = (np.log(spot / strike) + 0.5 * iv * iv * years) / (iv * np.sqrt(years))
    return float(np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (spot * iv * np.sqrt(years)))


def profile(ticker: str = "SPY", max_expiries: int = 3) -> dict | None:
    """Signed gamma per strike across the nearest expiries."""
    tk = yf.Ticker(ticker)
    try:
        hist = tk.history(period="1d")
        spot = float(hist["Close"].iloc[-1])
        expiries = tk.options[:max_expiries]
    except Exception:
        return None
    if not expiries or not np.isfinite(spot):
        return None

    today = datetime.now(timezone.utc).date()
    rows = {}
    legs = []          # (strike, oi, iv, years, sign) for the flip search
    used = []
    for exp in expiries:
        try:
            ch = tk.option_chain(exp)
        except Exception:
            continue
        days = max((datetime.strptime(exp, "%Y-%m-%d").date() - today).days, 0)
        # A 0DTE contract still has hours of life; treating it as zero would
        # zero out its gamma, which is exactly the strike that matters most.
        years = max(days, 0.5) / 365.0
        used.append(exp)

        for frame, sign in ((ch.calls, +1.0), (ch.puts, -1.0)):
            if frame is None or frame.empty:
                continue
            for _, r in frame.iterrows():
                k, oi, iv = r.get("strike"), r.get("openInterest"), r.get("impliedVolatility")
                if not (np.isfinite(k) and np.isfinite(oi) and np.isfinite(iv)) or oi <= 0:
                    continue
                if not (spot * 0.85 <= k <= spot * 1.15):
                    continue
                g = _gamma(spot, float(k), float(iv), years)
                # Dollar gamma per 1% move. The sign is the assumption above.
                val = sign * g * float(oi) * CONTRACT * spot * spot * 0.01
                rows[float(k)] = rows.get(float(k), 0.0) + val
                legs.append((float(k), float(oi), float(iv), years, sign))

    if not rows:
        return None

    strikes = np.array(sorted(rows))
    vals = np.array([rows[k] for k in strikes])

    # GAMMA FLIP. The level of SPOT at which total dealer gamma would be zero -
    # so it has to be found by repricing every strike's gamma at hypothetical
    # spot levels and looking for the sign change.
    #
    # The first version of this summed gamma-by-strike walking up the ladder and
    # called the zero crossing the flip. That is a different quantity entirely,
    # and it showed: SPY came back "negative gamma" while simultaneously
    # reporting spot ABOVE the flip, which is a contradiction - above the flip
    # IS the positive-gamma regime. A wrong number that disagrees with itself is
    # at least detectable; one that quietly agrees would not have been.
    flip = None
    grid = np.linspace(spot * 0.90, spot * 1.10, 161)
    totals = []
    for s_test in grid:
        tot = 0.0
        for (k, oi, iv, yrs, sign) in legs:
            tot += sign * _gamma(s_test, k, iv, yrs) * oi * CONTRACT * s_test * s_test * 0.01
        totals.append(tot)
    totals = np.array(totals)
    for i in range(1, len(totals)):
        if (totals[i - 1] < 0 <= totals[i]) or (totals[i - 1] > 0 >= totals[i]):
            a, b = totals[i - 1], totals[i]
            w = 0.0 if b == a else abs(a) / abs(b - a)
            flip = float(grid[i - 1] + w * (grid[i] - grid[i - 1]))
            break

    pos = strikes[np.argmax(vals)]
    neg = strikes[np.argmin(vals)]
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": ticker, "spot": round(spot, 2), "expiries": used,
        "total": float(vals.sum()),
        "flip": None if flip is None else round(flip, 2),
        "peak_call_strike": float(pos), "peak_call": float(vals.max()),
        "peak_put_strike": float(neg), "peak_put": float(vals.min()),
        "regime": "positive" if vals.sum() > 0 else "negative",
        "above_flip": None if flip is None else bool(spot > flip),
        "strikes": [[float(k), float(v)] for k, v in zip(strikes, vals)],
    }


def snapshot(tickers=("SPY", "QQQ")) -> list[dict]:
    """One row per ticker per day. Re-running the same day overwrites, so the
    15 scheduled runs do not turn into 15 phantom observations."""
    HISTORY.parent.mkdir(exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    keep = []
    if HISTORY.exists():
        for line in HISTORY.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (row.get("ts", "")[:10] == today and row.get("ticker") in tickers):
                keep.append(row)

    fresh = []
    for t in tickers:
        p = profile(t)
        if p:
            fresh.append(p)

    if fresh:
        LATEST.write_text(json.dumps(fresh, default=str), encoding="utf-8")

    with open(HISTORY, "w", encoding="utf-8") as f:
        for row in keep + fresh:
            # The full strike ladder is large and only the latest is useful
            # live; history keeps the levels, which is what gets tested.
            slim = {k: v for k, v in row.items() if k != "strikes"}
            f.write(json.dumps(slim, default=str) + "\n")
    return fresh


def load_latest() -> list[dict]:
    """Most recent full profiles, strike ladder included."""
    if not LATEST.exists():
        return []
    try:
        return json.loads(LATEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def load_history() -> list[dict]:
    if not HISTORY.exists():
        return []
    out = []
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


if __name__ == "__main__":
    for p in snapshot():
        print()
        print(f"  {p['ticker']}  spot ${p['spot']:,.2f}   expiries {', '.join(p['expiries'])}")
        print(f"    regime          {p['regime']} gamma"
              f"   (total ${p['total']/1e6:,.0f}M per 1% move)")
        if p["flip"] is not None:
            side = "ABOVE" if p["above_flip"] else "BELOW"
            print(f"    gamma flip      ${p['flip']:,.2f}   spot is {side} it")
        else:
            print("    gamma flip      not found in the +/-15% band")
        print(f"    largest call    ${p['peak_call_strike']:,.2f}  "
              f"(${p['peak_call']/1e6:,.0f}M)")
        print(f"    largest put     ${p['peak_put_strike']:,.2f}  "
              f"(${p['peak_put']/1e6:,.0f}M)")
    print()
    print(f"  {len(load_history())} snapshots recorded -> {HISTORY}")
    print("  UNVALIDATED. Recorded so it can be tested later, which is not the")
    print("  same as it working. The sign convention is an assumption; see the")
    print("  module docstring.")
