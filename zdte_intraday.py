"""0DTE with INTRADAY management - correcting a model that answered the wrong question.

zdte_test.py measured 0DTE held from the open to expiry, and reported a ~46%
total-loss rate at the money. That is arithmetically right and practically
irrelevant, because it is not how the trades are actually taken. Real 0DTE
tickets get CUT: a loser closed at -15% never becomes a -100%, and the -100%
outcomes in the earlier study were almost entirely an artefact of assuming the
position is held into settlement.

Evidence from a real account, three consecutive 0DTE tickets:

    SPXW 7,735 Put   held to expiry     -100.00%
    SPXW 7,760 Call  cut intraday        -15.39%
    SPXW 7,695 Put   taken intraday      +13.21%

One of those is the old model's world. The other two are the strategy.

WHAT THIS FILE MODELS
    Entry at a chosen time of day, at the money, priced with Black-Scholes on
    the session's own realised volatility. The option is then REPRICED on every
    5-minute bar with the true remaining time to expiry, so theta decay and the
    delta path are both present. Exit on a stop, a target, or the close.

WHAT IS STILL AN APPROXIMATION
    Premium comes from Black-Scholes, which describes a 0DTE option badly -
    gamma is enormous near the strike and implied vol collapses into the close.
    The UNDERLYING path is real 5-minute data; only the option's price is
    modelled. Directionally this understates both tails.

    yfinance serves roughly 60 days of 5-minute history, so the sample is one
    market regime. Given that period bias has already corrupted two studies in
    this project, treat the level with suspicion and the COMPARISON - managed
    versus held - as the finding.
"""

from __future__ import annotations

import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from ivcheck import _bs_call, _bs_put

MINUTES_TO_CLOSE = 390
SPREAD = 0.014          # ~1.4% of premium round trip at the money, measured on SPY 0DTE


def session_bars(days: int = 58) -> dict:
    """Regular-hours 5-minute bars, grouped by session."""
    d = yf.download("SPY", period=f"{days}d", interval="5m",
                    prepost=False, progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.index = d.index.tz_convert("America/New_York")
    out = {}
    for day, g in d.groupby(d.index.date):
        rth = g.between_time("09:30", "16:00").dropna()
        if len(rth) >= 60:
            out[day] = rth
    return out


def price_at(spot, strike, vol, minutes_left, kind):
    """BS value with the true remaining life. At zero minutes this is intrinsic."""
    yrs = max(minutes_left, 0.0) / MINUTES_TO_CLOSE / 252
    if yrs <= 0:
        return max(0.0, spot - strike) if kind == "call" else max(0.0, strike - spot)
    f = _bs_call if kind == "call" else _bs_put
    v = f(spot, strike, vol, yrs)
    return v if np.isfinite(v) else 0.0


def run_day(bars, entry_hhmm="10:00", kind="call", stop=-0.15, target=0.15,
            hold_to_expiry=False, vol_floor=0.08):
    """One 0DTE ticket. Returns net return after the spread, or None."""
    idx = bars.index
    entry_i = None
    for i, ts in enumerate(idx):
        if ts.strftime("%H:%M") >= entry_hhmm:
            entry_i = i
            break
    if entry_i is None or entry_i >= len(idx) - 6:
        return None

    # Session volatility from the bars BEFORE entry, so nothing looks ahead.
    pre = bars["Close"].iloc[: entry_i + 1]
    if len(pre) < 4:
        return None
    vol = float(pre.pct_change().std() * np.sqrt(78 * 252))
    if not np.isfinite(vol) or vol < vol_floor:
        vol = vol_floor

    spot0 = float(bars["Close"].iloc[entry_i])
    strike = round(spot0)
    mins_left0 = (len(idx) - entry_i) * 5
    prem0 = price_at(spot0, strike, vol, mins_left0, kind)
    if prem0 <= 0.02:
        return None

    for j in range(entry_i + 1, len(idx)):
        spot = float(bars["Close"].iloc[j])
        mins = (len(idx) - j) * 5
        val = price_at(spot, strike, vol, mins, kind)
        r = val / prem0 - 1.0
        if not hold_to_expiry and (r <= stop or r >= target):
            return (1 + r) * (1 - SPREAD) - 1

    settle = float(bars["Close"].iloc[-1])
    r = price_at(settle, strike, vol, 0, kind) / prem0 - 1.0
    return (1 + r) * (1 - SPREAD) - 1


def summarise(rs):
    rs = np.array([r for r in rs if r is not None])
    if len(rs) == 0:
        return None
    return {"n": len(rs), "ev": rs.mean(), "win": float((rs > 0).mean()),
            "total_loss": float((rs <= -0.99).mean()),
            "worst": rs.min(), "best": rs.max(), "median": float(np.median(rs))}


def main():
    sess = session_bars()
    print(f"\n  {len(sess)} sessions of real 5-minute SPY bars, "
          f"{min(sess)} to {max(sess)}")

    print("\n  MANAGED vs HELD TO EXPIRY  (ATM, entry 10:00, both directions)")
    print("  " + "=" * 78)
    print(f"  {'management':<26}{'n':>5}{'EV':>10}{'win%':>8}{'-100%':>8}"
          f"{'worst':>9}{'best':>8}")
    print("  " + "-" * 78)

    for label, kw in (
        ("held to expiry", dict(hold_to_expiry=True)),
        ("stop -15% / target +15%", dict(stop=-0.15, target=0.15)),
        ("stop -25% / target +25%", dict(stop=-0.25, target=0.25)),
        ("stop -15% / target +40%", dict(stop=-0.15, target=0.40)),
        ("stop -50% / target +100%", dict(stop=-0.50, target=1.00)),
    ):
        rs = []
        for d, b in sess.items():
            for k in ("call", "put"):
                rs.append(run_day(b, kind=k, **kw))
        s = summarise(rs)
        if s:
            print(f"  {label:<26}{s['n']:>5}{s['ev']*100:>9.1f}%{s['win']*100:>7.0f}%"
                  f"{s['total_loss']*100:>7.0f}%{s['worst']*100:>8.0f}%{s['best']*100:>7.0f}%")

    print("\n  WHAT TIME OF DAY  (stop -15% / target +15%)")
    print("  " + "=" * 78)
    print(f"  {'entry':<26}{'n':>5}{'EV':>10}{'win%':>8}{'median':>10}")
    print("  " + "-" * 78)
    for t in ("09:35", "10:00", "11:00", "13:00", "14:30"):
        rs = [run_day(b, entry_hhmm=t, kind=k) for d, b in sess.items()
              for k in ("call", "put")]
        s = summarise(rs)
        if s:
            print(f"  {t:<26}{s['n']:>5}{s['ev']*100:>9.1f}%{s['win']*100:>7.0f}%"
                  f"{s['median']*100:>9.1f}%")

    # The specific week, and specifically the sessions that were skipped.
    print("\n  THIS WEEK, DAY BY DAY  (stop -15% / target +15%, entry 10:00)")
    print("  " + "=" * 78)
    print(f"  {'session':<14}{'SPY move':>10}{'call':>9}{'put':>9}"
          f"{'best side':>11}{'both':>9}")
    print("  " + "-" * 78)
    week = sorted(d for d in sess if str(d) >= "2026-08-31")
    tot_best = tot_both = 0.0
    for d in week:
        b = sess[d]
        mv = float(b["Close"].iloc[-1]) / float(b["Open"].iloc[0]) - 1
        c = run_day(b, kind="call")
        p = run_day(b, kind="put")
        if c is None or p is None:
            continue
        best, both = max(c, p), c + p
        tot_best += best
        tot_both += both
        print(f"  {str(d):<14}{mv*100:>9.2f}%{c*100:>8.1f}%{p*100:>8.1f}%"
              f"{best*100:>10.1f}%{both*100:>8.1f}%")
    print("  " + "-" * 78)
    print(f"  {'week total':<14}{'':>10}{'':>9}{'':>9}{tot_best*100:>10.1f}%{tot_both*100:>8.1f}%")
    print("\n  'best side' is hindsight - it picks the winning direction after the")
    print("  fact and is an upper bound no one can trade. 'both' is what taking")
    print("  a call and a put every session actually returns.")


if __name__ == "__main__":
    main()
