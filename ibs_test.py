"""Internal Bar Strength: is 12% CAGR at 46% exposure skill, or time in market?

Agent 3's run (mr_out_daily.txt) put IBS - buy a close in the bottom fifth of
the day's range, sell a close in the top fifth - at 12.19% CAGR, Sharpe 0.96,
max drawdown -20.1%, over 93 names 2000-2026 at 5bp. Buy and hold over the
same window: 17.93%, Sharpe 0.89, drawdown -48.7%. And IBS was POSITIVE in
every bear year: +11.92% in 2008 against -32.27%.

I flagged one check as necessary before believing it and then never ran it.
This is that check.

THE QUESTION. IBS is in the market 46% of the time. A coin flip that is long
46% of the time in a market returning 18% collects roughly 8% for doing
nothing clever. IBS collects 12%. Is the gap skill, or is it that the coin flip
comparison is too crude?

THE TEST. The same design that dismantled the EMA-pullback claim: take IBS
entries, take RANDOM entries on the same names over the same period at the
same count, and compare forward returns. If IBS carries information, the IBS
arm wins by more than noise. Post-2011 only - IBS is widely published and the
pre-publication record is not evidence for today.

Then the sharper version, because the by-year table is where IBS looked most
different: repeat the comparison inside 2018 and 2022, the two down years in
the window. A mean-reversion entry that also holds up when the tape is falling
is a different animal from one that only works because everything went up.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import watchlist

START = "2011-01-01"
ENTRY, EXIT = 0.2, 0.8            # Agent 3's config
HORIZONS = (1, 2, 3, 5)
DRAWS = 3                         # control entries per IBS entry


def panel(tickers):
    d = yf.download(tickers, start=START, auto_adjust=True, progress=False, group_by="ticker")
    lvl0 = set(d.columns.get_level_values(0)) if isinstance(d.columns, pd.MultiIndex) else set()
    return {t: d[t].dropna() for t in tickers if t in lvl0 and len(d[t].dropna()) > 500}


def welch(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def collect(px, years=None, seed=0):
    rng = np.random.default_rng(seed)
    sig = {h: [] for h in HORIZONS}
    ctl = {h: [] for h in HORIZONS}
    for t, df in px.items():
        if years is not None:
            df = df[df.index.year.isin(years)]
        if len(df) < 60:
            continue
        o, h, l, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
        rngw = h - l
        ibs = np.where(rngw > 0, (c - l) / np.where(rngw > 0, rngw, 1), np.nan)
        n = len(c)
        ok = np.arange(1, n - max(HORIZONS))
        fired = ok[ibs[ok] < ENTRY]
        if len(fired) < 5:
            continue
        draws = rng.choice(ok, size=min(len(fired) * DRAWS, len(ok)), replace=False)
        # Entry at the NEXT open, as the backtest engine does. Never the signal bar's close.
        for hd in HORIZONS:
            sig[hd].extend(c[fired + hd] / o[fired + 1] - 1)
            ctl[hd].extend(c[draws + hd] / o[draws + 1] - 1)
    return sig, ctl


def report(label, sig, ctl):
    print(f"\n  {label}")
    print("  " + "=" * 72)
    print(f"  {'days':>5}{'n IBS':>9}{'IBS':>9}{'n rand':>9}{'random':>9}{'edge':>9}{'Welch t':>10}")
    print("  " + "-" * 72)
    for hd in HORIZONS:
        a, b = np.array(sig[hd]), np.array(ctl[hd])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 30:
            print(f"  {hd:>5}{len(a):>9}   too few")
            continue
        print(f"  {hd:>5}{len(a):>9,}{a.mean()*100:>8.2f}%{len(b):>9,}{b.mean()*100:>8.2f}%"
              f"{(a.mean()-b.mean())*100:>+8.2f}%{welch(a,b):>10.2f}")


def main():
    tickers = sorted(watchlist.SECTORS) + ["SPY", "QQQ", "IWM"]
    print(f"\n  loading {len(tickers)} names from {START} ...")
    px = panel(tickers)
    print(f"  {len(px)} usable   IBS entry < {ENTRY}, control = {DRAWS}x random entries, same names")

    report("ALL YEARS 2011-2026", *collect(px))
    report("DOWN YEARS ONLY: 2018 + 2022", *collect(px, years=[2018, 2022], seed=1))
    report("2020 (the crash-and-melt-up year)", *collect(px, years=[2020], seed=2))

    print("\n  'edge' = IBS forward return minus a random entry on the same names in")
    print("  the same period. That nets out time-in-market. If the down-year edge")
    print("  holds, IBS is not just being long a rising tape.")


if __name__ == "__main__":
    main()
