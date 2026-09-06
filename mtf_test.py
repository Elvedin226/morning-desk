"""Does higher-timeframe alignment improve lower-timeframe entries?

The claim: use a higher timeframe to pick direction, then drop to an "aligned"
lower timeframe for the entry (4H -> 15m, 1H -> 5m). Filtering entries by the
higher trend is supposed to make them meaningfully better.

That is a real, falsifiable proposition and it does not depend on any of the
surrounding vocabulary. Strip out "liquidity sweep" and "bull trap" and the
testable core is:

    take the SAME entries, then split them by whether the higher timeframe
    agreed. If alignment carries information, the aligned half wins by more.

SCALE. The claim is made at 1H/5m. yfinance serves about 60 days of 5-minute
bars - one regime, a few hundred entries, and this project has already been
burned twice by single-regime samples. So the principle is tested at
weekly/daily instead: same 12:1 timeframe ratio he uses, nineteen years of
history, 90 names. If higher-timeframe alignment is a real property of markets
it should show at both scales; if it only appears at 5 minutes it is a claim
about microstructure that 60 days cannot settle either.

PRIOR. The regime filter already tested in this project is the same idea in
index form - condition entries on a higher-timeframe trend - and it measured
nothing at the 21-day horizon (t = -0.57), while actively hurting at 63 days.
This asks the sharper version: alignment on the SAME instrument.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import watchlist

START = "2007-01-01"
HORIZONS = (3, 5, 10, 21)      # trading days held


def panel(tickers):
    d = yf.download(tickers, start=START, auto_adjust=True, progress=False,
                    group_by="ticker")
    lvl0 = set(d.columns.get_level_values(0)) if isinstance(d.columns, pd.MultiIndex) else set()
    return {t: d[t].dropna() for t in tickers if t in lvl0 and len(d[t].dropna()) > 600}


def main():
    tickers = sorted(watchlist.SECTORS)
    print(f"\n  loading {len(tickers)} names from {START} ...")
    px = panel(tickers)
    print(f"  {len(px)} usable")

    # Two lower-timeframe entry triggers, deliberately ordinary. The question is
    # what ALIGNMENT does to them, not whether the trigger itself is clever.
    entries = {
        "pullback to 20ema": lambda c, e20, e50: (np.abs(c / e20 - 1) <= 0.01),
        "2 consecutive down days": lambda c, e20, e50: (
            c.pct_change().lt(0) & c.pct_change().shift(1).lt(0)),
    }

    for name, trig in entries.items():
        al = {h: [] for h in HORIZONS}       # higher timeframe agrees
        ag = {h: [] for h in HORIZONS}       # it does not
        for t, df in px.items():
            c = df["Close"]
            e20 = c.ewm(span=20, adjust=False).mean()
            e50 = c.ewm(span=50, adjust=False).mean()
            # HIGHER TIMEFRAME: weekly trend, resampled then mapped back to
            # daily. ffill is what makes this honest - each day only ever sees
            # the last COMPLETED week, never the one it sits inside.
            wk = c.resample("W-FRI").last()
            wk_up = (wk > wk.ewm(span=20, adjust=False).mean()).shift(1)
            # astype(bool) is required, not cosmetic: .shift(1) on a boolean
            # Series promotes it to object dtype, and an object array cannot be
            # used as a mask.
            higher = (wk_up.reindex(c.index, method="ffill")
                      .fillna(False).astype(bool).to_numpy())

            fired = trig(c, e20, e50).fillna(False).to_numpy()
            arr = c.to_numpy()
            n = len(arr)
            idx = np.arange(60, n - max(HORIZONS))
            f = idx[fired[idx]]
            if len(f) < 20:
                continue
            for h in HORIZONS:
                r = arr[f + h] / arr[f] - 1
                m = higher[f]
                al[h].extend(r[m])
                ag[h].extend(r[~m])

        print(f"\n  ENTRY: {name}")
        print("  " + "=" * 74)
        print(f"  {'days':>5}{'n aligned':>11}{'aligned':>10}{'n against':>11}"
              f"{'against':>10}{'diff':>9}{'Welch t':>10}")
        print("  " + "-" * 74)
        for h in HORIZONS:
            a = np.array(al[h]); b = np.array(ag[h])
            a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
            if len(a) < 50 or len(b) < 50:
                continue
            va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
            t = (a.mean()-b.mean())/np.sqrt(va+vb)
            print(f"  {h:>5}{len(a):>11,}{a.mean()*100:>9.2f}%{len(b):>11,}"
                  f"{b.mean()*100:>9.2f}%{(a.mean()-b.mean())*100:>+8.2f}%{t:>10.2f}")

    print("\n  Alignment is worth whatever the diff column says, and no more.")
    print("  A positive, significant gap means the higher timeframe carries")
    print("  information. Anything else means it is a filter that removes")
    print("  trades without improving them.")


if __name__ == "__main__":
    main()
