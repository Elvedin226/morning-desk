"""Short a volume spike, cover the fade. The owner's pattern, tested mechanically.

The owner's real book this week: four for four, and every short cover was the
same shape - a name that spiked on enormous relative volume, shorted into or
just after the spike, covered as it faded (GPRO, MIMI, CHPT, TNON; SUNE called
and passed, +25% anyway). Ten strategies in this project have been put through
a permutation null. The one that is actually winning has not. That is the gap
this file closes.

THE RULE, stripped to something a machine can do:
    trigger   a session whose close/open move exceeds `move` AND whose volume
              exceeds `volmult` times the trailing 20-day average
    entry     short at the NEXT open (the bot cannot trade intraday)
    exit      cover at the close `hold` sessions later, or on a stop
    control   the same names on ordinary days, so any drift is netted out

UNIVERSE, AND WHY IT WILL UNDERSTATE. The owner's spikes are micro-floats -
TNON at $3.5M, SUNE at $28M. This project's universe is large caps, which do
not spike 40% in a day. Two reasons to test here anyway rather than on a
hand-picked list of small caps: (1) a hand-picked list of past spikers is the
exact selection error that produced a fake +33% options edge in this project,
and (2) if post-spike mean reversion is a real mechanism it should be visible
at ANY scale, the way multi-timeframe alignment was testable at weekly/daily.
So thresholds are scaled down and the result is a test of the MECHANISM.

What it cannot capture: the owner's discretion over which spike and when. The
read went 2 for 2 this week. No rule in this file has a read.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import watchlist

START = "2007-01-01"
HOLDS = (1, 2, 3, 5)
STOP = 0.10                       # cover if it runs this far against the short
GRID = [(0.08, 3), (0.12, 4), (0.15, 5), (0.20, 5)]   # (move, volmult)


def panel(tickers):
    d = yf.download(tickers, start=START, auto_adjust=True, progress=False, group_by="ticker")
    lvl0 = set(d.columns.get_level_values(0)) if isinstance(d.columns, pd.MultiIndex) else set()
    return {t: d[t].dropna() for t in tickers if t in lvl0 and len(d[t].dropna()) > 600}


def short_ret(o, h, c, entry_i, hold):
    """Short at the open of entry_i, cover at close entry_i+hold-1, stop on any
    high beyond STOP. Positive = the short made money."""
    e = o[entry_i]
    for j in range(entry_i, entry_i + hold):
        if h[j] >= e * (1 + STOP):
            return -STOP
    return 1 - c[entry_i + hold - 1] / e


def main():
    tickers = sorted(watchlist.SECTORS)
    print(f"\n  loading {len(tickers)} names from {START} ...")
    px = panel(tickers)
    print(f"  {len(px)} usable")
    rng = np.random.default_rng(0)

    for move, vm in GRID:
        sig = {h: [] for h in HOLDS}
        ctl = {h: [] for h in HOLDS}
        n_ev = 0
        for t, df in px.items():
            o, h, c, v = (df[k].to_numpy() for k in ("Open", "High", "Close", "Volume"))
            avgv = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
            dayret = c / o - 1
            n = len(c)
            ok = np.arange(21, n - max(HOLDS) - 1)
            trig = ok[(dayret[ok] > move) & (v[ok] > vm * avgv[ok])]
            n_ev += len(trig)
            if len(trig) == 0:
                continue
            # entry is the NEXT open after the spike day
            for hd in HOLDS:
                sig[hd].extend(short_ret(o, h, c, i + 1, hd) for i in trig)
                draws = rng.choice(ok, size=min(len(trig) * 5, len(ok)), replace=False)
                ctl[hd].extend(short_ret(o, h, c, i + 1, hd) for i in draws)

        print(f"\n  SPIKE > {move*100:.0f}% on > {vm}x volume   ({n_ev:,} events)")
        print("  " + "=" * 74)
        print(f"  {'hold':>5}{'n':>7}{'short ret':>11}{'win%':>7}{'control':>10}"
              f"{'edge':>9}{'Welch t':>10}")
        print("  " + "-" * 74)
        for hd in HOLDS:
            a = np.array(sig[hd]); b = np.array(ctl[hd])
            a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
            if len(a) < 30:
                print(f"  {hd:>5}{len(a):>7}   too few events")
                continue
            va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
            tt = (a.mean() - b.mean()) / np.sqrt(va + vb)
            print(f"  {hd:>5}{len(a):>7,}{a.mean()*100:>10.2f}%{np.mean(a>0)*100:>6.0f}%"
                  f"{b.mean()*100:>9.2f}%{(a.mean()-b.mean())*100:>+8.2f}%{tt:>10.2f}")

    print("\n  'edge' is the short's return on spike days minus its return on")
    print("  random days for the same names - what the SPIKE is worth, net of")
    print("  whatever shorting these names normally does. Stop is 10% against.")


if __name__ == "__main__":
    main()
