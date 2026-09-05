"""Buying pullbacks to the 50/200 EMA. Does the touch add anything to the trend?

The rule, as stated: on weekly bars, in a bullish trend, buy when price falls
back to the 50 EMA or the 200 EMA. Claimed to be "very very accurate" on any
stock.

WHY THE VIDEO CANNOT ESTABLISH THIS, whatever the charts show. Every example is
narrated backwards over a chart that already printed - AAPL, GOOGL, NVDA, all
large winners. On a stock that went up, price will touch its 50 EMA repeatedly
and bounce most times, because the stock went up. The bounces are a CONSEQUENCE
of the uptrend, not evidence that the touch predicted anything. The tell is at
4:33: "I haven't even looked at NVIDIA's chart but I guarantee you you'll be
able to tell."

So the test cannot be "did EMA entries make money" - in a rising market they
will, and so would entries chosen by coin flip. The test has to be:

    do EMA-touch entries beat RANDOM entries taken under the SAME
    trend conditions, on the same names, over the same period?

That comparison holds the trend constant and isolates the only thing in
dispute: whether touching the moving average carries information.

If the touch is real, its forward returns beat the random control. If it is
just a way of being long a rising stock, both arms land in the same place - and
the honest conclusion is that the trend was the edge and the EMA was decoration.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import watchlist

START = "2007-01-01"
NEAR = 0.02          # "at the EMA" = within 2%
HORIZONS = (4, 8, 13, 26)    # weeks forward
RANDOM_DRAWS = 40            # control samples per name


def weekly(tickers):
    d = yf.download(tickers, start=START, auto_adjust=True, progress=False,
                    group_by="ticker", interval="1wk")
    lvl0 = set(d.columns.get_level_values(0)) if isinstance(d.columns, pd.MultiIndex) else set()
    out = {}
    for t in tickers:
        if t not in lvl0:
            continue
        c = d[t]["Close"].dropna()
        if len(c) > 260:
            out[t] = c
    return out


def main():
    tickers = sorted(watchlist.SECTORS)
    print(f"\n  loading weekly bars for {len(tickers)} names from {START} ...")
    px = weekly(tickers)
    print(f"  {len(px)} usable")

    rng = np.random.default_rng(0)
    sig = {h: [] for h in HORIZONS}      # entries at an EMA touch
    ctl = {h: [] for h in HORIZONS}      # random entries, same uptrend filter
    n_touch = n_bull = 0

    for t, c in px.items():
        e50 = c.ewm(span=50, adjust=False).mean()
        e200 = c.ewm(span=200, adjust=False).mean()
        # "Bullish trend" as described: above the long average.
        bull = (c > e200).to_numpy()
        near50 = (np.abs(c / e50 - 1) <= NEAR).to_numpy()
        near200 = (np.abs(c / e200 - 1) <= NEAR).to_numpy()
        touch = bull & (near50 | near200)

        arr = c.to_numpy()
        n = len(arr)
        valid = np.arange(200, n - max(HORIZONS))
        if len(valid) < 60:
            continue

        t_idx = valid[touch[valid]]
        b_idx = valid[bull[valid]]          # the control pool: bullish, any week
        n_touch += len(t_idx)
        n_bull += len(b_idx)
        if len(t_idx) == 0 or len(b_idx) < 20:
            continue

        for h in HORIZONS:
            sig[h].extend(arr[t_idx + h] / arr[t_idx] - 1)
            draws = rng.choice(b_idx, size=min(RANDOM_DRAWS, len(b_idx)), replace=False)
            ctl[h].extend(arr[draws + h] / arr[draws] - 1)

    print(f"\n  {n_touch:,} EMA-touch weeks inside an uptrend, "
          f"out of {n_bull:,} uptrend weeks ({n_touch/n_bull*100:.0f}%)")

    print("\n  EMA TOUCH vs RANDOM ENTRY, both inside the same uptrend")
    print("  " + "=" * 76)
    print(f"  {'weeks':>6}{'n touch':>10}{'touch':>9}{'n random':>11}{'random':>9}"
          f"{'diff':>9}{'Welch t':>10}")
    print("  " + "-" * 76)
    for h in HORIZONS:
        a = np.array(sig[h]); b = np.array(ctl[h])
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        tstat = (a.mean() - b.mean()) / np.sqrt(va + vb)
        print(f"  {h:>6}{len(a):>10,}{a.mean()*100:>8.2f}%{len(b):>11,}"
              f"{b.mean()*100:>8.2f}%{(a.mean()-b.mean())*100:>+8.2f}%{tstat:>10.2f}")

    print("\n  The control is not a straw man: it is long the same names, in the")
    print("  same uptrends, over the same weeks - just entered at random instead")
    print("  of at the moving average. Any gap is what the EMA touch is worth.")

    # Hit rate, since the claim was about accuracy rather than return.
    print("\n  'VERY ACCURATE'?  share of entries profitable")
    print("  " + "=" * 76)
    for h in HORIZONS:
        a = np.array(sig[h]); b = np.array(ctl[h])
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        print(f"  {h:>3} weeks   touch {np.mean(a>0)*100:>5.1f}%   "
              f"random {np.mean(b>0)*100:>5.1f}%   diff {(np.mean(a>0)-np.mean(b>0))*100:>+5.1f} pts")


if __name__ == "__main__":
    main()
