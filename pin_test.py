"""Does price get pulled toward big option strikes? Testing the pinning claim.

Gamma exposure makes two claims, and only one of them is directional:

  1. REGIME (not directional). Negative dealer gamma amplifies whatever move
     starts; positive gamma damps it and compresses the range. This says how
     price moves, never which way. The literature is consistent on this.
  2. PINNING (directional, and testable). Under positive gamma, hedging pulls
     price toward the strikes carrying the most open interest, so price should
     finish nearer those strikes than chance implies.

Claim 2 is the only part of GEX that says anything about where price goes, so
it is the only part worth testing against a directional claim.

THE OBSTACLE, AND THE WAY AROUND IT. Historical gamma cannot be reconstructed -
it needs the full option chain at each past date, and free sources give only
today's chain. But open interest is not spread evenly: it concentrates on round
strikes, and it concentrates enormously on MONTHLY expiries (the third Friday),
which carry multiples of the open interest of an ordinary session.

That gives a natural experiment needing no options history at all. If pinning is
real, closes should sit closer to round strikes on monthly OPEX than on ordinary
days. If the effect is imaginary, the distance distributions are the same.

WHAT THIS CANNOT SETTLE. It tests pinning toward round strikes, not toward the
specific peak-gamma strike, and it says nothing about claim 1. A null result
here is evidence against the pinning story, not against gamma exposure as a
description of volatility regimes.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

START = "2011-01-01"


def bars(t="SPY", adjust=False):
    """TWO price series are needed and mixing them up manufactures an edge.

    UNADJUSTED (adjust=False) for the strike-distance test: options are struck
    against the price that actually traded, so back-adjusting would slide every
    close off the strike grid it was measured on.

    ADJUSTED (adjust=True) for RETURNS. This one bit: 61 of 183 monthly OPEX
    days are also SPY ex-dividend days - the quarterly ex-date IS the third
    Friday - so on unadjusted prices a third of the sample carries a mechanical
    ~0.2% drop. That alone produced "OPEX days are bearish, t = -3.51, 42% up
    days". Adjusted it is t = -1.59; dropping the ex-div days entirely it is
    t = +0.03. The whole effect was the dividend.
    """
    d = yf.download(t, start=START, auto_adjust=adjust, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d.dropna()


def is_monthly_opex(idx: pd.DatetimeIndex) -> np.ndarray:
    """Third Friday of each month."""
    out = []
    for ts in idx:
        third_friday = ts.day > 14 and ts.day <= 21 and ts.weekday() == 4
        out.append(bool(third_friday))
    return np.array(out)


def dist_to_grid(close: np.ndarray, grid: float) -> np.ndarray:
    """Distance to the nearest multiple of `grid`, as a fraction of that grid.
    0 = exactly on a strike, 0.5 = as far away as it is possible to be."""
    return np.abs(close / grid - np.round(close / grid))


def welch(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    return float(t)


def main():
    df = bars("SPY")                      # unadjusted: strike distance
    adj = bars("SPY", adjust=True)        # adjusted: returns
    close = df["Close"].to_numpy()
    opex = is_monthly_opex(df.index)
    opex_a = is_monthly_opex(adj.index)
    print(f"\n  SPY, {df.index[0].date()} to {df.index[-1].date()}, "
          f"{len(df):,} sessions, {int(opex.sum())} monthly OPEX days")

    print("\n  DISTANCE FROM CLOSE TO NEAREST STRIKE  (0 = pinned, 0.5 = furthest)")
    print("  " + "=" * 74)
    print(f"  {'grid':>7}{'OPEX mean':>12}{'other mean':>12}{'diff':>10}"
          f"{'Welch t':>10}{'verdict':>14}")
    print("  " + "-" * 74)
    for grid in (1.0, 5.0, 10.0, 25.0):
        d = dist_to_grid(close, grid)
        a, b = d[opex], d[~opex]
        t = welch(a, b)
        # Pinning predicts OPEX closes are CLOSER, so a negative difference.
        verdict = "pinning" if (a.mean() < b.mean() and t < -2) else "no effect"
        print(f"  ${grid:>6.0f}{a.mean():>12.4f}{b.mean():>12.4f}"
              f"{a.mean()-b.mean():>+10.4f}{t:>10.2f}{verdict:>14}")

    print("\n  The uniform-distribution expectation is 0.2500. A pinned market")
    print("  should print meaningfully below that on OPEX days.")

    # The regime claim, which is the part GEX actually makes: is the RANGE
    # different? This does not need gamma either - OPEX days simply carry more
    # of it, whatever its sign.
    print("\n  INTRADAY RANGE, OPEX vs OTHER  (the non-directional claim)")
    print("  " + "=" * 74)
    rng = (adj["High"] / adj["Low"] - 1).to_numpy()
    ret = (adj["Close"] / adj["Open"] - 1).to_numpy()
    for label, arr in (("high/low range", rng), ("|open-to-close|", np.abs(ret))):
        a, b = arr[opex_a], arr[~opex_a]
        print(f"  {label:<18} OPEX {a.mean()*100:>6.3f}%   other {b.mean()*100:>6.3f}%"
              f"   diff {(a.mean()-b.mean())*100:>+6.3f}%   t {welch(a,b):>6.2f}")

    # And the directional question asked plainly.
    print("\n  DOES OPEX PREDICT DIRECTION AT ALL?")
    print("  " + "=" * 74)
    for label, arr in (("open-to-close", ret),
                       ("close-to-close", adj["Close"].pct_change().to_numpy())):
        a, b = arr[opex_a], arr[~opex_a]
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        print(f"  {label:<18} OPEX {a.mean()*100:>+7.4f}%  other {b.mean()*100:>+7.4f}%"
              f"   t {welch(a,b):>6.2f}   up-days {np.mean(a>0)*100:>4.1f}% vs {np.mean(b>0)*100:>4.1f}%")

    print("\n  A directional edge would need a t well beyond +/-2 here.")


if __name__ == "__main__":
    main()
