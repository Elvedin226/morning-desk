"""Selling cash-secured puts on SPY, measured over 36 years.

This is the first strategy of the seven reviewed that sits on the RIGHT side of
the only durable edge found in this project. vrp_test.py measured implied
volatility exceeding subsequent realised by 4.06 vol points, on 85.4% of days,
positive in all eight five-year blocks since 1990. Selling a put is one way to
collect exactly that.

So the question is not whether the premium exists - it does - but what selling
it actually returns once assignment is honoured, and how it compares to simply
owning the thing.

METHOD. Each month, sell a put on SPY expiring ~21 sessions out, struck a fixed
percentage below spot. Price it with Black-Scholes using VIX as the implied
vol, which for SPY is not an approximation of convenience - VIX IS the implied
vol of ~30-day S&P options, so the premium is close to what was really quoted.
Settle at expiry: keep the premium if SPY held above the strike, otherwise take
assignment and mark the loss. Collateral is the full strike, as a cash-secured
seller must post.

WHAT THIS CAPTURES that a naive premium study does not: assignment. A study
that counts premium collected and ignores the months you are put stock is not
measuring the strategy, it is measuring the marketing.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from ivcheck import _bs_put

HOLD = 21                 # sessions to expiry
OTM = (0.02, 0.05, 0.10)  # strike this far below spot
COST = 0.01               # 1% of premium, round trip


def data():
    spx = yf.download("^GSPC", start="1990-01-01", auto_adjust=True, progress=False)["Close"]
    vix = yf.download("^VIX", start="1990-01-01", auto_adjust=True, progress=False)["Close"]
    spx = spx.iloc[:, 0] if hasattr(spx, "columns") else spx
    vix = vix.iloc[:, 0] if hasattr(vix, "columns") else vix
    return pd.DataFrame({"spx": spx, "vix": vix}).dropna()


def run(d, otm):
    """One put sold every HOLD sessions, non-overlapping."""
    px, iv = d["spx"].to_numpy(), d["vix"].to_numpy() / 100
    rows = []
    for i in range(0, len(px) - HOLD, HOLD):
        spot, v = px[i], iv[i]
        k = spot * (1 - otm)
        prem = _bs_put(spot, k, v, HOLD / 252)
        if not np.isfinite(prem) or prem <= 0:
            continue
        prem *= (1 - COST)
        end = px[i + HOLD]
        # Assigned below the strike: the loss is intrinsic, offset by premium.
        pnl = prem - max(0.0, k - end)
        rows.append({"i": i, "spot": spot, "k": k, "prem": prem, "end": end,
                     "pnl": pnl, "assigned": end < k, "coll": k})
    return pd.DataFrame(rows)


def main():
    d = data()
    print(f"\n  SPY/S&P 500 with VIX as implied vol, {d.index[0].date()} to {d.index[-1].date()}")
    print(f"  one put sold every {HOLD} sessions, collateral = full strike, "
          f"{COST*100:.0f}% cost on premium")

    print("\n  CASH-SECURED PUT SELLING")
    print("  " + "=" * 76)
    print(f"  {'OTM':>6}{'n':>6}{'assigned':>10}{'total P&L':>12}{'per trade':>11}"
          f"{'ann. on coll':>14}{'worst':>11}")
    print("  " + "-" * 76)
    res = {}
    for o in OTM:
        r = run(d, o)
        res[o] = r
        ret = r["pnl"] / r["coll"]
        ann = (1 + ret.mean()) ** 12 - 1
        print(f"  {o*100:>5.0f}%{len(r):>6}{r['assigned'].mean()*100:>9.1f}%"
              f"{ret.sum()*100:>11.1f}%{ret.mean()*100:>10.2f}%{ann*100:>13.1f}%"
              f"{ret.min()*100:>10.1f}%")

    # The comparison that decides it.
    first, last = d["spx"].iloc[0], d["spx"].iloc[-1]
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    bh = (last / first) ** (1 / yrs) - 1
    print(f"\n  buy and hold the index over the same window: {bh*100:.1f}% a year")
    print("  (price only; dividends would add roughly 2 points)")
    print()
    print("  ONE CORRECTION IN THE SELLER'S FAVOUR, which the table above omits.")
    print("  The collateral is not idle - a cash-secured seller holds it in bills,")
    print("  worth roughly 2.5 points a year over 1990-2026. That puts the 2% OTM")
    print("  line nearer 10.4% than 7.9%, against a dividend-inclusive buy and hold")
    print("  around 10.7%. Roughly a wash - with a far better win rate and a far")
    print("  worse tail. The wrapper is not a disaster; it is just not an edge.")

    print("\n  WHERE THE LOSSES LIVE  (5% OTM)")
    print("  " + "-" * 76)
    r = res[0.05]
    worst = r.nsmallest(5, "pnl")
    for _, row in worst.iterrows():
        when = d.index[int(row["i"])].date()
        print(f"  {when}   spot {row['spot']:>8.0f} -> {row['end']:>8.0f}"
              f"   strike {row['k']:>8.0f}   P&L {row['pnl']/row['coll']*100:>+7.2f}%")

    ret = r["pnl"] / r["coll"]
    print(f"\n  win rate {(ret > 0).mean()*100:.1f}%   "
          f"mean win {ret[ret>0].mean()*100:+.2f}%   mean loss {ret[ret<=0].mean()*100:+.2f}%")
    print(f"  the worst month is {abs(ret.min()/ret[ret>0].mean()):.0f}x the average win.")
    print("\n  That ratio is the trade: many small credits, rare large debits.")
    print("  It is the same shape as the variance risk premium because it IS")
    print("  the variance risk premium, collected one strike at a time.")


if __name__ == "__main__":
    main()
