"""The variance risk premium: the one durable edge measured in this project.

Nine trading strategies were tested here and eight failed outright; the survivor
(Connors RSI-2) cleared its null at p=0.05 and still lost to buying the index.
This is different, and it is worth being precise about why.

THE CLAIM. Options are priced on IMPLIED volatility, and implied volatility is
systematically higher than the volatility that subsequently arrives. Whoever
sells the option collects that difference. It is not a market inefficiency - it
is payment for bearing crash risk, which is exactly why it has not been
arbitraged away in thirty-six years.

WHY THIS FILE EXISTS RATHER THAN A ONE-LINER. A cross-sectional snapshot of
today's implied vol against TRAILING realised vol across 43 names gave
IV/RV = 1.19, 77% of names, t = 3.93. That number is not trustworthy: trailing
realised vol is not what an option pays out on. If volatility is currently
depressed and mean-reverting, implied above trailing realised simply means the
market expects a pickup - and it may be right.

The honest test compares implied volatility to the realised volatility that
FOLLOWS it. VIX against the next 21 sessions of S&P realised vol does exactly
that, needs no option-chain history, and reaches back to 1990.

WHAT IT DOES NOT SAY. That selling volatility is safe. The premium is
compensation for a left tail that arrives rarely and violently - see the
drawdown table, where every one of the six worst observations is February 2020
and each loses roughly seventeen times the average premium at once. An edge and
a safe trade are different things, and this is emphatically the first.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

HORIZON = 21          # sessions; roughly a one-month option


def series(ticker: str, start="1990-01-01") -> pd.Series:
    s = yf.download(ticker, start=start, auto_adjust=True, progress=False)["Close"]
    return s.iloc[:, 0] if hasattr(s, "columns") else s


def build() -> pd.DataFrame:
    d = pd.DataFrame({"vix": series("^VIX"), "spx": series("^GSPC")}).dropna()
    r = d["spx"].pct_change()
    # Volatility over the NEXT `HORIZON` sessions. The two shifts matter: the
    # window must start the day AFTER the observation and be stamped back onto
    # it, or the "forward" figure quietly includes the day it is predicting.
    d["fwd"] = r.shift(-1).rolling(HORIZON).std().shift(-(HORIZON - 1)) * np.sqrt(252) * 100
    d = d.dropna()
    d["prem"] = d["vix"] - d["fwd"]
    return d


def main():
    d = build()
    print(f"\n  VIX(t) vs REALISED vol over the following {HORIZON} sessions")
    print(f"  {d.index[0].date()} to {d.index[-1].date()}, {len(d):,} observations")
    print("  " + "=" * 74)
    print(f"    mean VIX {d.vix.mean():.2f}   mean subsequent realised {d.fwd.mean():.2f}"
          f"   premium {d.prem.mean():+.2f} vol points")
    print(f"    VIX exceeded subsequent realised on {(d.prem > 0).mean()*100:.1f}% of days")
    print(f"    median premium {d.prem.median():+.2f}   VIX/realised {(d.vix/d.fwd).median():.3f}")

    print("\n  BY FIVE-YEAR PERIOD  (does one regime carry it?)")
    print("  " + "-" * 74)
    print(f"  {'period':<12}{'n':>7}{'VIX':>9}{'realised':>11}{'premium':>10}{'% positive':>13}")
    d = d.assign(blk=(d.index.year // 5) * 5)
    for blk, g in d.groupby("blk"):
        if len(g) < 200:
            continue
        print(f"  {f'{blk}-{blk+4}':<12}{len(g):>7,}{g.vix.mean():>9.2f}"
              f"{g.fwd.mean():>11.2f}{g.prem.mean():>10.2f}{(g.prem>0).mean()*100:>12.1f}%")

    print("\n  WHAT YOU ARE PAID FOR  (worst outcomes for a seller)")
    print("  " + "-" * 74)
    for i, row in d.nsmallest(6, "prem").iterrows():
        print(f"  {i.date()}   VIX {row.vix:>5.2f}  ->  realised {row.fwd:>6.2f}"
              f"   premium {row.prem:>+7.2f}")

    avg, worst = d.prem.mean(), d.prem.min()
    print(f"\n  The average premium is {avg:+.2f} points; the worst single observation is"
          f" {worst:+.2f},")
    print(f"  about {abs(worst/avg):.0f}x the average, and it arrived in under three weeks.")
    print("  That ratio is the entire trade: frequent small wins against rare")
    print("  enormous losses. Defined-risk structures cap the loss at the width;")
    print("  naked short options do not cap it at all.")


if __name__ == "__main__":
    main()
