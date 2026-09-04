"""Attack the cross-sectional momentum result.

Top-5 12-1 momentum on this universe prints ~43% CAGR against a 19% equal-weight
benchmark. The published cross-sectional momentum premium is ~4-8%/yr, and it
comes with periodic crashes. A 24-point excess is not a discovery, it is a smell.
This file tries to kill it four ways:

  1. WHAT DOES IT HOLD?  If the winners are names that IPO'd mid-sample and
     ripped (PLTR, ARM, IONQ, ASTS, RKLB, SMCI...), the result is the universe,
     not the rule. Those names are in watchlist.SECTORS in 2026 BECAUSE they
     went up. A ranker that concentrates into 5 names finds them; equal-weight
     across 92 dilutes them. That is selection bias with extra steps.

  2. 2007-VINTAGE SUBSET.  Re-run using only names already trading in 2007. Still
     survivorship-biased, but it removes the "listed later, then went vertical"
     cohort entirely.

  3. SEASONING FILTER.  Require N years of listed history before a name is
     eligible. A freshly-listed rocket cannot be picked.

  4. RANK MONOTONICITY + TIME SPLIT.  A real risk premium is monotone in rank and
     shows up in both halves of the sample. A bias artefact is concentrated in
     the extreme bucket and in the recent half.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest
import strategy
import mom_data
import mom_dualmom as D

TF = mom_data.TF
SLIP = 0.0005


def ew_bh(panel, start, end=None):
    r = pd.DataFrame({t: backtest.run(d, strategy.build("buy_and_hold"), timeframe=TF,
                                      fee_pct=0.0, slippage_pct=0.0).returns
                      for t, d in panel.items()}).mean(axis=1).dropna()
    r = r.loc[start:] if end is None else r.loc[start:end]
    return r


def rank_slice(close, lo, hi, seasoning=0, first_bar=None):
    """Hold ranks [lo, hi) by 12-1 momentum. seasoning = required years listed."""
    mom = close.shift(21) / close.shift(252) - 1
    rebals = D.rebalance_dates(close.index)
    held = pd.DataFrame(False, index=close.index, columns=close.columns)
    col = {c: i for i, c in enumerate(close.columns)}
    pos = {d: i for i, d in enumerate(close.index)}
    for i, d in enumerate(rebals):
        row = mom.loc[d].dropna()
        row = row[close.loc[d, row.index].notna()]
        if seasoning and first_bar is not None:
            row = row[[t for t in row.index
                       if (d - first_bar[t]).days >= seasoning * 365]]
        if len(row) < hi:
            continue
        pick = list(row.sort_values(ascending=False).index[lo:hi])
        start = pos[d]
        end = pos[rebals[i + 1]] if i + 1 < len(rebals) else len(close.index)
        held.iloc[start:end, [col[t] for t in pick]] = True
    return held


def score(panel, held, k, start, end=None):
    p = D.portfolio(panel, held, k, SLIP)
    p = p.loc[start:] if end is None else p.loc[start:end]
    return backtest.core_metrics(p, TF), p


def main():
    panel = mom_data.load()
    close = D.close_panel(panel).loc["2008-01-01":]
    first_bar = {t: d.index[0] for t, d in panel.items()}
    K = 5
    START = close.index[0]

    bh = ew_bh(panel, START)
    bhm = backtest.core_metrics(bh, TF)
    print("\nBASELINE  equal-weight universe B&H  CAGR %.2f%%  Sharpe %.2f"
          % (bhm["cagr"] * 100, bhm["sharpe"]))

    held = D.selection(close, K, False)
    m, port = score(panel, held, K, START)
    print("BASELINE  top-5 12-1 momentum        CAGR %.2f%%  Sharpe %.2f"
          % (m["cagr"] * 100, m["sharpe"]))

    # ---- 1. what does it actually hold? -----------------------------------
    print("\n\n1. HOLDINGS -- most-held names, and when each first listed")
    print("   (a name first listed AFTER 2008 was not knowably investable then;")
    print("    it is in the 2026 watchlist because it worked out)")
    print("   " + "-" * 68)
    days = held.sum().sort_values(ascending=False)
    days = days[days > 0]
    print("   %-8s%10s%14s%10s" % ("ticker", "days held", "first bar", "share"))
    tot = float(days.sum())
    late = 0.0
    for t in days.index[:18]:
        share = days[t] / tot
        fb = first_bar[t]
        if fb > pd.Timestamp("2008-01-01"):
            late += share
        print("   %-8s%10d%14s%9.1f%%%s"
              % (t, days[t], fb.date(), share * 100,
                 "   <- listed mid-sample" if fb > pd.Timestamp("2008-01-01") else ""))
    all_late = sum(days[t] / tot for t in days.index
                   if first_bar[t] > pd.Timestamp("2008-01-01"))
    print("   " + "-" * 68)
    print("   %d distinct names ever held, of %d in the universe" % (len(days), len(days.index) and close.shape[1]))
    print("   share of all holding-days spent in names that listed after 2008: %.1f%%"
          % (all_late * 100))

    # ---- 2. 2007-vintage subset -------------------------------------------
    print("\n\n2. 2007-VINTAGE SUBSET -- only names already trading at the start")
    old = {t: d for t, d in panel.items() if first_bar[t] <= pd.Timestamp("2007-06-30")}
    oc = D.close_panel(old).loc["2008-01-01":]
    ob = ew_bh(old, START)
    obm = backtest.core_metrics(ob, TF)
    oh = D.selection(oc, K, False)
    om, _ = score(old, oh, K, START)
    print("   universe %d names" % len(old))
    print("   %-24s%10s%9s" % ("", "CAGR", "Sharpe"))
    print("   %-24s%9.2f%%%9.2f" % ("equal-weight B&H", obm["cagr"] * 100, obm["sharpe"]))
    print("   %-24s%9.2f%%%9.2f" % ("top-5 momentum", om["cagr"] * 100, om["sharpe"]))
    print("   %-24s%9.2f%%" % ("excess", (om["cagr"] - obm["cagr"]) * 100))
    print("   vs excess on the full 92-name universe: %.2f%%"
          % ((m["cagr"] - bhm["cagr"]) * 100))

    # ---- 3. seasoning ------------------------------------------------------
    print("\n\n3. SEASONING FILTER -- name must have been listed N years to be eligible")
    print("   %-14s%10s%9s%12s" % ("min history", "CAGR", "Sharpe", "vs EW B&H"))
    print("   " + "-" * 46)
    for yrs in (0, 1, 2, 3, 5):
        h = rank_slice(close, 0, K, seasoning=yrs, first_bar=first_bar)
        mm, _ = score(panel, h, K, START)
        print("   %-14s%9.2f%%%9.2f%11.2f%%"
              % ("%d yr" % yrs if yrs else "none", mm["cagr"] * 100, mm["sharpe"],
                 (mm["cagr"] - bhm["cagr"]) * 100))

    # ---- 4. rank monotonicity ---------------------------------------------
    print("\n\n4. RANK BUCKETS -- is the effect monotone in rank, or only at the top?")
    print("   %-14s%10s%9s%12s" % ("ranks", "CAGR", "Sharpe", "vs EW B&H"))
    print("   " + "-" * 46)
    n = close.shape[1]
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 60), (n - 10, n - 5), (n - 5, n)):
        h = rank_slice(close, lo, hi)
        mm, _ = score(panel, h, hi - lo, START)
        print("   %-14s%9.2f%%%9.2f%11.2f%%"
              % ("%d-%d" % (lo + 1, hi), mm["cagr"] * 100, mm["sharpe"],
                 (mm["cagr"] - bhm["cagr"]) * 100))

    # ---- 5. time split -----------------------------------------------------
    print("\n\n5. TIME SPLIT -- persistent, or all in the recent half?")
    print("   %-16s%12s%12s%12s%12s" % ("window", "mom CAGR", "B&H CAGR", "excess", "mom Sharpe"))
    print("   " + "-" * 66)
    for lab, a, b in (("2008-2016", "2008-01-01", "2016-12-31"),
                      ("2017-2026", "2017-01-01", None),
                      ("2008-2012", "2008-01-01", "2012-12-31"),
                      ("2013-2017", "2013-01-01", "2017-12-31"),
                      ("2018-2022", "2018-01-01", "2022-12-31"),
                      ("2023-2026", "2023-01-01", None)):
        mm, _ = score(panel, held, K, a, b)
        bb = backtest.core_metrics(ew_bh(panel, a, b), TF)
        print("   %-16s%11.2f%%%11.2f%%%11.2f%%%12.2f"
              % (lab, mm["cagr"] * 100, bb["cagr"] * 100,
                 (mm["cagr"] - bb["cagr"]) * 100, mm["sharpe"]))

    # ---- 6. what carried the biggest years? -------------------------------
    print("\n\n6. BIGGEST YEARS -- which names were held?")
    yr = D.by_year(port)
    for y in yr.sort_values(ascending=False).index[:3]:
        hy = held.loc[str(y)]
        names = hy.sum()
        names = names[names > 0].sort_values(ascending=False)
        print("   %d  (%+.1f%%): %s" % (y, yr[y] * 100,
              ", ".join("%s(%d)" % (t, names[t]) for t in names.index[:8])))


if __name__ == "__main__":
    main()
