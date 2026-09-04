"""The clean-universe test for cross-sectional momentum.

The 92-name stock universe is survivorship-selected: it is the 2026 watchlist, so
every candidate is known to have survived and grown. A ranker that concentrates
into 5 of those names is handed the answer. The diagnostics show exactly that --
the excess return sits entirely in ranks 1-5, is not monotone in rank, and grows
from +5% in 2008-2012 to +58% in 2023-2026.

So re-run the identical rule on a universe with NO survivorship bias:

    the nine original SPDR sector ETFs -- XLB XLE XLF XLI XLK XLP XLU XLV XLY

All nine launched in December 1998 and all nine still trade. Nothing was dropped,
nothing was added, and membership was not chosen with hindsight -- the set is
complete and was knowable in advance on every date tested. Their constituents
churn internally, which is the point: the ETF absorbs the delistings that the
stock universe silently omits.

If 12-1 cross-sectional momentum is a real effect, it should show up here, where
it cannot be manufactured by picking the era's winners out of a survivor list. If
it vanishes, the 46% CAGR on stocks was the universe, not the rule.

Same engine, same t+1 open fills, same cost ladder, same monthly rebalance.
"""

from __future__ import annotations

import os
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import backtest
import strategy
import mom_data
import mom_dualmom as D

TF = mom_data.TF
SLIP = 0.0005
RUNS = 300

SECTORS9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
CACHE = "data_cache/mom_sector_panel.pkl"
START = "1999-01-01"


def load():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    raw = yf.download(SECTORS9 + ["SPY"], start=START, auto_adjust=True,
                      progress=False, group_by="ticker")
    panel = {}
    for t in SECTORS9 + ["SPY"]:
        df = raw[t].rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        panel[t] = df.dropna()
    with open(CACHE, "wb") as fh:
        pickle.dump(panel, fh)
    return panel


def main():
    full = load()
    panel = {t: d for t, d in full.items() if t in SECTORS9}
    close = D.close_panel(panel).loc["2000-01-01":]
    start = close.index[0]
    print("\nCLEAN-UNIVERSE TEST: 9 original SPDR sector ETFs, %s to %s"
          % (start.date(), close.index[-1].date()))
    print("No survivorship bias -- all nine launched Dec 1998, all nine still trade,")
    print("membership was knowable in advance on every date tested.\n")

    bh = pd.DataFrame({t: backtest.run(d, strategy.build("buy_and_hold"), timeframe=TF,
                                       fee_pct=0.0, slippage_pct=0.0).returns
                       for t, d in panel.items()}).mean(axis=1).loc[start:].dropna()
    bhm = backtest.core_metrics(bh, TF)
    spy = backtest.run(full["SPY"].loc[start:], strategy.build("buy_and_hold"),
                       timeframe=TF).metrics
    print("BENCHMARK  equal-weight 9 sectors  CAGR %.2f%%  Sharpe %.2f  maxDD %.1f%%"
          % (bhm["cagr"] * 100, bhm["sharpe"], bhm["max_drawdown"] * 100))
    print("           SPY buy & hold          CAGR %.2f%%  Sharpe %.2f  maxDD %.1f%%"
          % (spy["cagr"] * 100, spy["sharpe"], spy["max_drawdown"] * 100))

    print("\n  %-26s%9s%8s%8s%12s" % ("variant", "CAGR", "Sharpe", "maxDD", "vs EW B&H"))
    print("  " + "-" * 64)
    results = {}
    for k in (1, 2, 3, 5):
        for abs_f in (False, True):
            held = D.selection(close, k, abs_f)
            m = backtest.core_metrics(D.portfolio(panel, held, k, SLIP).loc[start:], TF)
            results[(k, abs_f)] = m
            lab = "top%d %s" % (k, "dual (abs filter)" if abs_f else "relative only")
            print("  %-26s%8.2f%%%8.2f%7.1f%%%11.2f%%"
                  % (lab, m["cagr"] * 100, m["sharpe"], m["max_drawdown"] * 100,
                     (m["cagr"] - bhm["cagr"]) * 100))

    best = max(results, key=lambda kk: results[kk]["sharpe"])
    k, abs_f = best
    print("\n  best-by-Sharpe (in-sample, biased): top%d %s"
          % (k, "dual" if abs_f else "relative"))

    held = D.selection(close, k, abs_f)
    print("\n  COST LADDER")
    print("  %-16s%9s%8s%12s" % ("slip bp/side", "CAGR", "Sharpe", "vs EW B&H"))
    print("  " + "-" * 46)
    for slip in mom_data.SLIP_LEVELS:
        m = backtest.core_metrics(D.portfolio(panel, held, k, slip).loc[start:], TF)
        print("  %-16.0f%8.2f%%%8.2f%11.2f%%"
              % (slip * 10000, m["cagr"] * 100, m["sharpe"],
                 (m["cagr"] - bhm["cagr"]) * 100))

    port = D.portfolio(panel, held, k, SLIP).loc[start:]
    yr, byr = D.by_year(port), D.by_year(bh)
    print("\n  BY YEAR at 5bp")
    print("  %6s%11s%11s%10s" % ("year", "strategy", "EW B&H", "diff"))
    print("  " + "-" * 38)
    wins = 0
    for y in yr.index:
        d = yr[y] - byr.get(y, np.nan)
        wins += d > 0
        print("  %6d%10.2f%%%10.2f%%%9.2f%%" % (y, yr[y] * 100, byr.get(y, np.nan) * 100, d * 100))
    print("  " + "-" * 38)
    print("  beat equal-weight buy & hold in %d of %d years" % (wins, len(yr)))

    # rank buckets: with 9 assets, top-3 / mid-3 / bottom-3
    print("\n  RANK BUCKETS (top-3 / mid-3 / bottom-3 of 9)")
    print("  %-14s%10s%9s%12s" % ("ranks", "CAGR", "Sharpe", "vs EW B&H"))
    print("  " + "-" * 46)
    import mom_dm_diag as G
    for lo, hi in ((0, 3), (3, 6), (6, 9)):
        h = G.rank_slice(close, lo, hi)
        m = backtest.core_metrics(D.portfolio(panel, h, hi - lo, SLIP).loc[start:], TF)
        print("  %-14s%9.2f%%%9.2f%11.2f%%"
              % ("%d-%d" % (lo + 1, hi), m["cagr"] * 100, m["sharpe"],
                 (m["cagr"] - bhm["cagr"]) * 100))

    # time split
    print("\n  TIME SPLIT")
    print("  %-16s%12s%12s%12s" % ("window", "mom CAGR", "B&H CAGR", "excess"))
    print("  " + "-" * 54)
    for lab, a, b in (("2000-2008", "2000-01-01", "2008-12-31"),
                      ("2009-2017", "2009-01-01", "2017-12-31"),
                      ("2018-2026", "2018-01-01", None)):
        p = D.portfolio(panel, held, k, SLIP)
        p = p.loc[a:b] if b else p.loc[a:]
        bb = bh.loc[a:b] if b else bh.loc[a:]
        mm = backtest.core_metrics(p, TF)
        bm = backtest.core_metrics(bb, TF)
        print("  %-16s%11.2f%%%11.2f%%%11.2f%%"
              % (lab, mm["cagr"] * 100, bm["cagr"] * 100, (mm["cagr"] - bm["cagr"]) * 100))

    # random-K null
    real_sh = results[best]["sharpe"]
    real_cagr = results[best]["cagr"]
    print("\n  RANDOM-K NULL (%d draws, real prices, same dates)" % RUNS)
    rng = np.random.default_rng(0)
    sh, cg = [], []
    for _ in range(RUNS):
        h = D.selection(close, k, abs_f, picker=D.random_picker, rng=rng)
        m = backtest.core_metrics(D.portfolio(panel, h, k, SLIP).loc[start:], TF)
        sh.append(m["sharpe"])
        cg.append(m["cagr"])
    sh, cg = np.array(sh), np.array(cg)
    print("     real   Sharpe %6.2f   CAGR %6.2f%%" % (real_sh, real_cagr * 100))
    print("     random Sharpe med %5.2f  95th %5.2f   -> p = %.3f"
          % (np.median(sh), np.percentile(sh, 95), float(np.mean(sh >= real_sh))))
    print("     random CAGR   med %5.2f%% 95th %5.2f%%   -> p = %.3f"
          % (np.median(cg) * 100, np.percentile(cg, 95) * 100,
             float(np.mean(cg >= real_cagr))))


if __name__ == "__main__":
    main()
