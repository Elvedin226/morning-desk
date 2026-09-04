"""Cross-sectional / dual momentum: rank the universe by 12-1 momentum, hold the
top K, rebalance monthly.

This one cannot use the Strategy interface, because the interface is per-ticker
and this rule is a comparison BETWEEN tickers. So the selection is computed on
the panel, then handed back to the engine as a precomputed position series per
ticker -- which means the t+1 fill delay, the open-price fills and the cost
model are still the engine's, not a second implementation that could disagree.

Weighting: 1/K of capital per slot. If the absolute-momentum filter empties some
slots, that capital sits in cash earning 0. Un-filled slots are NOT redistributed
-- that would quietly turn a defensive rule into a concentrated one.

12-1 momentum = close.shift(21)/close.shift(252) - 1: a year of return excluding
the most recent month, the standard construction, which sidesteps short-term
reversal.

TWO NULLS:
  shuffled  -- every name's bars permuted. Preserves each name's drift, so this
               null ALREADY CONTAINS most of the "hold names that go up" effect.
               watchlist.py flags exactly this (momentum scored p=0.57 against
               it). Reported, but it is the weaker test.
  random-K  -- keep the real price paths, pick K names at random each month.
               This is the null that isolates the RANKING: does sorting on 12-1
               momentum beat throwing darts at the same 92 names on the same
               dates? For a selection rule, this is the honest question.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest
import strategy
import validate
import mom_data

TF = mom_data.TF
SLIP = 0.0005
RUNS = 200          # random-K null is cheap, so run it properly
SHUF_RUNS = 40


class _Given(strategy.Strategy):
    """Wrap a precomputed position series so backtest.run still owns execution."""

    name = "given"
    defaults: dict = {}

    def __init__(self, pos):
        self.params = {}
        self._pos = pos

    def signals(self, df):
        return self._pos.reindex(df.index).fillna(0.0)

    def describe(self):
        return "given"


def close_panel(panel):
    return pd.DataFrame({t: d["close"] for t, d in panel.items()}).sort_index()


def rebalance_dates(index):
    """Last trading day of each month."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().values)


def selection(close, k, abs_filter, picker=None, rng=None):
    """Boolean (date x ticker): held on that date. Ranking uses data through t only."""
    mom = close.shift(21) / close.shift(252) - 1
    rebals = rebalance_dates(close.index)
    held = pd.DataFrame(False, index=close.index, columns=close.columns)

    for i, d in enumerate(rebals):
        row = mom.loc[d].dropna()
        row = row[close.loc[d, row.index].notna()]
        if len(row) < k:
            continue
        if picker is None:
            pick = list(row.sort_values(ascending=False).index[:k])
            if abs_filter:
                pick = [t for t in pick if row[t] > 0]
        else:
            pick = picker(row, k, rng)
        if not pick:
            continue
        end = rebals[i + 1] if i + 1 < len(rebals) else close.index[-1]
        held.loc[(held.index >= d) & (held.index <= end), pick] = True
    return held


def random_picker(row, k, rng):
    return list(rng.choice(row.index.to_numpy(), size=k, replace=False))


def portfolio(panel, held, k, slip=SLIP):
    """1/K per slot. Sum of per-name engine returns / K."""
    total = None
    for t, df in panel.items():
        if t not in held.columns:
            continue
        pos = held[t].astype("float64").reindex(df.index).fillna(0.0)
        if pos.sum() == 0:
            continue
        r = backtest.run(df, _Given(pos), timeframe=TF, fee_pct=0.0,
                         slippage_pct=slip).returns
        total = r if total is None else total.add(r, fill_value=0.0)
    return (total / k).dropna()


def by_year(r):
    return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)


def main():
    panel = mom_data.load()
    close = close_panel(panel)
    close = close.loc["2008-01-01":]
    print("\nDUAL / RELATIVE MOMENTUM   universe %d names, %s to %s"
          % (len(panel), close.index[0].date(), close.index[-1].date()))

    bh = pd.DataFrame({t: backtest.run(d, strategy.build("buy_and_hold"), timeframe=TF,
                                       fee_pct=0.0, slippage_pct=0.0).returns
                       for t, d in panel.items()}).mean(axis=1).loc[close.index[0]:].dropna()
    bhm = backtest.core_metrics(bh, TF)
    spy = backtest.run(panel["SPY"].loc[close.index[0]:], strategy.build("buy_and_hold"),
                       timeframe=TF).metrics
    print("BENCHMARK  equal-weight universe B&H  CAGR %.2f%%  Sharpe %.2f  maxDD %.1f%%"
          % (bhm["cagr"] * 100, bhm["sharpe"], bhm["max_drawdown"] * 100))
    print("           SPY buy & hold             CAGR %.2f%%  Sharpe %.2f  maxDD %.1f%%"
          % (spy["cagr"] * 100, spy["sharpe"], spy["max_drawdown"] * 100))

    results = {}
    print("\n  %-26s%9s%8s%8s%8s%11s" % ("variant", "CAGR", "Sharpe", "maxDD",
                                         "expos", "vs EW B&H"))
    print("  " + "-" * 72)
    for k in (3, 5, 10):
        for abs_f in (False, True):
            held = selection(close, k, abs_f)
            port = portfolio(panel, held, k)
            m = backtest.core_metrics(port, TF)
            exposure = float(held.sum(axis=1).mean() / k)
            lab = "top%d %s" % (k, "dual (abs filter)" if abs_f else "relative only")
            results[(k, abs_f)] = (port, m, exposure)
            print("  %-26s%8.2f%%%8.2f%7.1f%%%7.1f%%%10.2f%%"
                  % (lab, m["cagr"] * 100, m["sharpe"], m["max_drawdown"] * 100,
                     exposure * 100, (m["cagr"] - bhm["cagr"]) * 100))

    best_key = max(results, key=lambda kk: results[kk][1]["sharpe"])
    k, abs_f = best_key
    print("\n  best-by-Sharpe (in-sample, biased): top%d %s"
          % (k, "dual" if abs_f else "relative"))

    print("\n  COST LADDER")
    print("  %-16s%9s%8s%11s" % ("slip bp/side", "CAGR", "Sharpe", "vs EW B&H"))
    print("  " + "-" * 46)
    held = selection(close, k, abs_f)
    for slip in mom_data.SLIP_LEVELS:
        m = backtest.core_metrics(portfolio(panel, held, k, slip), TF)
        print("  %-16.0f%8.2f%%%8.2f%10.2f%%"
              % (slip * 10000, m["cagr"] * 100, m["sharpe"],
                 (m["cagr"] - bhm["cagr"]) * 100))

    port = results[best_key][0]
    yr, byr = by_year(port), by_year(bh)
    print("\n  BY YEAR at 5bp")
    print("  %6s%11s%11s%10s" % ("year", "strategy", "EW B&H", "diff"))
    print("  " + "-" * 38)
    wins = 0
    for y in yr.index:
        d = yr[y] - byr.get(y, np.nan)
        wins += d > 0
        print("  %6d%10.2f%%%10.2f%%%9.2f%%"
              % (y, yr[y] * 100, byr.get(y, np.nan) * 100, d * 100))
    print("  " + "-" * 38)
    print("  beat equal-weight buy & hold in %d of %d years" % (wins, len(yr)))

    real_sh = results[best_key][1]["sharpe"]
    real_cagr = results[best_key][1]["cagr"]
    print("\n\n  NULL 1  RANDOM-K SELECTION (%d draws, real prices, same dates)" % RUNS)
    print("  Isolates the ranking: is sorting on 12-1 momentum better than darts?")
    rng = np.random.default_rng(0)
    sh, cg = [], []
    for i in range(RUNS):
        h = selection(close, k, abs_f, picker=random_picker, rng=rng)
        m = backtest.core_metrics(portfolio(panel, h, k), TF)
        sh.append(m["sharpe"])
        cg.append(m["cagr"])
    sh, cg = np.array(sh), np.array(cg)
    print("     real   Sharpe %6.2f   CAGR %6.2f%%" % (real_sh, real_cagr * 100))
    print("     random Sharpe med %5.2f  95th %5.2f   -> p = %.3f"
          % (np.median(sh), np.percentile(sh, 95), float(np.mean(sh >= real_sh))))
    print("     random CAGR   med %5.2f%% 95th %5.2f%%   -> p = %.3f"
          % (np.median(cg) * 100, np.percentile(cg, 95) * 100,
             float(np.mean(cg >= real_cagr))))

    print("\n  NULL 2  SHUFFLED BARS (%d panels)  -- weaker test, see docstring" % SHUF_RUNS)
    rng = np.random.default_rng(1)
    sh2 = []
    for i in range(SHUF_RUNS):
        sp = {t: validate.shuffle_bars(d, rng) for t, d in panel.items()}
        sc = close_panel(sp).loc[close.index[0]:]
        h = selection(sc, k, abs_f)
        m = backtest.core_metrics(portfolio(sp, h, k), TF)
        sh2.append(m["sharpe"])
    sh2 = np.array([s for s in sh2 if np.isfinite(s)])
    print("     shuffled Sharpe med %5.2f  95th %5.2f   -> p = %.3f"
          % (np.median(sh2), np.percentile(sh2, 95), float(np.mean(sh2 >= real_sh))))


if __name__ == "__main__":
    main()
