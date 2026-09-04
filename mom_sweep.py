"""Parameter sweep, cost ladder and per-year breakdown for the four single-name
momentum/breakout rules, run across the whole declared universe.

Aggregation: every rule is run per ticker, then the daily NET return series are
averaged across tickers present on that date -- an equal-weight, daily-rebalanced
portfolio of the rule applied to all 92 names. Cash days contribute 0, so a rule
that is flat 90% of the time shows the low return it actually earns rather than a
per-trade average that quietly ignores the idle capital. The buy-and-hold
benchmark is the identical construction with the position pinned to 1.0, so both
sides see the same names, same dates, same costs.
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
import mom_strategies as MS

TF = mom_data.TF


def portfolio_returns(panel, name, params, slip, fee=mom_data.FEE):
    """Equal-weight daily-rebalanced net returns of `name` applied to every ticker."""
    cols = {}
    for t, df in panel.items():
        st = strategy.build(name, **params)
        r = backtest.run(df, st, timeframe=TF, fee_pct=fee, slippage_pct=slip)
        cols[t] = r.returns
    return pd.DataFrame(cols).mean(axis=1).dropna()


def per_ticker(panel, name, params, slip, fee=mom_data.FEE):
    """Per-ticker metrics plus the matched buy-and-hold on the same window."""
    rows = []
    for t, df in panel.items():
        st = strategy.build(name, **params)
        m = backtest.run(df, st, timeframe=TF, fee_pct=fee, slippage_pct=slip).metrics
        bh = backtest.run(df, strategy.build("buy_and_hold"), timeframe=TF,
                          fee_pct=fee, slippage_pct=slip).metrics
        rows.append({"ticker": t, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "exposure": m["exposure"], "trades": m["num_trades"],
                     "maxdd": m["max_drawdown"], "win": m["win_rate"],
                     "avg_trade": m["avg_trade"],
                     "bh_cagr": bh["cagr"], "bh_sharpe": bh["sharpe"]})
    return pd.DataFrame(rows)


def by_year(returns: pd.Series) -> pd.Series:
    return returns.groupby(returns.index.year).apply(lambda r: (1 + r).prod() - 1)


def main():
    panel = mom_data.load()
    print(f"\nUNIVERSE: {len(panel)} tickers (watchlist.SECTORS + SPY/QQQ/IWM), "
          f"{mom_data.START} to present. Fixed in advance; no name chosen by outcome.")

    bh_port = portfolio_returns(panel, "buy_and_hold", {}, 0.0)
    bh_year = by_year(bh_port)
    bh_m = backtest.core_metrics(bh_port, TF)
    print(f"BENCHMARK equal-weight buy & hold: CAGR {bh_m['cagr']*100:.2f}%  "
          f"Sharpe {bh_m['sharpe']:.2f}  maxDD {bh_m['max_drawdown']*100:.1f}%  exposure 100%")

    store = {}
    for name in MS.MOM_STRATEGIES:
        combos = validate.param_combos(name)
        print(f"\n\n{'='*100}\n  {name.upper()}   -- parameter sweep at 5bp/side, "
              f"equal-weight across all {len(panel)} names\n{'='*100}")
        print(f"  {'params':<28}{'CAGR':>9}{'B&H':>9}{'Sharpe':>8}{'maxDD':>8}"
              f"{'expos':>8}{'trades/nm':>10}{'win':>7}{'beat B&H':>10}")
        print("  " + "-" * 96)
        best, best_sh = None, -np.inf
        for c in combos:
            port = portfolio_returns(panel, name, c, 0.0005)
            pm = backtest.core_metrics(port, TF)
            pt = per_ticker(panel, name, c, 0.0005)
            label = ", ".join(f"{k}={v}" for k, v in c.items())
            beat = float((pt["cagr"] > pt["bh_cagr"]).mean())
            print(f"  {label:<28}{pm['cagr']*100:>8.2f}%{bh_m['cagr']*100:>8.2f}%"
                  f"{pm['sharpe']:>8.2f}{pm['max_drawdown']*100:>7.1f}%"
                  f"{pt['exposure'].median()*100:>7.1f}%{pt['trades'].median():>10.0f}"
                  f"{pt['win'].median()*100:>6.0f}%{beat*100:>9.0f}%")
            if pm["sharpe"] > best_sh:
                best, best_sh = c, pm["sharpe"]
        store[name] = best
        print(f"\n  best-by-Sharpe params (SELECTED IN-SAMPLE -- upward biased): {best}")

        print(f"\n  COST LADDER on those params (equal-weight portfolio)")
        print(f"  {'slip bp/side':<16}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>8}{'vs B&H CAGR':>13}")
        print("  " + "-" * 56)
        for slip in mom_data.SLIP_LEVELS:
            port = portfolio_returns(panel, name, best, slip)
            pm = backtest.core_metrics(port, TF)
            print(f"  {slip*10000:<16.0f}{pm['cagr']*100:>8.2f}%{pm['sharpe']:>8.2f}"
                  f"{pm['max_drawdown']*100:>7.1f}%{(pm['cagr']-bh_m['cagr'])*100:>12.2f}%")

        port5 = portfolio_returns(panel, name, best, 0.0005)
        yr = by_year(port5)
        print(f"\n  BY YEAR at 5bp  (does one period carry the whole result?)")
        print(f"  {'year':>6}{'strategy':>11}{'buy&hold':>11}{'diff':>10}")
        print("  " + "-" * 38)
        wins = 0
        for y in yr.index:
            d = yr[y] - bh_year.get(y, np.nan)
            wins += d > 0
            print(f"  {y:>6}{yr[y]*100:>10.2f}%{bh_year.get(y, np.nan)*100:>10.2f}%{d*100:>9.2f}%")
        print("  " + "-" * 38)
        print(f"  beat buy & hold in {wins} of {len(yr)} years")
        store[name] = {"params": best, "year": yr}

    return store


if __name__ == "__main__":
    main()
