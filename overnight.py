"""Overnight-drift study: does US equity return accrue outside the session?

This does NOT run through backtest.py, deliberately. That engine's unit is a
whole daily bar; this strategy is in at the close and out at the next open —
two fills a day, holding only the gap. Forcing it into the daily engine would
silently score the wrong return, so the split is computed directly here and
scored with backtest.core_metrics, which is the same math every other result
in this project uses.

    overnight return = open[t+1] / close[t] - 1     (held while the market is shut)
    intraday  return = close[t]  / open[t]  - 1     (held while it is open)

Costs: the overnight leg pays a full round trip every single day, so this is a
strategy whose viability is decided almost entirely by execution cost. That
sensitivity is the point of the report below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

import backtest as B

TRADING_DAYS = 252


def load(symbol: str, start: str = "2005-01-01") -> pd.DataFrame:
    # auto_adjust=True on purpose. Dividends land ENTIRELY on the overnight leg —
    # measured, not assumed: adjusting moves SPY overnight +1.95%/yr and intraday
    # +0.00%. The overnight trader holds at the close before the ex-date, so they
    # are the holder of record: they eat the ex-date gap down AND collect the cash.
    # Price-only returns therefore understate this strategy by the dividend yield.
    df = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
    ).dropna()


def legs(df: pd.DataFrame) -> pd.DataFrame:
    """Split each day into its overnight and intraday components."""
    return pd.DataFrame(
        {
            "overnight": df["open"] / df["close"].shift(1) - 1,
            "intraday": df["close"] / df["open"] - 1,
            "full_day": df["close"] / df["close"].shift(1) - 1,
        },
        index=df.index,
    ).dropna()


def net_of_costs(gross: pd.Series, round_trip_pct: float) -> pd.Series:
    """One full round trip per day — this leg is entered and exited daily."""
    return gross - round_trip_pct


def summarise(name: str, r: pd.Series) -> dict:
    m = B.core_metrics(r, timeframe="1d", initial_cash=1.0)
    years = len(r) / TRADING_DAYS
    # Is the mean daily return distinguishable from zero? With ~5,000 observations
    # a t-stat is the honest way to ask, rather than eyeballing a total return.
    t = r.mean() / (r.std() / np.sqrt(len(r))) if r.std() > 0 else np.nan
    return {
        "leg": name,
        "cagr": (1 + r).prod() ** (1 / years) - 1,
        "sharpe": m["sharpe"] * np.sqrt(TRADING_DAYS / 365),  # re-annualise on 252
        "max_dd": m["max_drawdown"],
        "t_stat": t,
    }


def report(symbol: str, start: str = "2005-01-01") -> None:
    df = load(symbol, start)
    parts = legs(df)
    print(f"\n{symbol}  {parts.index[0].date()} -> {parts.index[-1].date()}  ({len(parts)} sessions)")

    rows = [
        summarise("overnight (gross)", parts["overnight"]),
        summarise("intraday (gross)", parts["intraday"]),
        summarise("buy & hold", parts["full_day"]),
    ]
    print(f"\n  {'leg':<22}{'CAGR':>9}{'Sharpe':>9}{'max DD':>9}{'t-stat':>9}")
    for r in rows:
        print(f"  {r['leg']:<22}{r['cagr']*100:>8.2f}%{r['sharpe']:>9.2f}{r['max_dd']*100:>8.1f}%{r['t_stat']:>9.2f}")

    # The decisive question for a small account: how much cost can it absorb?
    print(f"\n  overnight leg, net of a daily round trip:")
    print(f"    {'round-trip cost':<22}{'CAGR':>9}{'vs buy&hold':>14}")
    bh = rows[2]["cagr"]
    for cost_bp in (0, 1, 2, 3, 5, 10):
        net = net_of_costs(parts["overnight"], cost_bp / 10_000)
        cagr = (1 + net).prod() ** (1 / (len(net) / TRADING_DAYS)) - 1
        print(f"    {cost_bp:>3} bp{'':<15}{cagr*100:>8.2f}%{(cagr-bh)*100:>13.2f}%")

    # Does it hold out of sample, or is it one era?
    print(f"\n  overnight leg (gross) by period:")
    by_year = parts["overnight"].groupby(parts.index.year).apply(lambda s: (1 + s).prod() - 1)
    half = len(parts) // 2
    for label, seg in [("first half", parts["overnight"].iloc[:half]),
                       ("second half", parts["overnight"].iloc[half:])]:
        cagr = (1 + seg).prod() ** (1 / (len(seg) / TRADING_DAYS)) - 1
        t = seg.mean() / (seg.std() / np.sqrt(len(seg)))
        print(f"    {label:<14}{cagr*100:>8.2f}%   t={t:.2f}")
    pos = (by_year > 0).sum()
    print(f"    positive years: {pos}/{len(by_year)}")


if __name__ == "__main__":
    for sym in ("SPY", "QQQ"):
        report(sym)
