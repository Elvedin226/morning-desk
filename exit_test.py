"""Does swapping RSI-2's exit rule improve it?

Research across ~31 sources surfaced one exit rule appearing in NINE separate
published strategies from five independent authors: exit when today's close
exceeds YESTERDAY'S HIGH. That convergence is worth something on its own - it is
the most reused component in the whole survey.

The RSI-2 implementation here uses Connors' original exit instead: close above
the 5-day average. Swapping one line is close to a free experiment, and the two
rules are meaningfully different in character:

    close > SMA5            a level the price has to climb back to, so it can
                            wait several days and give profit back on the way
    close > prior high      a single strong day, so it exits faster and more
                            often takes the first pop

Faster exits cut both ways: less give-back, but also less time for a winner to
develop, and MORE trades - which matters because every strategy tested in this
project that died, died on transaction costs.

Tested on the broad universe, never a hand-picked ticker, with the same
permutation null and cost sweep as everything else here.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import backtest
import strategy
import validate
import watchlist
from strategy import Strategy, _hold, _rsi

START = "2011-01-01"     # post-publication only


class RSI2PriorHigh(Strategy):
    """Connors RSI-2 entry, but exit on a close above yesterday's high."""

    name = "rsi2_priorhigh"
    defaults = {"rsi_period": 2, "entry": 5, "trend_ma": 200}
    grid = {"rsi_period": [2, 3], "entry": [5, 10, 15], "trend_ma": [100, 200]}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        rsi = _rsi(close, self.params["rsi_period"])
        trend = close.rolling(self.params["trend_ma"]).mean()
        uptrend = close > trend
        enter = (rsi < self.params["entry"]) & uptrend
        # Yesterday's high. Shifting here is correct and is NOT the execution
        # delay - it is part of the rule ("close above the PRIOR high"). The
        # backtest applies the t+1 delay separately.
        exit_ = (close > df["high"].shift(1)) | (~uptrend)
        return _hold(enter, exit_)


strategy.STRATEGIES[RSI2PriorHigh.name] = RSI2PriorHigh


def bars(t: str) -> pd.DataFrame:
    raw = yf.download(t, start=START, auto_adjust=True, progress=False)
    if raw.empty:
        return raw
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    return df.dropna()


def main() -> None:
    names = sorted(watchlist.SECTORS)[:45] + ["SPY", "QQQ", "IWM"]
    print(f"\n  EXIT RULE COMPARISON on {len(names)} names, {START} to now, 5bp/side")
    print("  " + "=" * 78)
    print(f"  {'':<22}{'CAGR':>9}{'Sharpe':>9}{'trades':>9}{'win%':>7}"
          f"{'maxDD':>8}{'in mkt':>8}")
    print("  " + "-" * 78)

    rows = {"connors_rsi2": [], "rsi2_priorhigh": [], "buy_and_hold": []}
    for t in names:
        df = bars(t)
        if df.empty or len(df) < 600:
            continue
        for nm in rows:
            r = backtest.run(df, strategy.build(nm), fee_pct=0.0, slippage_pct=0.0005)
            rows[nm].append(r.metrics)

    def agg(nm, label):
        m = rows[nm]
        f = lambda k: np.nanmedian([x[k] for x in m])
        print(f"  {label:<22}{f('cagr')*100:>8.2f}%{f('sharpe'):>9.2f}"
              f"{f('num_trades'):>9.0f}{f('win_rate')*100:>6.0f}%"
              f"{f('max_drawdown')*100:>7.1f}%{f('exposure')*100:>7.0f}%")

    agg("connors_rsi2", "RSI-2, exit SMA5")
    agg("rsi2_priorhigh", "RSI-2, exit prior high")
    agg("buy_and_hold", "buy & hold")

    beat = np.mean([a["cagr"] > b["cagr"] for a, b in
                    zip(rows["rsi2_priorhigh"], rows["connors_rsi2"])])
    print(f"\n  prior-high exit beat SMA5 exit in {beat*100:.0f}% of names")

    # Costs decide this. The faster exit trades more, so it pays the toll more
    # often - that is the whole risk of the swap.
    print("\n  COST SENSITIVITY on SPY")
    print("  " + "=" * 78)
    spy = bars("SPY")
    print(f"  {'slippage':>10}{'SMA5 exit':>14}{'prior-high exit':>18}")
    for slip in (0.0, 0.0005, 0.0010, 0.0020, 0.0050):
        a = backtest.run(spy, strategy.build("connors_rsi2"), fee_pct=0, slippage_pct=slip)
        b = backtest.run(spy, strategy.build("rsi2_priorhigh"), fee_pct=0, slippage_pct=slip)
        print(f"  {slip*10000:>8.0f}bp{a.metrics['cagr']*100:>13.2f}%"
              f"{b.metrics['cagr']*100:>17.2f}%")

    print("\n  PERMUTATION NULL on SPY  (does it beat its own shuffled prices?)")
    print("  " + "=" * 78)
    for nm, label in (("connors_rsi2", "SMA5 exit"), ("rsi2_priorhigh", "prior-high exit")):
        wf = validate.walk_forward(spy, nm, fee=0.0, slippage=0.0005)
        real = wf.oos_metrics["sharpe"]
        null = validate.permutation_null(spy, nm, runs=40, fee=0.0, slippage=0.0005)
        p = float(np.mean(null >= real))
        print(f"  {label:<18} real OOS Sharpe {real:>5.2f}   shuffled 95th "
              f"{np.percentile(null,95):>5.2f}   p = {p:.2f}")


if __name__ == "__main__":
    main()
