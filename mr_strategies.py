"""Mean-reversion / oscillator strategies, registered into strategy.STRATEGIES.

Registered locally so strategy.py is untouched. Import this module before
calling strategy.build() or validate.walk_forward() with any name below.

Every class follows the project convention: signals() may read bar t's own
close, and backtest.run applies the t+1 execution delay itself. Nothing here
shifts internally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import backtest
import strategy
from strategy import Strategy, _hold, _rsi

# backtest.PERIODS_PER_YEAR is calibrated for crypto: "1d" -> 365 bars a year.
# US equities trade ~252. Left at 365 it overstates CAGR (years = bars/365
# undercounts elapsed years) and overstates Sharpe by sqrt(365/252) = 1.20x.
# Registering a key here fixes the annualization without touching backtest.py.
# NOTE: earlier results in this project (connors_test.py included) used 365, so
# their Sharpes are ~1.2x the ones below. The Connors baseline is re-run here on
# the 252 scale so everything in this study is comparable.
TF = "1d_equity"
backtest.PERIODS_PER_YEAR[TF] = 252


class CumRSI(Strategy):
    """Connors' Cumulative RSI: sum RSI(2) over N days, buy the low tail.

    Published variant of RSI-2 that asks for persistent, not instantaneous,
    oversold. Same 200sma trend filter and same 5sma exit as the RSI-2 baseline
    in strategy.py, so the only thing being changed is the entry statistic —
    otherwise the comparison to the baseline measures the exit rule too.
    """

    name = "mr_cumrsi"
    defaults = {"rsi_period": 2, "days": 2, "entry": 35, "exit_ma": 5, "trend_ma": 200}
    grid = {"days": [2, 3], "entry": [35, 45, 60]}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        cum = _rsi(close, self.params["rsi_period"]).rolling(self.params["days"]).sum()
        trend = close.rolling(self.params["trend_ma"]).mean()
        fast = close.rolling(self.params["exit_ma"]).mean()
        uptrend = close > trend
        return _hold((cum < self.params["entry"]) & uptrend,
                     (close > fast) | (~uptrend))


class CumRSINoTrend(CumRSI):
    """Same entry, no 200sma filter — reported so the filter's effect is visible."""

    name = "mr_cumrsi_nt"

    def signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        cum = _rsi(close, self.params["rsi_period"]).rolling(self.params["days"]).sum()
        fast = close.rolling(self.params["exit_ma"]).mean()
        return _hold(cum < self.params["entry"], close > fast)


class VWAPReversion(Strategy):
    """Daily-bar VWAP proxy: rolling volume-weighted typical price.

    True VWAP is intraday. This is the standard daily stand-in: typical price
    (h+l+c)/3 weighted by volume over a rolling window. Entry is K rolling
    standard deviations of (close - vwap) below the vwap; exit is a close back
    at or above the vwap. Stated as a proxy because it is one.
    """

    name = "mr_vwap"
    defaults = {"window": 20, "k": 1.5}
    grid = {"window": [10, 20, 50], "k": [1.0, 1.5, 2.0]}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        n = self.params["window"]
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        pv = (typical * df["volume"]).rolling(n).sum()
        vol = df["volume"].rolling(n).sum()
        vwap = pv / vol.replace(0.0, np.nan)
        dev = df["close"] - vwap
        sd = dev.rolling(n).std()
        lower = vwap - self.params["k"] * sd
        return _hold(df["close"] < lower, df["close"] >= vwap)


class IBS(Strategy):
    """Internal Bar Strength: (close - low) / (high - low).

    Close near the day's low -> buy; close near the day's high -> sell. Very
    well documented on index ETFs. The open question is single names, where the
    high-low range is wider and the close is less mechanically pinned.

    Zero-range bars (high == low) give 0/0; those bars are left as NaN, which
    _hold reads as "no trigger".
    """

    name = "mr_ibs"
    defaults = {"entry": 0.2, "exit": 0.8}
    grid = {"entry": [0.1, 0.2, 0.3], "exit": [0.7, 0.8, 0.9]}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        rng = (df["high"] - df["low"]).replace(0.0, np.nan)
        ibs = (df["close"] - df["low"]) / rng
        return _hold(ibs < self.params["entry"], ibs > self.params["exit"])


class NDownDays(Strategy):
    """K consecutive lower closes inside a 200sma uptrend; exit on the first up close."""

    name = "mr_ndown"
    defaults = {"k": 3, "trend_ma": 200}
    grid = {"k": [2, 3, 4]}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        down = close < close.shift(1)
        k = self.params["k"]
        streak = down.rolling(k).sum() == k
        uptrend = close > close.rolling(self.params["trend_ma"]).mean()
        up = close > close.shift(1)
        return _hold(streak & uptrend, up | (~uptrend))


for _cls in (CumRSI, CumRSINoTrend, VWAPReversion, IBS, NDownDays):
    strategy.STRATEGIES[_cls.name] = _cls


# ---------------------------------------------------------------------------
# A permutation null that preserves the whole bar, not just its body.
#
# validate.shuffle_bars rebuilds high/low as max/min(open, close), which throws
# the wicks away. For IBS and VWAP that is not a null, it is a different
# strategy: with no wicks, IBS collapses to 1.0 on every up bar and 0.0 on every
# down bar. So the shuffle here carries each bar's full (gap, high, low, close)
# geometry as one unit and only shuffles the ORDER of bars. Same destruction of
# across-bar structure, no destruction of within-bar structure.
# ---------------------------------------------------------------------------

def shuffle_bars_full(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    open_ = df["open"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    volume = df["volume"].to_numpy(float)

    gap = open_[1:] / close[:-1]          # previous close -> this open
    r_hi = high[1:] / open_[1:]           # bar geometry, all relative to its own open
    r_lo = low[1:] / open_[1:]
    r_cl = close[1:] / open_[1:]

    order = rng.permutation(len(gap))
    gap, r_hi, r_lo, r_cl = gap[order], r_hi[order], r_lo[order], r_cl[order]

    n = len(gap) + 1
    o = np.empty(n); h = np.empty(n); lo_ = np.empty(n); c = np.empty(n)
    o[0], h[0], lo_[0], c[0] = open_[0], high[0], low[0], close[0]
    for i in range(1, n):
        o[i] = c[i - 1] * gap[i - 1]
        c[i] = o[i] * r_cl[i - 1]
        h[i] = o[i] * r_hi[i - 1]
        lo_[i] = o[i] * r_lo[i - 1]

    vol = np.concatenate([[volume[0]], volume[1:][order]])
    return pd.DataFrame(
        {"open": o, "high": np.maximum(h, np.maximum(o, c)),
         "low": np.minimum(lo_, np.minimum(o, c)),
         "close": c, "volume": vol},
        index=df.index,
    )


def permutation_null_full(df: pd.DataFrame, name: str, runs: int = 40,
                          seed: int = 0, **kwargs) -> np.ndarray:
    """validate.permutation_null, but with the bar-preserving shuffle above."""
    import validate

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(runs):
        wf = validate.walk_forward(shuffle_bars_full(df, rng), name, **kwargs)
        s = wf.oos_metrics["sharpe"]
        if not np.isnan(s):
            out.append(s)
    return np.array(out)
