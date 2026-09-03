"""Trading strategies.

A strategy turns a price DataFrame into a *target position* series:
    1.0 = hold the asset, 0.0 = hold cash.

Long/flat only — no shorting, no leverage, no position sizing. That keeps the
backtest honest about what it is measuring; shorting brings borrow costs and
margin rules that a v1 engine would only fake.

Signals may use bar t's own close. The backtest is responsible for delaying
execution to bar t+1 — see backtest.py. Do not shift inside a strategy, or the
delay gets applied twice.

To add a strategy: subclass Strategy, implement signals(), add it to STRATEGIES.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _hold(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Convert entry/exit triggers into a held position series.

    Enter on an entry bar, stay in until an exit bar, then stay out. If both fire
    on the same bar, exit wins. Bars before the indicator warms up are flat.
    """
    pos = pd.Series(np.nan, index=entry.index, dtype="float64")
    pos[entry.fillna(False)] = 1.0
    pos[exit_.fillna(False)] = 0.0
    return pos.ffill().fillna(0.0)


class Strategy:
    """Base class. Subclasses set `name` and implement `signals`."""

    name = "base"

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(
                f"{self.name}: unknown parameter(s) {sorted(unknown)}. "
                f"Valid: {sorted(self.defaults)}"
            )
        self.params = {**self.defaults, **params}

    defaults: dict = {}

    # Parameter values walk-forward validation searches over. Deliberately coarse —
    # a finer grid does not find a better strategy, it finds a better fit to the
    # training window, which is the thing validation exists to catch.
    grid: dict = {}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def describe(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({args})"


class MACross(Strategy):
    """Trend following: long while the fast moving average is above the slow one."""

    name = "ma_cross"
    defaults = {"fast": 20, "slow": 50}
    grid = {"fast": [5, 10, 20, 30], "slow": [40, 50, 100, 150]}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        fast = df["close"].rolling(self.params["fast"]).mean()
        slow = df["close"].rolling(self.params["slow"]).mean()
        # NaN during warmup compares False, which is the flat position we want.
        return (fast > slow).astype("float64")


class RSIReversion(Strategy):
    """Mean reversion: buy oversold, sell once momentum recovers."""

    name = "rsi"
    defaults = {"period": 14, "oversold": 30, "overbought": 55}
    grid = {"period": [2, 7, 14], "oversold": [20, 30, 40], "overbought": [50, 60, 70]}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = _rsi(df["close"], self.params["period"])
        return _hold(rsi < self.params["oversold"], rsi > self.params["overbought"])


class ConnorsRSI2(Strategy):
    """Larry Connors' 2-period RSI mean reversion, with the 200-day trend filter.

    Buy a short, sharp pullback INSIDE an uptrend; exit as soon as price recovers
    above its 5-day average. Holds are typically 1-3 days, which is why it is a
    plausible fit for a small account with a daily target - unlike a 21-day swing
    that ties capital up for a month.

    Rules as published:
        enter  RSI(2) < 5  and close > 200sma
        exit   close > 5sma, or close falls below the 200sma

    The published record is 75-85% win rates on indices from the mid-1990s to
    2010. That period is the concern, not the claim: the strategy was published
    in 2008-2011 and is now in every screener. McLean & Pontiff measured 58%
    decay in published anomalies AFTER publication, so the honest expectation is
    that most of that record is gone. Which is what the test is for.

    Long/flat only, like every strategy in this file. Connors' short side needs
    borrow costs the engine does not model.
    """

    name = "connors_rsi2"
    defaults = {"rsi_period": 2, "entry": 5, "exit_ma": 5, "trend_ma": 200}
    grid = {"rsi_period": [2, 3], "entry": [5, 10, 15], "exit_ma": [5, 10],
            "trend_ma": [100, 200]}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        rsi = _rsi(close, self.params["rsi_period"])
        trend = close.rolling(self.params["trend_ma"]).mean()
        fast = close.rolling(self.params["exit_ma"]).mean()
        uptrend = close > trend
        enter = (rsi < self.params["entry"]) & uptrend
        exit_ = (close > fast) | (~uptrend)
        return _hold(enter, exit_)


class BollingerReversion(Strategy):
    """Mean reversion: buy a close below the lower band, exit back at the middle."""

    name = "bollinger"
    defaults = {"period": 20, "num_std": 2.0}
    grid = {"period": [10, 20, 30, 50], "num_std": [1.5, 2.0, 2.5, 3.0]}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        mid = close.rolling(self.params["period"]).mean()
        sd = close.rolling(self.params["period"]).std()
        lower = mid - self.params["num_std"] * sd
        return _hold(close < lower, close > mid)


class BuyAndHold(Strategy):
    """The benchmark every other strategy has to beat to be worth running."""

    name = "buy_and_hold"
    defaults: dict = {}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index)


# ---------------------------------------------------------------------------
# Godmode Oscillator support
#
# Ported from the open-source Godmode Oscillator (MPL-2.0; original by LEGION,
# with contributions from LazyBear, xSilas and Ni6HTH4wK). Four components —
# TCI, CSI, Money Flow and Willy — averaged into one 0-100 oscillator, then a
# crossover against its own smoothed line confirmed by a rising/falling filter.
#
# Deviation worth stating: the published script offers a multi-exchange mode
# feeding two different price sources (x and y) into CSI. There is one source
# here, so x == y. Source is close; the Pine default is user-selectable and the
# published spec did not pin it.
# ---------------------------------------------------------------------------


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder / SMMA smoothing — what the script's default slow line uses."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Divide, turning a zero denominator into NaN rather than inf.

    All three components below can hit a flat window (zero deviation, zero
    range). inf would propagate into the averaged oscillator and poison every
    downstream comparison; NaN just reads as "no signal yet".
    """
    return a / b.replace(0.0, np.nan)


def _tci(x: pd.Series, channel: int, average: int) -> pd.Series:
    ema_x = _ema(x, channel)
    deviation = _ema((x - ema_x).abs(), channel)
    return _ema(_safe_div(x - ema_x, 0.025 * deviation), average) + 50


def _money_flow(x: pd.Series, volume: pd.Series, n: int) -> pd.Series:
    change = x.diff()
    up = (volume * x.where(change > 0, 0.0)).rolling(n).sum()
    down = (volume * x.where(change < 0, 0.0)).rolling(n).sum()
    return 100 - 100 / (1 + _safe_div(up, down))


def _willy(x: pd.Series, n: int) -> pd.Series:
    high = x.rolling(n).max()
    low = x.rolling(n).min()
    return 60 * _safe_div(x - high, high - low) + 80


def _tsi(x: pd.Series, short: int, long: int) -> pd.Series:
    """True Strength Index, range -1..1 (Pine's ta.tsi convention)."""
    change = x.diff()
    smoothed = _ema(_ema(change, long), short)
    smoothed_abs = _ema(_ema(change.abs(), long), short)
    return _safe_div(smoothed, smoothed_abs)


class Godmode(Strategy):
    """Godmode Oscillator: four momentum components averaged, traded on crossover."""

    name = "godmode"
    defaults = {"channel": 9, "average": 26, "short": 13, "slow": 32}
    grid = {"channel": [5, 9, 14], "average": [13, 26], "short": [13, 21], "slow": [32]}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        channel, average = self.params["channel"], self.params["average"]
        short, slow = self.params["short"], self.params["slow"]
        src = df["close"]

        tci = _tci(src, channel, average)
        money_flow = _money_flow(src, df["volume"], short)
        willy = _willy(src, average)
        csi = (_tsi(src, channel, average) * 50 + 50 + _rsi(src, short)) / 2

        fast = (tci + csi + money_flow + willy) / 4
        slow_line = _rma(fast, slow)
        # The published signal filter: a crossover only counts when this
        # smoothed spread is still moving in the same direction.
        difference = _ema((fast - slow_line) * 2 + 50, 13)

        crossed_up = (fast > slow_line) & (fast.shift(1) <= slow_line.shift(1))
        crossed_down = (fast < slow_line) & (fast.shift(1) >= slow_line.shift(1))
        rising = difference > difference.shift(1)
        falling = difference < difference.shift(1)

        return _hold(crossed_up & rising, crossed_down & falling)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI. ewm(alpha=1/period) is Wilder smoothing, not a plain EMA."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (MACross, RSIReversion, BollingerReversion, ConnorsRSI2,
                              Godmode, BuyAndHold)
}


def build(name: str, **params) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name](**params)
