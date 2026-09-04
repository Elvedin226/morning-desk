"""Momentum / breakout strategies, registered into strategy.STRATEGIES at import.

Defined here rather than in strategy.py because strategy.py is owned by another
agent. Registration at import time is functionally identical.

All four obey the engine's contract: signals may read bar t's own close, and
backtest.run applies the t+1 execution delay itself. Nothing is shifted here
except where a rule genuinely needs YESTERDAY's value (the Donchian channel must
exclude today's bar or the breakout is trivially self-referential; ATR likewise,
since a big-range bar inflates its own ATR and would swallow its own signal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import strategy as S


def _hold_n(signal: pd.Series, n: int) -> pd.Series:
    """Long for the n bars following a signal. Overlapping signals just extend."""
    return signal.fillna(False).astype("float64").rolling(n, min_periods=1).max().fillna(0.0)


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


class Donchian(S.Strategy):
    """Turtle breakout: long on a close above the prior N-day high, out below the
    prior M-day low. Stop-and-reverse style, so exposure is high by construction."""

    name = "mom_donchian"
    defaults = {"entry": 20, "exit": 10}
    grid = {"entry": [20, 55], "exit": [10, 20]}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        # shift(1): the channel is the PRIOR N bars. Including today's own high
        # would make "close above the N-day high" nearly unbreakable by definition.
        hi = df["high"].rolling(self.params["entry"]).max().shift(1)
        lo = df["low"].rolling(self.params["exit"]).min().shift(1)
        return S._hold(close > hi, close < lo)


class VolBreakout(S.Strategy):
    """Range expansion: long when the close exceeds the open by K x ATR(14).

    ATR is the PRIOR bar's, for the reason above. No exit is specified in the
    classic formulation, so the exit is a fixed hold -- swept, not assumed."""

    name = "mom_volbreak"
    defaults = {"k": 1.0, "atr": 14, "hold": 5}
    grid = {"k": [0.5, 1.0, 1.5], "hold": [1, 5, 10]}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        atr = _atr(df, self.params["atr"]).shift(1)
        fired = df["close"] > df["open"] + self.params["k"] * atr
        return _hold_n(fired, self.params["hold"])


class High52(S.Strategy):
    """Long while price sits within X% of its 52-week high; hold N days.

    The 252-day max includes today's close, which is legitimate -- it is known at
    the moment the signal is evaluated, and the engine still delays the fill."""

    name = "mom_52wk"
    defaults = {"pct": 0.02, "hold": 21, "lookback": 252}
    grid = {"pct": [0.02, 0.05], "hold": [21, 63]}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        hi = close.rolling(self.params["lookback"]).max()
        near = close >= hi * (1 - self.params["pct"])
        return _hold_n(near, self.params["hold"])


class GapGo(S.Strategy):
    """Daily proxy for gap-and-go: open gaps > X% above the prior close AND the
    close holds above the open. Both facts are known at bar t's close; the fill
    is at bar t+1's open, so this does NOT capture the intraday gap move itself.

    That is the honest version. A backtest that buys the gap at the open and
    sells at the same day's close is reading the future twice -- it needs the
    close to know the gap held, then trades on the open that already passed."""

    name = "mom_gapgo"
    defaults = {"gap": 0.02, "hold": 3}
    grid = {"gap": [0.01, 0.02, 0.03], "hold": [1, 3, 5]}

    def signals(self, df: pd.DataFrame) -> pd.Series:
        gapped = df["open"] / df["close"].shift(1) - 1 > self.params["gap"]
        held = df["close"] > df["open"]
        return _hold_n(gapped & held, self.params["hold"])


for _cls in (Donchian, VolBreakout, High52, GapGo):
    S.STRATEGIES[_cls.name] = _cls

MOM_STRATEGIES = ["mom_donchian", "mom_volbreak", "mom_52wk", "mom_gapgo"]
