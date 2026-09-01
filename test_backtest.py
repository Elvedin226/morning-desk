"""Tests for the two properties that decide whether a backtest is fiction.

Run: .venv/Scripts/python.exe test_backtest.py
"""

import numpy as np
import pandas as pd

import backtest as B
from strategy import Strategy


def _bars(closes, opens=None):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D", tz="UTC")
    opens = opens if opens is not None else closes
    return pd.DataFrame(
        {"open": opens, "high": closes, "low": closes, "close": closes, "volume": [1.0] * len(closes)},
        index=idx,
    )


class FlagOn(Strategy):
    """Signals long on exactly one bar index, flat everywhere else."""

    name = "flag"
    defaults = {"bar": 2}

    def signals(self, df):
        sig = pd.Series(0.0, index=df.index)
        sig.iloc[self.params["bar"]] = 1.0
        return sig


class Oracle(Strategy):
    """Cheats: at bar t it already knows bar t+1 closes higher."""

    name = "oracle"
    defaults: dict = {}

    def signals(self, df):
        return (df["close"].shift(-1) > df["close"]).astype("float64")


def test_signal_executes_one_bar_late():
    df = _bars([100.0] * 6)
    r = B.run(df, FlagOn(bar=2), fee_pct=0, slippage_pct=0)
    held = list(np.flatnonzero(r.position.to_numpy() > 0))
    assert held == [3], f"signal on bar 2 must be held on bar 3, got bars {held}"


def test_entry_fills_at_open_not_at_signal_close():
    # Signal fires on bar 1. Bar 2 opens at 200 and closes at 210.
    # Filling at bar 2's open earns 210/200-1 = 5%.
    # Filling at bar 1's close (100) would earn 110% — the classic inflated backtest.
    df = _bars(closes=[100.0, 100.0, 210.0, 210.0], opens=[100.0, 100.0, 200.0, 210.0])
    r = B.run(df, FlagOn(bar=1), fee_pct=0, slippage_pct=0)
    got = r.metrics["total_return"]
    assert abs(got - 0.05) < 1e-9, f"expected 5% (open fill), got {got:.4%}"


def test_costs_are_charged_on_both_sides():
    df = _bars([100.0] * 6)
    flat = B.run(df, FlagOn(bar=2), fee_pct=0.001, slippage_pct=0.0005)
    # Flat price, one full round trip: pay 0.15% in and 0.15% out. The two charges
    # compound rather than adding, so the drag is (1-0.0015)^2 - 1, not -0.3% flat.
    expected = (1 - 0.0015) ** 2 - 1
    assert abs(flat.metrics["total_return"] - expected) < 1e-6, flat.metrics["total_return"]


def test_oracle_shows_lookahead_is_detectable():
    """A strategy that peeks at t+1 should print absurd returns.

    If this ever fails, the engine has started shifting twice and is hiding real
    lookahead rather than exposing it.
    """
    rng = np.random.default_rng(0)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
    # Each bar must OPEN where the previous one closed. With open == close (the
    # default fixture) an entry fills at the same price the bar ends at, so the
    # entry bar earns exactly nothing and even a cheating strategy looks flat.
    opens = np.concatenate([[closes[0]], closes[:-1]])
    r = B.run(_bars(list(closes), list(opens)), Oracle(), fee_pct=0, slippage_pct=0)
    assert r.metrics["total_return"] > 10, r.metrics["total_return"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")
