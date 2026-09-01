"""Backtest engine: turn a strategy's target positions into an equity curve + metrics.

Execution model, stated plainly because this is where backtests lie:

  * A signal is computed from bars up to and including bar t.
  * The resulting position is held starting at bar t+1 (`shift(1)`).
  * Entries and exits fill at the OPEN of the bar they take effect on, not at the
    close of the bar that generated them. You cannot trade at a price you only
    learned about when the bar closed.
  * Every position change pays `fee_pct + slippage_pct` on the traded notional.

Long/flat only, all-in / all-out, no leverage, no compounding of partial sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy import Strategy

# Bar counts used to annualize. Crypto trades 24/7, so a "year" of daily bars is
# 365, not the 252 an equities backtest would use.
PERIODS_PER_YEAR = {"1d": 365, "4h": 365 * 6, "1h": 365 * 24, "15m": 365 * 96}


@dataclass
class Result:
    strategy: str
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def run(
    df: pd.DataFrame,
    strategy: Strategy,
    timeframe: str = "1d",
    fee_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    initial_cash: float = 10_000.0,
) -> Result:
    target = strategy.signals(df)
    # The one line that separates a real backtest from a fantasy: act on bar t+1.
    position = target.shift(1).fillna(0.0)
    prev = position.shift(1).fillna(0.0)

    open_, close = df["open"], df["close"]
    prev_close = close.shift(1)

    entering = (position == 1) & (prev == 0)
    holding = (position == 1) & (prev == 1)
    exiting = (position == 0) & (prev == 1)

    gross = pd.Series(
        np.select(
            [entering, holding, exiting],
            [close / open_ - 1, close / prev_close - 1, open_ / prev_close - 1],
            default=0.0,
        ),
        index=df.index,
    ).fillna(0.0)

    cost_rate = fee_pct + slippage_pct
    costs = (position - prev).abs() * cost_rate
    net = gross - costs

    equity = initial_cash * (1 + net).cumprod()
    trades = _round_trips(df, position, cost_rate)
    metrics = _metrics(equity, net, position, trades, timeframe, initial_cash)
    return Result(strategy.describe(), equity, net, position, trades, metrics)


def _round_trips(df: pd.DataFrame, position: pd.Series, cost_rate: float) -> pd.DataFrame:
    """Pair entries with exits into completed trades, priced at the fills actually used."""
    changes = position.diff().fillna(position.iloc[0])
    entries = list(df.index[changes > 0])
    exits = list(df.index[changes < 0])

    rows = []
    for i, entry_ts in enumerate(entries):
        entry_px = df.at[entry_ts, "open"]
        if i < len(exits):
            exit_ts = exits[i]
            exit_px = df.at[exit_ts, "open"]
            closed = True
        else:
            # Still holding at the end of the data — mark it out at the last close
            # rather than dropping it, so an open winner can't flatter the win rate
            # by being invisible.
            exit_ts = df.index[-1]
            exit_px = df.at[exit_ts, "close"]
            closed = False
        ret = (exit_px / entry_px - 1) - 2 * cost_rate
        rows.append(
            {
                "entry": entry_ts,
                "exit": exit_ts,
                "bars": df.index.get_loc(exit_ts) - df.index.get_loc(entry_ts),
                "entry_px": entry_px,
                "exit_px": exit_px,
                "return": ret,
                "closed": closed,
            }
        )
    return pd.DataFrame(rows)


def core_metrics(net: pd.Series, timeframe: str = "1d", initial_cash: float = 10_000.0) -> dict:
    """Return/risk metrics derivable from a net-return series alone.

    Split out from _metrics so walk-forward validation can score a *stitched*
    out-of-sample return series with exactly the same math a single backtest uses,
    instead of a second implementation that could quietly disagree.
    """
    ppy = PERIODS_PER_YEAR.get(timeframe, 365)
    equity = initial_cash * (1 + net).cumprod()
    years = len(net) / ppy
    final = float(equity.iloc[-1])
    sd = float(net.std())
    drawdown = equity / equity.cummax() - 1

    return {
        "final_equity": final,
        "total_return": final / initial_cash - 1,
        "cagr": (final / initial_cash) ** (1 / years) - 1 if years > 0 else float("nan"),
        # Excess-return Sharpe with a 0% risk-free rate. Fine for comparing strategies
        # on the same data; not a number to quote anywhere else.
        "sharpe": float(net.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
    }


def _metrics(equity, net, position, trades, timeframe, initial_cash) -> dict:
    wins = trades["return"] > 0 if not trades.empty else pd.Series(dtype=bool)
    return {
        **core_metrics(net, timeframe, initial_cash),
        "exposure": float((position > 0).mean()),
        "num_trades": int(len(trades)),
        "win_rate": float(wins.mean()) if len(wins) else float("nan"),
        "avg_trade": float(trades["return"].mean()) if not trades.empty else float("nan"),
    }
