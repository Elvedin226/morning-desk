"""Walk-forward validation and a permutation baseline.

A single backtest tells you how a strategy would have done if you had already known
the best parameters. You didn't. Walk-forward measures the thing you actually care
about: pick parameters using only the past, then trade them on data you haven't seen.

    train window -> pick best params -> trade the next test window -> roll forward

Only test-window returns are kept. Stitched end to end they form one out-of-sample
equity curve, which is the honest number.

The permutation baseline answers the follow-up question: is that number *good*? The
same optimisation is run against price series built by shuffling the real bars, which
destroys any exploitable time structure while keeping the return distribution intact.
Whatever Sharpe the search extracts from shuffled data is the score it can produce
from nothing. A real strategy has to beat that distribution, not merely beat zero.

Two honest limits on the permutation baseline:

  * It is degenerate for buy_and_hold. There is no timing to destroy, so shuffling
    only changes which returns happen to land in the test windows. Read the p-value
    for timing strategies; ignore it for always-in ones.
  * Shuffling preserves drift but destroys volatility clustering, and clustering
    depresses Sharpe. So the null sits ABOVE what a zero-drift baseline would give,
    and the test is conservative: it asks whether the strategy beats a rising asset
    with no exploitable ordering, which is the relevant bar for a timing rule.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

import backtest as B
import strategy as S


@dataclass
class WalkForward:
    strategy: str
    oos_returns: pd.Series
    windows: pd.DataFrame
    oos_metrics: dict
    in_sample_metrics: dict
    in_sample_params: dict


def param_combos(name: str) -> list[dict]:
    grid = S.STRATEGIES[name].grid
    if not grid:
        return [{}]
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _score(df: pd.DataFrame, name: str, params: dict, timeframe, fee, slippage) -> float:
    """Sharpe of one parameter set on one slice. NaN scores sort last."""
    result = B.run(df, S.build(name, **params), timeframe, fee, slippage)
    sharpe = result.metrics["sharpe"]
    return -np.inf if np.isnan(sharpe) else sharpe


def optimize(df: pd.DataFrame, name: str, timeframe="1d", fee=0.001, slippage=0.0005):
    """Best parameters on this slice, by Sharpe."""
    combos = param_combos(name)
    scores = [_score(df, name, c, timeframe, fee, slippage) for c in combos]
    best = int(np.argmax(scores))
    return combos[best], scores[best]


def walk_forward(
    df: pd.DataFrame,
    name: str,
    train: int = 365,
    test: int = 90,
    timeframe: str = "1d",
    fee: float = 0.001,
    slippage: float = 0.0005,
) -> WalkForward:
    rows, oos_chunks = [], []

    start = 0
    while start + train + test <= len(df):
        train_slice = df.iloc[start : start + train]
        # The backtest runs over train+test together so indicators are already warm
        # when the test window opens; only the test portion's returns are kept. Scoring
        # a cold-started test window would penalise every strategy for its own warmup.
        context = df.iloc[start : start + train + test]

        params, train_sharpe = optimize(train_slice, name, timeframe, fee, slippage)
        result = B.run(context, S.build(name, **params), timeframe, fee, slippage)
        oos = result.returns.iloc[train:]
        oos_chunks.append(oos)

        rows.append(
            {
                "test_start": oos.index[0].date(),
                "test_end": oos.index[-1].date(),
                "params": ", ".join(f"{k}={v}" for k, v in params.items()) or "-",
                "train_sharpe": train_sharpe,
                "test_return": float((1 + oos).prod() - 1),
            }
        )
        start += test

    if not oos_chunks:
        raise ValueError(
            f"Not enough data: need at least {train + test} bars for train={train} "
            f"test={test}, have {len(df)}."
        )

    oos_returns = pd.concat(oos_chunks)
    is_params, _ = optimize(df, name, timeframe, fee, slippage)
    is_result = B.run(df, S.build(name, **is_params), timeframe, fee, slippage)

    return WalkForward(
        strategy=name,
        oos_returns=oos_returns,
        windows=pd.DataFrame(rows),
        oos_metrics=B.core_metrics(oos_returns, timeframe),
        in_sample_metrics=is_result.metrics,
        in_sample_params=is_params,
    )


def shuffle_bars(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Rebuild the price path from the same bars in a random order.

    Each bar contributes two moves: previous close -> this open (the gap) and this
    open -> this close (the body). Shuffling those pairs together keeps each bar's
    internal shape and the overall return distribution, while destroying the
    across-bar ordering that every strategy here depends on.
    """
    open_, close = df["open"].to_numpy(), df["close"].to_numpy()
    gap = open_[1:] / close[:-1]
    body = close[1:] / open_[1:]

    order = rng.permutation(len(gap))
    gap, body = gap[order], body[order]

    new_close = close[0] * np.concatenate([[1.0], np.cumprod(gap * body)])
    new_open = np.concatenate([[open_[0]], new_close[:-1] * gap])

    return pd.DataFrame(
        {
            "open": new_open,
            "high": np.maximum(new_open, new_close),
            "low": np.minimum(new_open, new_close),
            "close": new_close,
            "volume": df["volume"].to_numpy(),
        },
        index=df.index,
    )


def permutation_null(
    df: pd.DataFrame,
    name: str,
    runs: int = 50,
    seed: int = 0,
    **kwargs,
) -> np.ndarray:
    """Out-of-sample Sharpe from `runs` shuffled price series — the no-edge distribution."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(runs):
        wf = walk_forward(shuffle_bars(df, rng), name, **kwargs)
        out.append(wf.oos_metrics["sharpe"])
    return np.array([s for s in out if not np.isnan(s)])
