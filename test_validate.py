"""Tests for walk-forward validation.

Run: .venv/Scripts/python.exe test_validate.py
"""

import numpy as np
import pandas as pd

import validate as V


def _series(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, n)))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": np.maximum(opens, closes),
         "low": np.minimum(opens, closes), "close": closes, "volume": np.ones(n)},
        index=idx,
    )


def test_test_windows_are_disjoint_ordered_and_unseen():
    """The whole point: no bar is scored twice, and none is scored before its window."""
    df = _series()
    wf = V.walk_forward(df, "ma_cross", train=365, test=90, )
    assert wf.oos_returns.index.is_monotonic_increasing
    assert not wf.oos_returns.index.duplicated().any()
    # Nothing inside the very first training window may ever appear in the scored set.
    first_train_end = df.index[365]
    assert wf.oos_returns.index.min() >= first_train_end


def test_scored_bars_equal_windows_times_test_length():
    df = _series()
    wf = V.walk_forward(df, "ma_cross", train=365, test=90)
    assert len(wf.oos_returns) == len(wf.windows) * 90


def test_rejects_data_too_short_for_one_window():
    df = _series(n=200)
    try:
        V.walk_forward(df, "ma_cross", train=365, test=90)
    except ValueError as err:
        assert "Not enough data" in str(err)
    else:
        raise AssertionError("expected ValueError on insufficient data")


def test_shuffle_preserves_returns_but_destroys_order():
    df = _series()
    shuffled = V.shuffle_bars(df, np.random.default_rng(1))

    real = np.sort((df["close"] / df["close"].shift(1)).dropna().to_numpy())
    fake = np.sort((shuffled["close"] / shuffled["close"].shift(1)).dropna().to_numpy())
    # Same bag of moves, different order — that is what makes it a fair null.
    assert np.allclose(real, fake), "shuffling must not change the return distribution"

    same_order = np.allclose(df["close"].to_numpy(), shuffled["close"].to_numpy())
    assert not same_order, "shuffling must actually change the path"


def test_param_grid_expands_to_full_product():
    assert len(V.param_combos("ma_cross")) == 4 * 4
    assert len(V.param_combos("buy_and_hold")) == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")
