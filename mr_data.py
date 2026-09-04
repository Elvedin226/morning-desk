"""Universe loader + OHLCV cache for the mean-reversion study.

Universe is stated once, here, and never hand-picked: every ticker in
watchlist.SECTORS (~92 large caps grouped by sector ETF) plus SPY, QQQ and IWM.

SURVIVORSHIP: watchlist.SECTORS is today's large-cap list. Everything that
delisted, got acquired or blew up on the way here is missing. Measured elsewhere
in this project at roughly 5 percentage points of annual overstatement. Every
absolute return below is inflated by that; the comparisons between arms are not.
"""

from __future__ import annotations

import os
import pickle
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf

import watchlist

START = "2000-01-01"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "data_cache", "mr_panel.pkl")

INDEXES = ["SPY", "QQQ", "IWM"]
UNIVERSE = sorted(set(watchlist.SECTORS) | set(INDEXES))


def _clean(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            sub = raw.xs(ticker, axis=1, level=1)
        except KeyError:
            return pd.DataFrame()
    else:
        sub = raw
    sub = sub.rename(columns=str.lower)
    need = ["open", "high", "low", "close", "volume"]
    if not set(need).issubset(sub.columns):
        return pd.DataFrame()
    return sub[need].dropna()


def load(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """{ticker: OHLCV df}. Cached — yfinance is slow and rate-limited."""
    if not refresh and os.path.exists(CACHE):
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)

    out: dict[str, pd.DataFrame] = {}
    chunk = 25
    for i in range(0, len(UNIVERSE), chunk):
        batch = UNIVERSE[i:i + chunk]
        raw = yf.download(batch, start=START, auto_adjust=True,
                          progress=False, group_by="column")
        for t in batch:
            df = _clean(raw, t)
            if len(df) >= 500:
                out[t] = df
        print(f"  fetched {min(i + chunk, len(UNIVERSE))}/{len(UNIVERSE)}", flush=True)

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh)
    return out


if __name__ == "__main__":
    data = load(refresh=True)
    print(f"\n  {len(data)} tickers cached of {len(UNIVERSE)} requested")
    lens = pd.Series({t: len(d) for t, d in data.items()})
    print(f"  bars: min {lens.min()}  median {int(lens.median())}  max {lens.max()}")
    first = pd.Series({t: d.index[0].date() for t, d in data.items()})
    print(f"  earliest start {first.min()}   latest start {first.max()}")
