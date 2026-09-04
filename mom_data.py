"""Shared data + cost/annualisation setup for the momentum-breakout study.

UNIVERSE: watchlist.SECTORS (the ~90 names the bot already trades) plus
SPY/QQQ/IWM. Fixed, declared up front, never filtered by outcome. This is the
whole point -- the options study that burned this project used NVDA/AMD/TSLA/
PLTR/COIN/SMCI, which is just "the names that went up", chosen with hindsight.

SURVIVORSHIP: this universe is today's large caps. Everything that delisted,
blew up or was acquired on the way here is missing. Measured elsewhere in this
project at roughly a 5 percentage point annual overstatement. Every CAGR below
-- strategy AND buy-and-hold -- is inflated by that. It biases both arms, so
strategy-minus-benchmark is the number to read, not the level.

ANNUALISATION: backtest.PERIODS_PER_YEAR ships with 365 bars/year for "1d"
because the engine was written for crypto. US equities trade ~252. Rather than
edit backtest.py (owned by another agent) this registers an extra key at import
time and every call below passes timeframe="1d_eq". Same math, correct constant.
"""

from __future__ import annotations

import os
import pickle
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf

import backtest
import watchlist

backtest.PERIODS_PER_YEAR["1d_eq"] = 252
TF = "1d_eq"

FEE = 0.0                                        # commission-free broker
SLIP_LEVELS = [0.0, 0.0005, 0.0010, 0.0020, 0.0050]   # 0, 5, 10, 20, 50 bp per side

START = "2007-01-01"
CACHE = "data_cache/mom_panel.pkl"

INDEX = ["SPY", "QQQ", "IWM"]
UNIVERSE = sorted(set(list(watchlist.SECTORS) + INDEX))


def load(min_bars: int = 750) -> dict[str, pd.DataFrame]:
    """{ticker: OHLCV DataFrame}. Cached to disk; delete the pickle to refresh."""
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as fh:
            panel = pickle.load(fh)
    else:
        raw = yf.download(UNIVERSE, start=START, auto_adjust=True,
                          progress=False, group_by="ticker")
        panel = {}
        for t in UNIVERSE:
            try:
                df = raw[t].rename(columns=str.lower)
            except KeyError:
                continue
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            if len(df):
                panel[t] = df
        os.makedirs("data_cache", exist_ok=True)
        with open(CACHE, "wb") as fh:
            pickle.dump(panel, fh)
    return {t: d for t, d in panel.items() if len(d) >= min_bars}


if __name__ == "__main__":
    p = load()
    print(f"universe declared : {len(UNIVERSE)} tickers")
    print(f"usable (>=750 bars): {len(p)}")
    lens = sorted((len(d), t) for t, d in p.items())
    print(f"shortest: {lens[0][1]} {lens[0][0]} bars, starts {p[lens[0][1]].index[0].date()}")
    print(f"longest : {lens[-1][1]} {lens[-1][0]} bars, starts {p[lens[-1][1]].index[0].date()}")
    full = [t for t, d in p.items() if d.index[0].year <= 2007]
    print(f"with full history from 2007: {len(full)}")
