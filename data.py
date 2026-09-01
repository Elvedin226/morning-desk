"""OHLCV price data: fetch from an exchange, cache to disk, hand back a DataFrame."""

from __future__ import annotations

import time
from pathlib import Path

import ccxt
import pandas as pd

CACHE_DIR = Path(__file__).parent / "data_cache"

# Exchanges cap a single fetch_ohlcv call (1000 bars is the common ceiling), so any
# span longer than that has to be walked forward one page at a time.
PAGE_LIMIT = 1000

# Hard ceiling on pagination so a misbehaving exchange cannot spin forever.
# 200 pages x 1000 bars is far more than any sane backtest window.
MAX_PAGES = 200

# Exchange choice is not cosmetic — it decides how much history you actually get:
#   binance  451s from some regions (including this machine) — unusable here.
#   kraken   IGNORES `since` and always returns only the most recent ~720 bars. Ask for
#            2020 and you silently get 2024 onward. Do not use it for backtests.
#   kucoin   paginates correctly, broad symbol coverage (USDT pairs). Default.
#   bitstamp paginates correctly, fewer pairs, quotes in USD not USDT.
DEFAULT_EXCHANGE = "kucoin"


def _cache_path(exchange_id: str, symbol: str, timeframe: str) -> Path:
    return CACHE_DIR / f"{exchange_id}_{symbol.replace('/', '-')}_{timeframe}.csv"


def _drop_unclosed_bar(df: pd.DataFrame, timeframe_ms: int) -> pd.DataFrame:
    """Drop the final bar if its period hasn't ended yet.

    The most recent candle is still forming — its close will keep moving until the
    period ends. Backtesting on it means acting on a price that never existed at
    decision time, which quietly inflates results.
    """
    if df.empty:
        return df
    now_ms = int(time.time() * 1000)
    last_open_ms = int(df.index[-1].timestamp() * 1000)
    if last_open_ms + timeframe_ms > now_ms:
        return df.iloc[:-1]
    return df


def load_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    since: str = "2020-01-01",
    exchange_id: str = DEFAULT_EXCHANGE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame indexed by UTC open time with open/high/low/close/volume.

    Results are cached as CSV, so re-running a backtest doesn't re-hit the network.
    Pass refresh=True to force a fresh pull.
    """
    path = _cache_path(exchange_id, symbol, timeframe)
    since_ts = pd.Timestamp(since, tz="UTC")

    # The cache holds whatever span was last fetched, and `since` is applied as a
    # slice on the way out. Keying the file on `since` instead would refetch the
    # whole history for every new start date; returning the cache unsliced (the
    # original bug) silently handed back 2020-onward data to someone who asked for
    # 2023-onward, so the backtest quietly covered the wrong period.
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
        if not df.empty and df.index[0] <= since_ts:
            return df.loc[df.index >= since_ts]
        # Cache doesn't reach back far enough for this request — refetch.

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    since_ms = exchange.parse8601(f"{since}T00:00:00Z")

    now_ms = int(time.time() * 1000)
    rows: list[list] = []
    # Stop on: no data, no forward progress, or having reached the present. NOT on a
    # short page — KuCoin bounds each request by a computed end time, so a page of 999
    # means "end of this window", not "end of history". Treating it as the end silently
    # truncated 14 months off a 2020-start fetch.
    for _ in range(MAX_PAGES):
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=PAGE_LIMIT)
        if not batch:
            break
        rows.extend(batch)
        next_since = batch[-1][0] + 1
        # Guard against an exchange that ignores `since` and returns the same page
        # forever — without this the loop never terminates.
        if next_since <= since_ms:
            break
        since_ms = next_since
        if batch[-1][0] + timeframe_ms > now_ms:
            break

    if not rows:
        raise RuntimeError(f"No data returned for {symbol} {timeframe} from {exchange_id}.")

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    df = _drop_unclosed_bar(df, timeframe_ms)

    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(path)
    return df.loc[df.index >= since_ts]
