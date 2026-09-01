"""Daily options snapshot collector — grows the dataset you can't afford to buy.

Historical option chains with IV cost $100-300. Live chains are free. Run this
once a trading day and in ~4-6 months you own the equivalent history for your
own universe, with the forward outcomes attached by the time you need them.

    .venv/Scripts/python.exe collect.py

Appends one row per ticker per run to data_cache/iv_history.csv. Idempotent per
day: re-running the same day replaces that day's rows rather than duplicating.
Schedule it with Windows Task Scheduler, weekdays, shortly before the close.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import ivcheck
import screener

OUT = Path(__file__).parent / "data_cache" / "iv_history.csv"


def snapshot() -> pd.DataFrame:
    close = screener.load_universe(start="2022-01-01")
    df = ivcheck.compare(close, list(close.columns))
    if df.empty:
        raise RuntimeError("No option chains returned — check network or market hours.")

    # Spot and realized vol travel with the row: without them, a future reader
    # cannot reconstruct what the IV was rich or cheap RELATIVE TO.
    df["date"] = pd.Timestamp.utcnow().tz_localize(None).normalize()
    df["spot"] = [float(close[t].iloc[-1]) for t in df["ticker"]]
    return df


def append(df: pd.DataFrame) -> int:
    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists():
        old = pd.read_csv(OUT, parse_dates=["date"])
        # Drop any existing rows for the same day so a re-run corrects rather
        # than duplicates — the collector will be re-run after failures.
        old = old[old["date"].dt.normalize() != df["date"].iloc[0]]
        df = pd.concat([old, df], ignore_index=True)
    df = df.sort_values(["date", "ticker"])
    df.to_csv(OUT, index=False)
    return len(df)


if __name__ == "__main__":
    today = snapshot()
    total = append(today)
    print(f"captured {len(today)} tickers on {today['date'].iloc[0].date()}")
    print(f"iv_history.csv now holds {total} rows across "
          f"{pd.read_csv(OUT, parse_dates=['date'])['date'].nunique()} day(s)")
    print(f"  median IV/realized {today['iv_over_realized'].median():.2f} | "
          f"median spread {today['spread_pct'].median()*100:.1f}%")
    print(f"\n  -> {OUT}")
