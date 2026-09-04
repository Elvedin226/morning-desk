"""Shared measurement helpers for the mean-reversion study.

Conventions, stated once:

UNIVERSE      Every ticker in watchlist.SECTORS plus SPY/QQQ/IWM (93 with usable
              history). Never hand-picked, never filtered on outcome.
COSTS         fee = 0 (commission-free broker), slippage swept per side.
              backtest.run charges fee+slippage on every position change, so a
              round trip pays twice the quoted number.
ANNUALIZE     252 bars/year (mr_strategies.TF), not the engine's crypto default.
PORTFOLIO     Equal weight across all names every day: the daily portfolio return
              is the cross-sectional mean of each name's net return. A name in
              cash contributes 0. This is the number that includes exposure drag,
              and it is the one that has to beat buy-and-hold.
EV/TRADE      Pooled round-trip returns, net of both sides' costs. The t-stat is
              clustered by entry date (average all same-day trades into one
              observation first), because mean reversion fires on many names at
              once and treating those as independent would inflate t by ~3x.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

import backtest
import strategy
import mr_data
import mr_strategies
from mr_strategies import TF

SLIPS = [0.0, 0.0005, 0.0010, 0.0020, 0.0050]
FEE = 0.0

_DATA: dict[str, pd.DataFrame] | None = None


def universe() -> dict[str, pd.DataFrame]:
    global _DATA
    if _DATA is None:
        _DATA = mr_data.load()
    return _DATA


def evaluate(name: str, params: dict, slip: float,
             data: dict[str, pd.DataFrame] | None = None) -> dict:
    """Run one config across the whole universe. Returns aggregate + raw pieces."""
    data = data or universe()
    st = strategy.build(name, **params)
    bh = strategy.build("buy_and_hold")

    rows, trades, port, bench = [], [], {}, {}
    for t, df in data.items():
        r = backtest.run(df, st, timeframe=TF, fee_pct=FEE, slippage_pct=slip)
        b = backtest.run(df, bh, timeframe=TF, fee_pct=FEE, slippage_pct=0.0)
        m = r.metrics
        rows.append({"ticker": t, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "maxdd": m["max_drawdown"], "exposure": m["exposure"],
                     "trades": m["num_trades"], "win_rate": m["win_rate"],
                     "avg_trade": m["avg_trade"], "bh_cagr": b.metrics["cagr"],
                     "bh_sharpe": b.metrics["sharpe"]})
        if not r.trades.empty:
            tr = r.trades[["entry", "return", "bars", "closed"]].copy()
            tr["ticker"] = t
            trades.append(tr)
        port[t] = r.returns
        bench[t] = b.returns

    per_ticker = pd.DataFrame(rows)
    all_trades = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(
        columns=["entry", "return", "bars", "closed", "ticker"])

    pr = pd.DataFrame(port).mean(axis=1).dropna()
    br = pd.DataFrame(bench).mean(axis=1).dropna()

    return {"per_ticker": per_ticker, "trades": all_trades,
            "port_returns": pr, "bench_returns": br,
            "summary": summarize(per_ticker, all_trades, pr, br)}


def clustered_t(trades: pd.DataFrame) -> tuple[float, float, int]:
    """(mean return, t-stat clustered by entry date, n clusters)."""
    if trades.empty:
        return float("nan"), float("nan"), 0
    daily = trades.groupby("entry")["return"].mean()
    if len(daily) < 3:
        return float(daily.mean()), float("nan"), len(daily)
    t, _ = stats.ttest_1samp(daily.to_numpy(), 0.0)
    return float(daily.mean()), float(t), len(daily)


def summarize(per_ticker: pd.DataFrame, trades: pd.DataFrame,
              pr: pd.Series, br: pd.Series) -> dict:
    ev, tstat, nclust = clustered_t(trades)
    pm = backtest.core_metrics(pr, TF)
    bm = backtest.core_metrics(br, TF)
    return {
        "n_tickers": len(per_ticker),
        "med_cagr": float(per_ticker["cagr"].median()),
        "med_bh_cagr": float(per_ticker["bh_cagr"].median()),
        "beat_bh": float((per_ticker["cagr"] > per_ticker["bh_cagr"]).mean()),
        "med_sharpe": float(per_ticker["sharpe"].median()),
        "med_exposure": float(per_ticker["exposure"].median()),
        "total_trades": int(per_ticker["trades"].sum()),
        "med_win_rate": float(per_ticker["win_rate"].median()),
        "ev_trade": ev, "t_clustered": tstat, "n_clusters": nclust,
        "port_cagr": pm["cagr"], "port_sharpe": pm["sharpe"],
        "port_maxdd": pm["max_drawdown"],
        "bench_cagr": bm["cagr"], "bench_sharpe": bm["sharpe"],
        "bench_maxdd": bm["max_drawdown"],
    }


def by_year(res: dict) -> pd.DataFrame:
    """Per-calendar-year: portfolio vs benchmark return, and trade EV."""
    pr, br, tr = res["port_returns"], res["bench_returns"], res["trades"]
    rows = []
    years = sorted(set(pr.index.year))
    for y in years:
        p = pr[pr.index.year == y]
        b = br[br.index.year == y]
        yr_tr = tr[pd.to_datetime(tr["entry"]).dt.year == y] if not tr.empty else tr
        ev, t, n = clustered_t(yr_tr)
        rows.append({"year": y,
                     "port_ret": float((1 + p).prod() - 1),
                     "bh_ret": float((1 + b).prod() - 1),
                     "trades": int(len(yr_tr)), "ev_trade": ev, "t": t})
    return pd.DataFrame(rows)


def fmt_pct(x, nd=2):
    return "  n/a " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.{nd}f}%"


def fmt(x, nd=2):
    return " n/a " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"
