"""Permutation nulls + walk-forward for the mean-reversion strategies.

Two departures from validate.py, both stated so they can be argued with:

1. THE SHUFFLE. validate.shuffle_bars rebuilds high/low as max/min(open, close),
   which throws the wicks away. For IBS that is not a null - with no wicks, IBS
   is exactly 1.0 on every up bar and 0.0 on every down bar, so the strategy
   being tested against the null is not the strategy. mr_strategies.shuffle_bars_full
   shuffles the ORDER of bars while carrying each bar's full (gap, high, low,
   close) geometry as one unit. Same destruction of across-bar structure, no
   destruction of within-bar structure. Used for every strategy here so the
   comparison across them is on one scale.

2. THE TICKERS. Nulls are run on SPY/QQQ/IWM plus five names drawn from the
   universe with a fixed seed - not chosen by me, and not chosen after seeing
   results. Eight separate p-values, reported individually AND combined by
   Fisher, because one p-value on SPY is a single draw.

The null's own limitation still applies: shuffling preserves drift, so the
distribution sits above a zero-drift baseline and the test is conservative.
"""

from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

import validate
import mr_core as C
import mr_strategies
from mr_strategies import TF, permutation_null_full

FEE, SLIP = 0.0, 0.0005
RUNS = 40
NAMES = ["connors_rsi2", "mr_cumrsi", "mr_cumrsi_nt", "mr_vwap", "mr_ibs", "mr_ndown"]


def tickers(k: int = 5) -> list[str]:
    data = C.universe()
    pool = sorted(t for t, d in data.items()
                  if len(d) > 5000 and t not in ("SPY", "QQQ", "IWM"))
    rng = np.random.default_rng(20260903)
    picked = list(rng.choice(pool, size=k, replace=False))
    return ["SPY", "QQQ", "IWM"] + sorted(picked)


def one(name: str, ticker: str, df: pd.DataFrame) -> dict:
    t0 = time.time()
    wf = validate.walk_forward(df, name, timeframe=TF, fee=FEE, slippage=SLIP)
    real = wf.oos_metrics["sharpe"]
    null = permutation_null_full(df, name, runs=RUNS, timeframe=TF,
                                 fee=FEE, slippage=SLIP)
    p = float((np.sum(null >= real) + 1) / (len(null) + 1))   # add-one, never 0.00
    return {"strategy": name, "ticker": ticker, "oos_sharpe": real,
            "null_med": float(np.median(null)), "null_p95": float(np.percentile(null, 95)),
            "runs": len(null), "p": p, "secs": time.time() - t0,
            "is_sharpe": wf.in_sample_metrics["sharpe"],
            "params": wf.in_sample_params}


def main() -> None:
    which = sys.argv[1:] or NAMES
    data = C.universe()
    tick = tickers()
    print(f"  permutation nulls: {RUNS} runs, bar-preserving shuffle, "
          f"walk-forward train=365 test=90, 5bp/side")
    print(f"  tickers (3 indexes + 5 seeded-random): {', '.join(tick)}\n")

    rows = []
    for name in which:
        print(f"\n  {name}")
        print("  {:<8}{:>12}{:>11}{:>11}{:>9}{:>10}{:>8}".format(
            "ticker", "OOS Sharpe", "null med", "null p95", "p", "IS Sharpe", "secs"))
        print("  " + "-" * 70)
        for t in tick:
            r = one(name, t, data[t])
            rows.append(r)
            print("  {:<8}{:>12}{:>11}{:>11}{:>9}{:>10}{:>8.0f}".format(
                t, C.fmt(r["oos_sharpe"]), C.fmt(r["null_med"]), C.fmt(r["null_p95"]),
                f"{r['p']:.3f}", C.fmt(r["is_sharpe"]), r["secs"]), flush=True)
        sub = pd.DataFrame([r for r in rows if r["strategy"] == name])
        chi = -2 * np.log(sub["p"]).sum()
        fisher = 1 - stats.chi2.cdf(chi, 2 * len(sub))
        print(f"  combined (Fisher, {len(sub)} tickers): p = {fisher:.4f}"
              f"   | median OOS Sharpe {sub['oos_sharpe'].median():.2f}"
              f" vs median null {sub['null_med'].median():.2f}"
              f" | {int((sub['p'] < 0.10).sum())}/{len(sub)} tickers at p<0.10")

    out = pd.DataFrame(rows)
    out.to_csv("mr_perm_results.csv", index=False)
    print("\n\n  SUMMARY (Fisher-combined across 8 tickers)")
    print("  " + "-" * 62)
    for name in which:
        sub = out[out["strategy"] == name]
        chi = -2 * np.log(sub["p"]).sum()
        fisher = 1 - stats.chi2.cdf(chi, 2 * len(sub))
        print(f"  {name:<16} p = {fisher:.4f}   median OOS Sharpe "
              f"{sub['oos_sharpe'].median():>5.2f}   null {sub['null_med'].median():>5.2f}")


if __name__ == "__main__":
    main()
