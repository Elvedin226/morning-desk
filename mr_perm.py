"""Permutation nulls + walk-forward for the mean-reversion strategies.

Four departures from validate.py, all stated so they can be argued with:

1. THE SHUFFLE. validate.shuffle_bars rebuilds high/low as max/min(open, close),
   which throws the wicks away. For IBS that is not a null - with no wicks, IBS
   is exactly 1.0 on every up bar and 0.0 on every down bar, so the thing tested
   against the null is not the thing being claimed. mr_strategies.shuffle_bars_full
   shuffles the ORDER of bars while carrying each bar's full (gap, high, low,
   close) geometry as one unit. Same destruction of across-bar structure, none
   of the within-bar structure. Used for every strategy so the scale is common.

2. THE TICKERS. Nulls run on SPY/QQQ/IWM plus three names drawn from the
   universe with a fixed seed - not chosen by me, and not chosen after seeing
   results. Six separate p-values, reported individually AND combined by
   Fisher, because one p-value on SPY is a single draw.

3. ADD-ONE p-values: (#null >= real + 1) / (runs + 1). With 40 runs a raw p can
   read 0.000, which is not a thing 40 draws can establish.

4. connors_rsi2's published grid is 24 combinations, which makes 40 walk-forward
   nulls take ~12 minutes per ticker. `mr_connors` below is the same strategy
   with the entry threshold as the only swept parameter, so the baseline gets a
   null on the same footing as the others in a tractable time. The point
   estimate is unaffected - the grid only enters the walk-forward search.

The null's own limitation still applies: shuffling preserves drift, so the
distribution sits above a zero-drift baseline and the test is conservative.
"""

from __future__ import annotations

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

import strategy
import validate
import mr_core as C
import mr_strategies
from mr_strategies import TF, permutation_null_full


class ConnorsCoarse(strategy.ConnorsRSI2):
    name = "mr_connors"
    grid = {"entry": [5, 10, 15]}


strategy.STRATEGIES[ConnorsCoarse.name] = ConnorsCoarse

FEE, SLIP = 0.0, 0.0005
RUNS = 40
NAMES = ["mr_connors", "mr_cumrsi", "mr_cumrsi_nt", "mr_vwap", "mr_ibs", "mr_ndown"]


def tickers(k: int = 3) -> list[str]:
    data = C.universe()
    pool = sorted(t for t, d in data.items()
                  if len(d) > 5000 and t not in ("SPY", "QQQ", "IWM"))
    rng = np.random.default_rng(20260903)
    return ["SPY", "QQQ", "IWM"] + sorted(rng.choice(pool, size=k, replace=False))


def one(job: tuple[str, str]) -> dict:
    name, ticker = job
    df = C.universe()[ticker]
    t0 = time.time()
    wf = validate.walk_forward(df, name, timeframe=TF, fee=FEE, slippage=SLIP)
    real = wf.oos_metrics["sharpe"]
    null = permutation_null_full(df, name, runs=RUNS, timeframe=TF,
                                 fee=FEE, slippage=SLIP)
    p = float((np.sum(null >= real) + 1) / (len(null) + 1))
    return {"strategy": name, "ticker": ticker, "oos_sharpe": real,
            "null_med": float(np.median(null)), "null_p95": float(np.percentile(null, 95)),
            "runs": len(null), "p": p, "secs": time.time() - t0,
            "is_sharpe": wf.in_sample_metrics["sharpe"],
            "oos_cagr": wf.oos_metrics["cagr"]}


def fisher(sub: pd.DataFrame) -> float:
    chi = -2 * np.log(sub["p"]).sum()
    return float(1 - stats.chi2.cdf(chi, 2 * len(sub)))


def main() -> None:
    which = sys.argv[1:] or NAMES
    tick = tickers()
    jobs = [(n, t) for n in which for t in tick]
    print(f"  permutation nulls: {RUNS} runs each, bar-preserving shuffle,")
    print(f"  walk-forward train=365 test=90, 5bp/side, 2000-2026 full history")
    print(f"  tickers (3 indexes + 3 seeded-random): {', '.join(tick)}")
    print(f"  {len(jobs)} jobs\n", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for r in ex.map(one, jobs):
            rows.append(r)
            print("  {:<14}{:<7}OOS {:>6}  null med {:>6}  p95 {:>6}  p={:<7}({:.0f}s)".format(
                r["strategy"], r["ticker"], C.fmt(r["oos_sharpe"]), C.fmt(r["null_med"]),
                C.fmt(r["null_p95"]), f"{r['p']:.3f}", r["secs"]), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("mr_perm_results.csv", index=False)

    print("\n\n  SUMMARY  (Fisher-combined across the 6 tickers)")
    print("  {:<14}{:>10}{:>14}{:>12}{:>14}".format(
        "strategy", "Fisher p", "med OOS Shp", "med null", "tickers p<.10"))
    print("  " + "-" * 66)
    for name in which:
        sub = out[out["strategy"] == name]
        print("  {:<14}{:>10}{:>14}{:>12}{:>14}".format(
            name, f"{fisher(sub):.4f}", C.fmt(sub["oos_sharpe"].median()),
            C.fmt(sub["null_med"].median()),
            f"{int((sub['p'] < 0.10).sum())}/{len(sub)}"))


if __name__ == "__main__":
    main()
