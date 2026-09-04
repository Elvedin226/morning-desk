"""Permutation nulls for the four single-name momentum/breakout rules.

Two nulls, because they answer different questions and neither alone is enough.

(A) POOLED PANEL NULL -- the one that matters here.
    Shuffle EVERY ticker's bars independently (validate.shuffle_bars), rebuild
    the whole equal-weight portfolio, and re-run the same parameter search that
    was run on real data. Repeat 40x. This asks: given 92 names, a grid to search
    and 19 years, how much Sharpe does this procedure manufacture from price
    paths with no exploitable ordering? The real number has to beat that.
    Sample is ~92x larger than a single-ticker null, so it has real power.

(B) WALK-FORWARD NULL on SPY -- the project's existing convention, kept so this
    study is comparable to connors_test.py. Parameters are refit every 90 days on
    the trailing 365, only test-window returns are scored.

KNOWN LIMITATIONS OF THE NULL, stated rather than buried:
  * shuffle_bars rebuilds high/low as max/min of open/close, so shuffled bars
    have NO intrabar range. That shrinks ATR and narrows Donchian channels,
    which makes breakouts EASIER to trigger on shuffled data. For mom_donchian
    and mom_volbreak the null is therefore generous to the null -- i.e. it is a
    harder bar than a like-for-like shuffle would be. Conservative direction.
  * Shuffling preserves each name's drift. A rule whose only content is "be long
    a rising stock" scores well on shuffled data too. That is a feature: it
    strips out the part of the result that is just equity beta.
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import backtest
import strategy
import validate
import mom_data
import mom_strategies as MS

TF = mom_data.TF
SLIP = 0.0005
RUNS = 40


def best_portfolio_sharpe(panel, name) -> tuple[float, dict]:
    """Search the strategy's own grid; return the best equal-weight Sharpe."""
    best, best_c = -np.inf, None
    for c in validate.param_combos(name):
        cols = {t: backtest.run(df, strategy.build(name, **c), timeframe=TF,
                                fee_pct=0.0, slippage_pct=SLIP).returns
                for t, df in panel.items()}
        port = pd.DataFrame(cols).mean(axis=1).dropna()
        sh = backtest.core_metrics(port, TF)["sharpe"]
        if np.isfinite(sh) and sh > best:
            best, best_c = sh, c
    return best, best_c


def pooled_null(panel, name, runs=RUNS, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(runs):
        shuffled = {t: validate.shuffle_bars(df, rng) for t, df in panel.items()}
        sh, _ = best_portfolio_sharpe(shuffled, name)
        out.append(sh)
        print(f"      run {i+1:>2}/{runs}  sharpe {sh:>6.2f}", flush=True)
    return np.array([s for s in out if np.isfinite(s)])


def main():
    panel = mom_data.load()
    print(f"\nPOOLED PANEL PERMUTATION NULL  ({len(panel)} tickers, {RUNS} shuffles, "
          f"5bp/side, full grid search re-run on every shuffle)")
    print("=" * 78)
    rows = []
    for name in MS.MOM_STRATEGIES:
        real, params = best_portfolio_sharpe(panel, name)
        print(f"\n  {name}: real best Sharpe {real:.3f} at {params}", flush=True)
        null = pooled_null(panel, name)
        p = float(np.mean(null >= real))
        rows.append((name, real, params, np.median(null), np.percentile(null, 95), p, len(null)))
        print(f"  --> null median {np.median(null):.3f}  95th {np.percentile(null,95):.3f}"
              f"  p = {p:.3f}", flush=True)

    print("\n\n  SUMMARY -- pooled panel null")
    print("  " + "-" * 74)
    print(f"  {'strategy':<16}{'real Sh':>9}{'null med':>10}{'null 95th':>11}{'p':>8}{'runs':>7}")
    for n, r, _, med, p95, p, k in rows:
        print(f"  {n:<16}{r:>9.3f}{med:>10.3f}{p95:>11.3f}{p:>8.3f}{k:>7}")

    print("\n\nWALK-FORWARD PERMUTATION NULL on SPY / QQQ / IWM "
          f"({RUNS} shuffles, params refit every 90d)")
    print("=" * 78)
    print(f"  {'strategy':<16}{'ticker':<8}{'real OOS Sh':>13}{'null med':>10}"
          f"{'null 95th':>11}{'p':>8}")
    for name in MS.MOM_STRATEGIES:
        for t in ("SPY", "QQQ", "IWM"):
            df = panel[t]
            wf = validate.walk_forward(df, name, timeframe=TF, fee=0.0, slippage=SLIP)
            real = wf.oos_metrics["sharpe"]
            null = validate.permutation_null(df, name, runs=RUNS, timeframe=TF,
                                             fee=0.0, slippage=SLIP)
            p = float(np.mean(null >= real))
            print(f"  {name:<16}{t:<8}{real:>13.2f}{np.median(null):>10.2f}"
                  f"{np.percentile(null,95):>11.2f}{p:>8.3f}", flush=True)


if __name__ == "__main__":
    main()
