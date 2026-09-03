"""Test Connors RSI-2 the same way everything else in this project was tested.

The claim (Connors & Alvarez, published 2008-2011): buy a 2-period RSI below 5
inside a 200-day uptrend, exit above the 5-day average. Reported 75-85% win
rates on indices, mid-1990s to 2010.

Three things have to be true for that to matter here:

  1. It must still work AFTER publication. McLean & Pontiff measured 58% decay
     in published anomalies post-publication. This one has been in every screener
     for fifteen years.
  2. It must beat its own shuffled-bars null. A high win rate is not an edge -
     a strategy that takes tiny profits and rare huge losses wins 85% of the
     time and still loses money.
  3. It must survive costs. One survivor of eight strategies tested in this
     project died exactly there.

So: split at publication, run the permutation null, and charge realistic costs.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import backtest
import strategy
import validate

# Index ETFs (what Connors tested) plus liquid large caps (what the bot trades).
UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "JNJ", "KO", "PG",
            "JPM", "XOM", "WMT", "HD", "MRK", "CVX"]

FEE = 0.0            # commission-free broker
SLIP = 0.0005        # 5bp per side, same as the paper account


def bars(ticker: str, start="1995-01-01") -> pd.DataFrame:
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        return raw
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    return df.dropna()


def summarise(df: pd.DataFrame, name: str, label: str) -> dict:
    st = strategy.build(name)
    r = backtest.run(df, st, fee_pct=FEE, slippage_pct=SLIP)
    m = r.metrics
    bh = backtest.run(df, strategy.build("buy_and_hold"), fee_pct=FEE, slippage_pct=SLIP)
    return {"window": label, "bars": len(df), "trades": m.get("num_trades", 0),
            "exposure": m.get("exposure"),
            "cagr": m.get("cagr"), "sharpe": m.get("sharpe"),
            "maxdd": m.get("max_drawdown"), "winrate": m.get("win_rate"),
            "bh_cagr": bh.metrics.get("cagr")}


def main() -> None:
    print("\n  CONNORS RSI-2  ::  in-sample vs post-publication")
    print("  Connors' tested window ended ~2010; the strategy was published")
    print("  2008-2011. Everything after that is out-of-sample for the WORLD,")
    print("  not just for me - which is the only split that matters here.")
    print("  " + "=" * 86)
    print(f"  {'ticker':<7}{'window':<14}{'trades':>7}{'CAGR':>9}{'B&H':>9}"
          f"{'Sharpe':>8}{'maxDD':>8}{'win':>7}{'in mkt':>8}")
    print("  " + "-" * 86)

    pre_rows, post_rows = [], []
    for t in UNIVERSE:
        df = bars(t)
        if df.empty or len(df) < 1500:
            continue
        pre = df[df.index < "2011-01-01"]
        post = df[df.index >= "2011-01-01"]
        for label, d, bucket in (("pre-2011", pre, pre_rows), ("2011-now", post, post_rows)):
            if len(d) < 500:
                continue
            s = summarise(d, "connors_rsi2", label)
            bucket.append(s)
            print(f"  {t:<7}{label:<14}{s['trades']:>7}"
                  f"{s['cagr']*100:>8.2f}%{s['bh_cagr']*100:>8.2f}%"
                  f"{s['sharpe']:>8.2f}{s['maxdd']*100:>7.1f}%{s['winrate']*100:>6.0f}%{s['exposure']*100:>7.0f}%")
        print("  " + "-" * 86)

    for label, rows in (("PRE-2011 (Connors' own window)", pre_rows),
                        ("2011-NOW (post-publication)", post_rows)):
        if not rows:
            continue
        cagr = np.array([r["cagr"] for r in rows], dtype=float)
        bh = np.array([r["bh_cagr"] for r in rows], dtype=float)
        sh = np.array([r["sharpe"] for r in rows], dtype=float)
        wr = np.array([r["winrate"] for r in rows], dtype=float)
        beat = np.mean(cagr > bh)
        print(f"\n  {label}   n={len(rows)}")
        print(f"     median CAGR {np.nanmedian(cagr)*100:>6.2f}%   "
              f"vs buy & hold {np.nanmedian(bh)*100:>6.2f}%   "
              f"beat B&H in {beat*100:.0f}% of names")
        print(f"     median Sharpe {np.nanmedian(sh):>5.2f}   "
              f"median win rate {np.nanmedian(wr)*100:.0f}%")

    # The win rate is the seduction. Check what it is actually worth by asking
    # whether the strategy beats its own shuffled prices out of sample.
    print("\n\n  PERMUTATION NULL ON SPY, 2011-NOW")
    print("  Shuffle the bars to destroy any real structure, re-run the whole")
    print("  walk-forward, and see where the real result lands in that")
    print("  distribution. A strategy that cannot beat its own noise has no edge")
    print("  no matter how high the win rate looks.")
    print("  " + "=" * 86)
    spy = bars("SPY")
    spy_post = spy[spy.index >= "2011-01-01"]
    wf = validate.walk_forward(spy_post, "connors_rsi2", fee=FEE, slippage=SLIP)
    real = wf.oos_metrics["sharpe"]
    null = validate.permutation_null(spy_post, "connors_rsi2", runs=40,
                                     fee=FEE, slippage=SLIP)
    pval = float(np.mean(null >= real))
    print(f"     real out-of-sample Sharpe   {real:>6.2f}")
    print(f"     shuffled median / 95th pct  {np.median(null):>6.2f} / "
          f"{np.percentile(null, 95):>5.2f}   ({len(null)} runs)")
    print(f"     p-value                     {pval:>6.2f}"
          f"   {'<-- indistinguishable from noise' if pval > 0.10 else '<-- survives'}")

    # Costs killed the one survivor of the earlier eight. Find the break-even.
    print("\n\n  COST SENSITIVITY, SPY 2011-NOW  (where does the edge die?)")
    print("  " + "=" * 86)
    st = strategy.build("connors_rsi2")
    for slip in (0.0, 0.0002, 0.0005, 0.0010, 0.0020, 0.0050):
        r = backtest.run(spy_post, st, fee_pct=0.0, slippage_pct=slip)
        print(f"     slippage {slip*10000:>4.0f} bp/side   "
              f"CAGR {r.metrics['cagr']*100:>7.2f}%   "
              f"Sharpe {r.metrics['sharpe']:>5.2f}   "
              f"trades {r.metrics['num_trades']:>4}")
    bh = backtest.run(spy_post, strategy.build("buy_and_hold"), fee_pct=0.0, slippage_pct=0.0)
    print(f"     buy & hold (no costs)      CAGR {bh.metrics['cagr']*100:>7.2f}%   "
          f"Sharpe {bh.metrics['sharpe']:>5.2f}")


if __name__ == "__main__":
    main()
