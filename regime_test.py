"""Does the regime filter earn its keep?

The filter says: only take a long when SPY's 10sma > 20sma > 50sma and the 10sma
is rising. It has kept the bot flat for nine straight sessions, so it is worth
knowing whether it is protecting the account or just preventing it.

METHOD - an event study, not a backtest. For every (date, ticker) pair in the
universe, evaluate bot.py's checklist using only data available on that date,
then measure what happened next. Split those picks by the regime that was in
force when they were taken. If the filter works, red-regime picks should be
materially worse than green-regime picks.

This isolates the filter. A backtest of the whole strategy would confound the
filter with entry timing, sizing and exits; comparing the same picks under two
regimes does not.

LOOK-AHEAD: every input is shifted so a decision on day t uses only data through
day t, and the outcome measured is strictly after t. The rolling windows are
computed on the full series but only ever read at their own date, which is safe
for trailing windows (mean, max, min) and would not be for anything centred.

SURVIVORSHIP: the universe is today's large caps, so it excludes everything that
failed on the way here. Measured elsewhere in this project at roughly a 5-point
annual overstatement (12.09% -> 6.97% on a dip-buying study). That bias inflates
BOTH arms of this comparison, which is why the comparison is the finding and the
absolute numbers are not.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import watchlist

START = "2007-01-01"      # includes 2008, which is the point
HOLD = 21                 # bot.HOLD_DAYS
MAX_FROM_HIGH = -0.15
MIN_MOMENTUM = 0.10
MAX_EXTENDED = 0.10
MAX_RISK_PCT = 0.08
STOP_BUFFER = 0.02


def load(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(tickers, start=START, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    return close.dropna(how="all")


def regime_series(spy: pd.Series) -> pd.DataFrame:
    """Green/red for every date, on the same rule bot.py applies live."""
    s10, s20, s50 = (spy.rolling(n).mean() for n in (10, 20, 50))
    green = (s10 > s20) & (s20 > s50) & (s10 > s10.shift(5))
    return pd.DataFrame({"green": green, "s10": s10, "s20": s20, "s50": s50})


def checklist(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Boolean matrix of (date x ticker): did this name pass on this date?

    Vectorised across the whole panel rather than looped per name and per day,
    which turns a multi-hour scan into seconds.
    """
    s20 = close.rolling(20).mean()
    s50 = close.rolling(50).mean()
    hi52 = close.rolling(252).max()
    stop = close.rolling(20).min() * (1 - STOP_BUFFER)
    risk = (close - stop) / close
    # Momentum is 12-1: a year of return excluding the most recent month, which
    # is the standard construction and avoids the short-term reversal effect.
    mom = close.shift(21) / close.shift(252) - 1

    passes = ((close > s20) & (s20 > s50)
              & (close / hi52 - 1 >= MAX_FROM_HIGH)
              & (mom >= MIN_MOMENTUM)
              & (close / s20 - 1 <= MAX_EXTENDED)
              & (risk > 0) & (risk <= MAX_RISK_PCT))
    return passes.fillna(False), risk


def forward(close: pd.DataFrame, days: int) -> pd.DataFrame:
    """Return over the next `days` sessions. Strictly after the decision date."""
    return close.shift(-days) / close - 1


def summarise(name: str, r: np.ndarray) -> dict:
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return {"arm": name, "n": len(r), "mean": np.nan, "median": np.nan,
                "hit": np.nan, "t": np.nan, "p5": np.nan}
    return {"arm": name, "n": len(r), "mean": r.mean(), "median": float(np.median(r)),
            "hit": float((r > 0).mean()),
            "t": float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))),
            "p5": float(np.percentile(r, 5))}


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch t for green-minus-red. Unequal variance is the realistic assumption:
    red regimes are more volatile almost by construction."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 30 or len(b) < 30:
        return np.nan, np.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(t), float(df)


def run() -> None:
    tickers = sorted(set(list(watchlist.SECTORS) + ["SPY"]))
    print(f"\n  downloading {len(tickers)} tickers from {START} ...")
    close = load(tickers)
    print(f"  {len(close):,} sessions, {close.shape[1]} names, "
          f"{close.index[0].date()} to {close.index[-1].date()}")

    reg = regime_series(close["SPY"].dropna())
    names = [t for t in watchlist.SECTORS if t in close.columns]
    px = close[names]

    passes, _ = checklist(px)
    green = reg["green"].reindex(px.index).fillna(False)

    print(f"\n  regime was green on {green.mean()*100:.1f}% of sessions "
          f"({int(green.sum()):,} of {len(green):,})")
    print(f"  checklist fired {int(passes.to_numpy().sum()):,} times across the panel")

    rows, arms = [], {}
    for horizon in (5, 10, 21, 63):
        fwd = forward(px, horizon)
        g = fwd.where(passes & green.to_numpy()[:, None]).to_numpy().ravel()
        r = fwd.where(passes & ~green.to_numpy()[:, None]).to_numpy().ravel()
        sg, sr = summarise("green", g), summarise("red", r)
        t, df = welch(g, r)
        rows.append({"horizon": horizon, **{f"g_{k}": v for k, v in sg.items() if k != "arm"},
                     **{f"r_{k}": v for k, v in sr.items() if k != "arm"},
                     "diff": sg["mean"] - sr["mean"], "welch_t": t})
        if horizon == HOLD:
            arms = {"green": g, "red": r}

    print("\n  FORWARD RETURN OF PICKS, BY REGIME AT ENTRY")
    print("  " + "-" * 74)
    print(f"  {'days':>4} {'n green':>8} {'green':>8} {'hit':>6} | "
          f"{'n red':>8} {'red':>8} {'hit':>6} | {'diff':>8} {'Welch t':>8}")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  {r['horizon']:>4} {r['g_n']:>8,} {r['g_mean']*100:>7.2f}% "
              f"{r['g_hit']*100:>5.0f}% | {r['r_n']:>8,} {r['r_mean']*100:>7.2f}% "
              f"{r['r_hit']*100:>5.0f}% | {r['diff']*100:>7.2f}% {r['welch_t']:>8.2f}")

    # Tail risk is the filter's real job. A filter can be neutral on the mean and
    # still be worth keeping if it removes the left tail - which is what actually
    # ruins a small account.
    g, r = arms["green"], arms["red"]
    gf, rf = g[np.isfinite(g)], r[np.isfinite(r)]
    print(f"\n  TAIL AT {HOLD} DAYS  (what the filter is actually for)")
    print("  " + "-" * 74)
    for lab, a in (("green", gf), ("red", rf)):
        print(f"  {lab:>6}  worst {a.min()*100:>7.1f}%   5th pct {np.percentile(a,5)*100:>6.1f}%   "
              f"below -10% {np.mean(a < -0.10)*100:>5.1f}%   below -20% {np.mean(a < -0.20)*100:>4.1f}%")

    # Same question again, but by calendar year, because a single crisis can carry
    # the whole result and that would not be an edge you can rely on.
    print(f"\n  BY YEAR AT {HOLD} DAYS  (is it one crisis, or is it persistent?)")
    print("  " + "-" * 74)
    fwd = forward(px, HOLD)
    gmask, rmask = passes & green.to_numpy()[:, None], passes & ~green.to_numpy()[:, None]
    print(f"  {'year':>6} {'n green':>8} {'green':>8} | {'n red':>8} {'red':>8} | {'diff':>8}")
    wins = 0, 0
    ok, tot = 0, 0
    for yr, idx in fwd.groupby(fwd.index.year).groups.items():
        gy = fwd.loc[idx].where(gmask.loc[idx]).to_numpy().ravel()
        ry = fwd.loc[idx].where(rmask.loc[idx]).to_numpy().ravel()
        gy, ry = gy[np.isfinite(gy)], ry[np.isfinite(ry)]
        if len(gy) < 20 or len(ry) < 20:
            continue
        d = gy.mean() - ry.mean()
        tot += 1
        ok += d > 0
        print(f"  {yr:>6} {len(gy):>8,} {gy.mean()*100:>7.2f}% | "
              f"{len(ry):>8,} {ry.mean()*100:>7.2f}% | {d*100:>7.2f}%")
    print("  " + "-" * 74)
    print(f"  filter helped in {ok} of {tot} years with enough data in both arms")


if __name__ == "__main__":
    run()
