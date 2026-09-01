"""High tight flag / volatility-contraction breakout — encoded and tested.

From the @jv_trading walkthrough; the setup itself is the Minervini/O'Neil VCP
breakout, which has decades of history behind it and has never, as far as I can
find, been published with an out-of-sample test.

Every rule below is mechanical, which is why this one is testable at all:

    market regime   SPY 10 > 20 > 50 SMA, stacked and rising
    prior move      up PRIOR_GAIN over PRIOR_WINDOW days
    pullback        close pulled back to the 10/20 MA zone recently
    contraction     recent ATR compressed vs its own longer average ("the coil")
    ADR             average daily range above ADR_MIN (he says 5%)
    volume taper    5-day volume below its 20-day average
    breakout        close takes out the consolidation high

WHY NOT --walkforward: that validator fits parameters on one asset's history and
scores the next window. This strategy is cross-sectional — it ranks many stocks
and trades a handful — so a single-asset equity curve would not represent it. The
honest analogue is an event study (does the trigger beat baseline?) plus a
permutation null (does the same detector find "edges" in shuffled prices?).

THE COMPARISON THAT MATTERS: the setup only fires in bullish regimes, and stocks
drift up in bullish regimes. Measured against all stock-days it would look good
for that reason alone. So the baseline below is restricted to days when the
market filter was ALREADY on — isolating the setup from the regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

HORIZONS = (5, 10, 21, 63)

# Setup parameters. Deliberately close to what the video states, and coarse —
# tuning these against the result is how you fit noise.
PRIOR_WINDOW = 40      # ~2 months for the "big move up"
PRIOR_GAIN = 0.25      # how much it must have run
PULLBACK_WINDOW = 15   # look-back for the pullback to the MAs
CONTRACTION_MAX = 0.85 # ATR(10) / ATR(40); below 1 means the range is tightening
ADR_MIN = 0.04         # he says 5%; 4% keeps the sample usable
BREAKOUT_WINDOW = 10   # close must exceed the high of this many prior days


def load_panel(tickers: list[str], start: str = "2010-01-01") -> dict[str, pd.DataFrame]:
    """OHLCV per ticker. The close-only loader in screener.py is not enough here —
    ADR, ATR and volume taper all need the full bar."""
    raw = yf.download(tickers, start=start, auto_adjust=True,
                      progress=False, group_by="ticker")
    out = {}
    for t in tickers:
        try:
            df = raw[t].dropna() if isinstance(raw.columns, pd.MultiIndex) else raw.dropna()
        except KeyError:
            continue
        if len(df) > 300:
            out[t] = df.rename(columns=str.lower)
    return out


def market_filter(spy: pd.DataFrame) -> pd.Series:
    """Bullish regime: the three averages stacked, and the fastest one rising."""
    c = spy["close"]
    s10, s20, s50 = c.rolling(10).mean(), c.rolling(20).mean(), c.rolling(50).mean()
    return (s10 > s20) & (s20 > s50) & (s10 > s10.shift(5))


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def detect(df: pd.DataFrame) -> pd.Series:
    """Bars where every condition holds AND the breakout happens.

    All inputs are known at the close of the signal bar; entry is modelled at the
    NEXT bar's open, same convention the backtest engine uses.
    """
    c, v = df["close"], df["volume"]
    s10, s20 = c.rolling(10).mean(), c.rolling(20).mean()

    prior_move = c / c.shift(PRIOR_WINDOW) - 1 > PRIOR_GAIN

    # Pullback: at some point recently the close sat at or under the 10-day MA,
    # which is what separates a flag from a stock running away uninterrupted.
    touched = (c <= s10 * 1.01).rolling(PULLBACK_WINDOW).max().astype(bool)

    # Still structurally intact — holding above the 20-day.
    intact = c > s20

    contraction = _atr(df, 10) / _atr(df, 40) < CONTRACTION_MAX
    adr = ((df["high"] / df["low"] - 1).rolling(20).mean()) > ADR_MIN
    taper = v.rolling(5).mean() < v.rolling(20).mean()

    prior_high = df["high"].shift(1).rolling(BREAKOUT_WINDOW).max()
    breakout = c > prior_high

    return (prior_move & touched & intact & contraction & adr & taper & breakout).fillna(False)


def study(panel: dict[str, pd.DataFrame], regime: pd.Series,
          cooldown: int = 10) -> tuple[pd.DataFrame, int]:
    """Forward returns after a trigger, against a same-regime baseline."""
    hits: dict[int, list] = {h: [] for h in HORIZONS}
    base: dict[int, list] = {h: [] for h in HORIZONS}
    n_events = 0

    for _, df in panel.items():
        sig = detect(df)
        reg = regime.reindex(df.index).ffill().fillna(False)
        sig = sig & reg

        # Entry at the next open; exit at the open h bars later. Same fill
        # convention as backtest.py, so results are comparable.
        entry = df["open"].shift(-1)
        for h in HORIZONS:
            fwd = df["open"].shift(-1 - h) / entry - 1
            # Baseline is every bar in the SAME regime, not every bar overall.
            base[h].extend(fwd[reg].dropna().to_numpy())

        idx = np.flatnonzero(sig.to_numpy())
        last = -10**9
        for i in idx:
            if i - last < cooldown:
                continue
            last = i
            n_events += 1
            for h in HORIZONS:
                val = (df["open"].iloc[i + 1 + h] / df["open"].iloc[i + 1] - 1
                       if i + 1 + h < len(df) else np.nan)
                if np.isfinite(val):
                    hits[h].append(val)

    rows = []
    for h in HORIZONS:
        a, b = np.array(hits[h]), np.array(base[h])
        b = b[np.isfinite(b)]
        se = a.std() / np.sqrt(len(a)) if len(a) > 1 else np.nan
        rows.append({
            "horizon": h,
            "n": len(a),
            "setup": a.mean() if len(a) else np.nan,
            "baseline": b.mean() if len(b) else np.nan,
            "edge": (a.mean() - b.mean()) if len(a) and len(b) else np.nan,
            "t_stat": (a.mean() - b.mean()) / se if se and se > 0 else np.nan,
            "win": (a > 0).mean() if len(a) else np.nan,
            "base_win": (b > 0).mean() if len(b) else np.nan,
        })
    return pd.DataFrame(rows), n_events


def shuffle_panel(panel: dict[str, pd.DataFrame], rng) -> dict[str, pd.DataFrame]:
    """Rebuild each series from the same bars in random order.

    Preserves each bar's internal shape and the return distribution, destroys the
    across-bar structure the setup claims to read. Whatever edge the detector
    still finds is what it manufactures from nothing.
    """
    out = {}
    for t, df in panel.items():
        o, c = df["open"].to_numpy(), df["close"].to_numpy()
        h, l, v = df["high"].to_numpy(), df["low"].to_numpy(), df["volume"].to_numpy()
        gap, body = o[1:] / c[:-1], c[1:] / o[1:]
        up = np.divide(h[1:], np.maximum(o[1:], c[1:]),
                       out=np.ones(len(o) - 1), where=np.maximum(o[1:], c[1:]) > 0)
        dn = np.divide(l[1:], np.minimum(o[1:], c[1:]),
                       out=np.ones(len(o) - 1), where=np.minimum(o[1:], c[1:]) > 0)

        k = rng.permutation(len(gap))
        gap, body, up, dn, vol = gap[k], body[k], up[k], dn[k], v[1:][k]

        nc = c[0] * np.concatenate([[1.0], np.cumprod(gap * body)])
        no = np.concatenate([[o[0]], nc[:-1] * gap])
        nh = np.concatenate([[h[0]], np.maximum(no[1:], nc[1:]) * up])
        nl = np.concatenate([[l[0]], np.minimum(no[1:], nc[1:]) * dn])
        out[t] = pd.DataFrame(
            {"open": no, "high": nh, "low": nl, "close": nc,
             "volume": np.concatenate([[v[0]], vol])},
            index=df.index,
        )
    return out
