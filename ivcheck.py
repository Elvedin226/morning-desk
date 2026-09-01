"""Replace the guessed implied-vol assumption with a measurement.

screener.vol_study had to assume IV = trailing realized x 1.10. That assumption
is load-bearing: if option markets already price the vol expansion that follows
compression, the apparent edge is just my guess being wrong.

yfinance carries live chains, so the assumption is checkable for free — as a
single snapshot rather than a history. Two questions:

  1. What IS the IV / trailing-realized ratio, really?
  2. Does it rise with compression? If compressed names carry a HIGHER premium,
     the market prices the effect and the edge in vol_study is illusory.

It also measures real bid-ask spreads, the other cost vol_study ignored.

Snapshot caveat: one day, one expiry. Enough to falsify a bad assumption, not
enough to validate a strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

TRADING_YEAR = 252
TARGET_DTE = 30


def atm_quote(ticker: str, spot: float) -> dict | None:
    """ATM implied vol and round-trip spread for the expiry nearest 30 days out."""
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options
        if not expiries:
            return None
        today = pd.Timestamp.utcnow().tz_localize(None).normalize()
        dtes = {e: abs((pd.Timestamp(e) - today).days - TARGET_DTE) for e in expiries}
        expiry = min(dtes, key=dtes.get)
        dte = (pd.Timestamp(expiry) - today).days
        if dte <= 5:
            return None

        chain = tk.option_chain(expiry)
        rows = []
        for side in (chain.calls, chain.puts):
            if side.empty:
                continue
            leg = side.iloc[(side["strike"] - spot).abs().argsort()[:1]]
            r = leg.iloc[0]
            bid, ask = float(r.get("bid", 0) or 0), float(r.get("ask", 0) or 0)
            mid = (bid + ask) / 2
            rows.append({
                "iv": float(r.get("impliedVolatility", np.nan)),
                # Round-trip cost of crossing the spread, as a share of mid. This
                # is paid twice per straddle leg and is what quietly eats the edge.
                "spread_pct": (ask - bid) / mid if mid > 0 else np.nan,
            })
        if not rows:
            return None
        return {
            "iv": float(np.nanmean([r["iv"] for r in rows])),
            "spread_pct": float(np.nanmean([r["spread_pct"] for r in rows])),
            "dte": dte,
        }
    except Exception:
        return None


def compare(close: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    daily = close.pct_change()
    vol_short = (daily.rolling(21).std() * np.sqrt(TRADING_YEAR)).iloc[-1]
    vol_long = (daily.rolling(TRADING_YEAR, min_periods=120).std() * np.sqrt(TRADING_YEAR)).iloc[-1]

    rows = []
    for t in tickers:
        if t not in close.columns or not np.isfinite(vol_short.get(t, np.nan)):
            continue
        q = atm_quote(t, float(close[t].iloc[-1]))
        if not q or not np.isfinite(q["iv"]) or q["iv"] <= 0:
            continue
        rows.append({
            "ticker": t,
            "vol_ratio": vol_short[t] / vol_long[t],
            "realized_21d": vol_short[t],
            "implied": q["iv"],
            "iv_over_realized": q["iv"] / vol_short[t],
            "spread_pct": q["spread_pct"],
            "dte": q["dte"],
        })
    return pd.DataFrame(rows)


def term_structure(ticker: str, spot: float, max_expiries: int = 6) -> pd.DataFrame:
    """ATM implied vol across expiries — is the near or far month richer?"""
    tk = yf.Ticker(ticker)
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    rows = []
    for expiry in tk.options[:max_expiries * 3]:
        dte = (pd.Timestamp(expiry) - today).days
        if dte < 7:
            continue
        try:
            chain = tk.option_chain(expiry)
            leg = chain.calls.iloc[(chain.calls["strike"] - spot).abs().argsort()[:1]].iloc[0]
            bid, ask = float(leg.get("bid", 0) or 0), float(leg.get("ask", 0) or 0)
            mid = (bid + ask) / 2
            rows.append({
                "expiry": expiry,
                "dte": dte,
                "atm_iv": float(leg["impliedVolatility"]),
                "spread_pct": (ask - bid) / mid if mid > 0 else np.nan,
            })
        except Exception:
            continue
        if len(rows) >= max_expiries:
            break
    return pd.DataFrame(rows)


def strike_ladder(ticker: str, spot: float, expiry: str, width: float = 0.18) -> pd.DataFrame:
    """Call strikes around spot with the move each one needs just to break even.

    `breakeven_move` is the honest headline: a call is not profitable when the
    stock rises, it is profitable when the stock rises PAST strike + premium.
    Cheaper strikes need bigger moves — that is the whole trade-off, made visible
    rather than argued about.
    """
    chain = yf.Ticker(ticker).option_chain(expiry).calls
    lo, hi = spot * (1 - width), spot * (1 + width)
    sel = chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)].copy()

    bid = sel["bid"].fillna(0).astype(float)
    ask = sel["ask"].fillna(0).astype(float)
    sel["mid"] = (bid + ask) / 2
    sel["spread_pct"] = np.where(sel["mid"] > 0, (ask - bid) / sel["mid"], np.nan)
    sel["moneyness"] = sel["strike"] / spot - 1
    sel["breakeven"] = sel["strike"] + sel["mid"]
    sel["breakeven_move"] = sel["breakeven"] / spot - 1
    sel["cost_pct_of_spot"] = sel["mid"] / spot
    return sel[["strike", "moneyness", "mid", "spread_pct", "impliedVolatility",
                "breakeven", "breakeven_move", "cost_pct_of_spot"]].reset_index(drop=True)


def _p_finish(spot: float, level: float, iv: float, years: float,
              kind: str, lose: bool = False) -> float:
    """Risk-neutral probability the stock finishes past `level` by expiry.

    Lognormal with r = 0, using the option's own implied vol. These are the
    MARKET's implied odds, not a forecast: they embed the risk premium and are
    only as good as IV is as a vol estimate. Quoting them as "the chance this
    works" would overstate what they are.
    """
    if not np.isfinite(iv) or iv <= 0 or level <= 0 or spot <= 0:
        return float("nan")
    d2 = (np.log(spot / level) - 0.5 * iv ** 2 * years) / (iv * np.sqrt(years))
    above = float(norm.cdf(d2))
    # A call spread profits above the level and is worthless below the long
    # strike; a put spread is the mirror image.
    if kind == "call":
        return 1 - above if lose else above
    return above if lose else 1 - above


def _bs_call(spot: float, strike: float, iv: float, years: float) -> float:
    """Black-Scholes call value, r = 0."""
    if iv <= 0 or years <= 0 or strike <= 0:
        return max(spot - strike, 0.0)
    d1 = (np.log(spot / strike) + 0.5 * iv ** 2 * years) / (iv * np.sqrt(years))
    d2 = d1 - iv * np.sqrt(years)
    return float(spot * norm.cdf(d1) - strike * norm.cdf(d2))


def _bs_put(spot: float, strike: float, iv: float, years: float) -> float:
    """Put via put-call parity at r = 0."""
    return _bs_call(spot, strike, iv, years) - spot + strike


def _spread_ev(spot, k1, k2, iv, years, kind, debit_real) -> float:
    """Risk-neutral expected value of the spread, in dollars, net of what you pay.

    The fair value of a vertical is the difference between the two legs' option
    values. Under risk-neutral pricing that fair value IS the break-even price,
    so EV collapses to (fair value - what you actually paid) — which is to say,
    the negative of your slippage. This is the number that shows every spread on
    a board is the same bet wearing different clothes.
    """
    if not np.isfinite(iv) or iv <= 0:
        return float("nan")
    price = _bs_call if kind == "call" else _bs_put
    fair = abs(price(spot, k1, iv, years) - price(spot, k2, iv, years))
    return fair * 100 - debit_real


def spread_ladder(
    ticker: str,
    spot: float,
    expiry: str,
    kind: str = "call",
    widths: tuple[float, ...] = (5, 10, 20, 30),
    max_debit: float | None = None,
) -> pd.DataFrame:
    """Debit vertical spreads — bullish with calls, bearish with puts.

        call: buy strike K, sell K + width   -> profits as the stock RISES
        put:  buy strike K, sell K - width   -> profits as the stock FALLS

    The point of the structure is outlay. You give up everything beyond the short
    strike and in exchange pay a fraction of the naked option's premium, which is
    what brings positions inside a small account's sizing rule.

    Two debits are reported because the difference is not academic. `debit_mid`
    assumes you fill at the midpoint of both legs; `debit_real` is what you pay
    crossing the spread on each leg (ask on the long, bid on the short). A
    two-legged trade pays the spread twice, and on thin chains debit_real can be
    20-30% worse than debit_mid — which comes straight out of max profit.
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")

    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    years = max((pd.Timestamp(expiry) - today).days, 1) / 365

    chains = yf.Ticker(ticker).option_chain(expiry)
    chain = (chains.calls if kind == "call" else chains.puts).copy()
    chain["bid"] = chain["bid"].fillna(0).astype(float)
    chain["ask"] = chain["ask"].fillna(0).astype(float)
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2
    quotes = chain.set_index("strike")

    longs = chain[(chain["strike"] >= spot * 0.90) & (chain["strike"] <= spot * 1.10)]
    rows = []
    for _, leg in longs.iterrows():
        k1 = float(leg["strike"])
        for w in widths:
            # A call spread sells the strike above; a put spread sells below.
            k2 = k1 + w if kind == "call" else k1 - w
            if k2 not in quotes.index:
                continue
            short = quotes.loc[k2]
            debit_mid = float(leg["mid"] - short["mid"])
            debit_real = float(leg["ask"] - short["bid"])
            if debit_mid <= 0 or debit_real <= 0:
                continue
            max_profit = w - debit_real
            if max_profit <= 0:
                continue
            # Calls break even above the long strike, puts below it.
            breakeven = k1 + debit_real if kind == "call" else k1 - debit_real
            iv = float(leg.get("impliedVolatility", np.nan))
            rows.append({
                "long": k1,
                "short": k2,
                "width": w,
                "debit_mid": debit_mid * 100,
                "debit_real": debit_real * 100,
                "max_profit": max_profit * 100,
                "rr": max_profit / debit_real,
                "breakeven": breakeven,
                "needs_move": breakeven / spot - 1,
                "slippage_pct": (debit_real - debit_mid) / debit_mid,
                # Market-implied odds, from the long leg's own IV. Total loss is
                # decided by the LONG strike alone — the short leg changes cost and
                # caps upside, it does not change whether you lose everything.
                "p_loss": _p_finish(spot, k1, iv, years, kind, lose=True),
                "p_profit": _p_finish(spot, breakeven, iv, years, kind),
                "p_max": _p_finish(spot, k2, iv, years, kind),
                "ev": _spread_ev(spot, k1, k2, iv, years, kind, debit_real * 100),
            })

    df = pd.DataFrame(rows)
    if max_debit is not None and not df.empty:
        df = df[df["debit_real"] <= max_debit]
    return df.sort_values(["long", "width"]).reset_index(drop=True)
