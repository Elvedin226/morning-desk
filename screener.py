"""Large-move screener + event studies on setups worth testing.

Modes:

  (no args)        dip event study + current screen
  --ticker AAPL    full readout for one name, including the live option chain

Read the studies before trusting the screen. A screener always produces
candidates; that is all it does. Whether they go on to outperform is a separate
question and the only one that matters.

SURVIVORSHIP BIAS — the trap that dominates the dip study:
The universe below is companies that exist TODAY. Testing "buy quality stocks
after a crash" on survivors is close to rigged: the ones that crashed and never
recovered were delisted and are absent. Measured, not assumed — adding 22
recoverable casualties cut the 126-day edge from 12.09% to 6.97%, and the true
figure is lower still, because the actual zeros (BBBY, SIVB, WeWork) return no
data at all from free providers. Treat every dip number here as an upper bound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

import ivcheck

# Liquid large/mid caps across sectors, plus names that came up in discussion.
# Not an index — see the survivorship warning above.
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "MRVL", "AMD", "INTC",
    "MU", "QCOM", "TXN", "ADBE", "CRM", "ORCL", "CSCO", "IBM", "NOW", "PANW",
    "JPM", "BAC", "GS", "MS", "WFC", "V", "MA", "AXP", "SCHW", "BLK",
    "JNJ", "PFE", "MRK", "LLY", "ABBV", "UNH", "TMO", "ABT", "BMY", "GILD",
    "XOM", "CVX", "COP", "SLB", "OXY", "PG", "KO", "PEP", "WMT", "COST",
    "HD", "MCD", "NKE", "SBUX", "TGT", "DIS", "NFLX", "T", "VZ", "CMCSA",
    "BA", "CAT", "DE", "GE", "LMT", "RTX", "HON", "UPS", "FDX", "F",
    "TSLA", "GM", "UBER", "ABNB", "SHOP", "PYPL", "COIN", "PLTR", "SOFI",
    "ASTS", "RKLB", "LUNR", "ACHR", "IONQ", "SMCI", "ARM", "DELL", "WDC", "STX",
]

LOOKBACK_HIGH = 252
HORIZONS = (21, 63, 126)
TRADING_YEAR = 252

# An ATM straddle costs roughly 0.8 x IV x sqrt(T) of spot.
STRADDLE_COST_FACTOR = 0.8
# Superseded by live measurement — ivcheck puts the real ratio near 1.17 for
# compressed names and 0.76 for expanded ones. Kept only so vol_study's
# `breakeven` column stays comparable to the original run.
IV_OVER_REALIZED = 1.10


def load_universe(tickers=UNIVERSE, start="2010-01-01") -> pd.DataFrame:
    raw = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    return close.dropna(how="all")


def metrics(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    high_252 = close.rolling(LOOKBACK_HIGH, min_periods=60).max()
    ma200 = close.rolling(200, min_periods=60).mean()
    daily = close.pct_change()
    return {
        "drawdown": close / high_252 - 1,
        "vs_200dma": close / ma200 - 1,
        # Short-run vol against its own long-run level. Low values precede
        # expansion, which is what a large move requires.
        "vol_ratio": daily.rolling(21).std() / daily.rolling(TRADING_YEAR, min_periods=60).std(),
        # 12-1 momentum, standard academic construction (skips the last month).
        "momentum": close.shift(21) / close.shift(TRADING_YEAR) - 1,
    }


def screen(close: pd.DataFrame, min_drawdown: float = -0.30) -> pd.DataFrame:
    m = metrics(close)
    latest = pd.DataFrame({k: v.iloc[-1] for k, v in m.items()})
    latest["price"] = close.iloc[-1]
    return latest[latest["drawdown"] <= min_drawdown].sort_values("drawdown")


def dip_study(close: pd.DataFrame, threshold: float = -0.30, cooldown: int = 126) -> pd.DataFrame:
    """After a stock first falls `threshold` below its 52-week high, what next —
    and is that better than a random day in the same sample?

    `cooldown` stops one long drawdown registering as hundreds of events.
    """
    dd = metrics(close)["drawdown"]
    forward = {h: close.shift(-h) / close - 1 for h in HORIZONS}

    events: list[dict] = []
    for ticker in close.columns:
        series = dd[ticker].dropna()
        triggered = series <= threshold
        last_event = -10**9
        for i, (date, hit) in enumerate(zip(series.index, triggered.to_numpy())):
            if not hit or i - last_event < cooldown:
                continue
            last_event = i
            row = {"ticker": ticker, "date": date}
            for h in HORIZONS:
                row[h] = forward[h][ticker].get(date, np.nan)
            events.append(row)

    ev = pd.DataFrame(events)
    rows = []
    for h in HORIZONS:
        after = ev[h].dropna()
        # Baseline: every stock-day, same horizon. "Up 8% in 3 months" means
        # nothing until you know the average stock-day was up 6%.
        base = forward[h].to_numpy().ravel()
        base = base[np.isfinite(base)]
        se = after.std() / np.sqrt(len(after)) if len(after) > 1 else np.nan
        rows.append({
            "horizon_days": h,
            "events": len(after),
            "after_dip_mean": after.mean(),
            "baseline_mean": base.mean(),
            "edge": after.mean() - base.mean(),
            "t_stat": (after.mean() - base.mean()) / se if se and se > 0 else np.nan,
            "win_rate": (after > 0).mean(),
            "baseline_win": (base > 0).mean(),
        })
    return pd.DataFrame(rows)


def vol_panel(close: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Long-format panel of (compression now -> what happened next).

    The trap: a depressed 21-day vol mean-reverts upward almost by construction,
    so "compression predicts expansion" proves nothing tradeable on its own.
    Option sellers price it too. The only question is whether the move clears
    the premium, which is what `breakeven` tests.
    """
    daily = close.pct_change()
    vol_short = daily.rolling(21).std()
    vol_long = daily.rolling(TRADING_YEAR, min_periods=120).std()

    panel = pd.DataFrame({
        "ratio": (vol_short / vol_long).stack(),
        "vol_now": vol_short.stack(),
        "vol_next": vol_short.shift(-horizon).stack(),
        "fwd_return": (close.shift(-horizon) / close - 1).stack(),
    }).replace([np.inf, -np.inf], np.nan).dropna()

    panel["abs_move"] = panel["fwd_return"].abs()
    panel["expansion"] = panel["vol_next"] / panel["vol_now"]
    implied = panel["vol_now"] * np.sqrt(TRADING_YEAR) * IV_OVER_REALIZED
    panel["breakeven"] = STRADDLE_COST_FACTOR * implied * np.sqrt(horizon / TRADING_YEAR)
    panel["cleared"] = panel["abs_move"] > panel["breakeven"]
    return panel


def vol_study(close: pd.DataFrame, horizon: int = 21, buckets: int = 5) -> pd.DataFrame:
    panel = vol_panel(close, horizon)
    panel["bucket"] = pd.qcut(panel["ratio"], buckets, labels=False)
    rows = []
    for b in range(buckets):
        g = panel[panel["bucket"] == b]
        rows.append({
            "bucket": b + 1,
            "n": len(g),
            "vol_ratio": g["ratio"].mean(),
            "expansion": g["expansion"].mean(),
            "abs_move": g["abs_move"].mean(),
            "breakeven": g["breakeven"].mean(),
            "cleared": g["cleared"].mean(),
            "fwd_return": g["fwd_return"].mean(),
        })
    return pd.DataFrame(rows)


def spread_report(close: pd.DataFrame, ticker: str, expiry: str | None = None,
                  budget: float | None = None, kind: str = "call") -> None:
    """Debit vertical spreads for one name, optionally filtered to what a budget buys."""
    if ticker not in close.columns:
        close = load_universe(tickers=list(UNIVERSE) + [ticker])
    spot = float(close[ticker].iloc[-1])

    ts = ivcheck.term_structure(ticker, spot, max_expiries=10)
    if ts.empty:
        raise SystemExit(f"No option chains for {ticker}.")
    if expiry is None:
        # Default to ~3 months: long enough that decay is not the dominant term,
        # short enough that the capital is not tied up for a year.
        expiry = ts.iloc[(ts["dte"] - 90).abs().argsort()[:1]].iloc[0]["expiry"]
    dte = int(ts.set_index("expiry").loc[expiry, "dte"]) if expiry in set(ts["expiry"]) else None

    df = ivcheck.spread_ladder(ticker, spot, expiry, kind=kind, max_debit=budget)
    if df.empty:
        msg = f" under ${budget:,.0f}" if budget else ""
        raise SystemExit(f"No priceable {kind} spreads for {ticker} {expiry}{msg}.")

    label = "bull call spreads (profit if it RISES)" if kind == "call" \
        else "bear put spreads (profit if it FALLS)"
    header = f"=== {ticker} @ {spot:,.2f}  {label}  {expiry}"
    header += f" ({dte}d)" if dte else ""
    if budget:
        header += f"  [budget ${budget:,.0f}]"
    print()
    print(header + " ===")
    print()
    print(f"  {'long':>6}{'short':>7}{'debit':>8}{'slip':>6}"
          f"{'max':>8}{'R:R':>6}{'breakeven':>11}{'needs':>8}"
          f"{'P(loss)':>9}{'P(win)':>8}{'P(max)':>8}")
    for _, r in df.iterrows():
        pct = lambda v: f"{v*100:.0f}%" if np.isfinite(v) else "-"
        print(f"  {r['long']:>6.0f}{r['short']:>7.0f}{r['debit_real']:>8,.0f}"
              f"{r['slippage_pct']*100:>5.0f}%"
              f"{r['max_profit']:>8,.0f}{r['rr']:>6.2f}{r['breakeven']:>11.2f}"
              f"{r['needs_move']*100:>+7.1f}%"
              f"{pct(r['p_loss']):>9}{pct(r['p_profit']):>8}{pct(r['p_max']):>8}")
    print()
    print("  debit = what you pay crossing both legs (ask on long, bid on short).")
    print("  max profit and R:R are computed off that real debit, not the midpoint.")
    print(f"  needs = move required by expiry to break even "
          f"({'up' if kind == 'call' else 'down'}side).")
    print(f"  ALL rows above are the {expiry} expiry"
          + (f" ({dte} days out)." if dte else "."))
    print("  P(loss) is set by the LONG strike alone - the short leg cuts cost and")
    print("  caps upside, it does not change your odds of losing the whole debit.")
    print("  Risk-neutral odds from the option's own IV, not a forecast.")


def ticker_report(close: pd.DataFrame, ticker: str) -> None:
    """Everything the systematic criteria say about one name, in one place."""
    if ticker not in close.columns:
        close = load_universe(tickers=list(UNIVERSE) + [ticker])
    if ticker not in close.columns:
        raise SystemExit(f"No data for {ticker}.")

    m = metrics(close)
    row = {k: v[ticker].iloc[-1] for k, v in m.items()}
    spot = float(close[ticker].iloc[-1])
    realized = float((close[ticker].pct_change().rolling(21).std() * np.sqrt(TRADING_YEAR)).iloc[-1])

    panel = vol_panel(close, 21)
    edges = pd.qcut(panel["ratio"], 5, retbins=True)[1]
    bucket = int(np.searchsorted(edges[1:-1], row["vol_ratio"])) + 1
    label = "compressed" if bucket <= 2 else "expanded" if bucket >= 4 else "middle"

    print()
    print(f"=== {ticker} @ {spot:,.2f} ===")
    print()
    print(f"  drawdown from 52w high   {row['drawdown']*100:>8.1f}%   (dip study triggers at -30%)")
    print(f"  vs 200dma                {row['vs_200dma']*100:>8.1f}%")
    print(f"  momentum 12-1            {row['momentum']*100:>8.1f}%")
    print(f"  vol ratio                {row['vol_ratio']:>8.2f}    bucket {bucket}/5 ({label})")
    print(f"  realized vol 21d         {realized*100:>8.1f}%")

    ts = ivcheck.term_structure(ticker, spot)
    if ts.empty:
        print()
        print("  no option chains available")
        return

    print()
    print("  IV term structure (ATM):")
    print(f"    {'expiry':<13}{'dte':>5}{'IV':>8}{'IV/realized':>13}{'spread':>9}")
    for _, r in ts.iterrows():
        print(f"    {r['expiry']:<13}{int(r['dte']):>5}{r['atm_iv']*100:>7.1f}%"
              f"{r['atm_iv']/realized:>13.2f}{r['spread_pct']*100:>8.1f}%")

    target = ts.iloc[(ts["dte"] - 45).abs().argsort()[:1]].iloc[0]
    ladder = ivcheck.strike_ladder(ticker, spot, target["expiry"])
    print()
    print(f"  call ladder, {target['expiry']} ({int(target['dte'])}d):")
    print(f"    {'strike':>8}{'moneyness':>11}{'mid':>9}{'spread':>9}{'IV':>8}"
          f"{'breakeven':>11}{'needs move':>12}{'cost/spot':>11}")
    for _, r in ladder.iterrows():
        if not np.isfinite(r["mid"]) or r["mid"] <= 0:
            continue
        print(f"    {r['strike']:>8.1f}{r['moneyness']*100:>10.1f}%{r['mid']:>9.2f}"
              f"{r['spread_pct']*100:>8.1f}%{r['impliedVolatility']*100:>7.1f}%"
              f"{r['breakeven']:>11.2f}{r['breakeven_move']*100:>11.1f}%{r['cost_pct_of_spot']*100:>10.1f}%")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Large-move screener and setup studies.")
    ap.add_argument("--ticker", help="full readout for one name (metrics + live option chain)")
    ap.add_argument("--spreads", action="store_true",
                    help="with --ticker: price debit vertical spreads instead of naked options")
    ap.add_argument("--puts", action="store_true",
                    help="with --spreads: bear PUT spreads (downside) instead of bull call spreads")
    ap.add_argument("--budget", type=float,
                    help="with --spreads: only show spreads costing at most this many dollars")
    ap.add_argument("--expiry", help="with --spreads: specific expiry, e.g. 2027-01-15")
    ap.add_argument("--min-drawdown", type=float, default=-0.30,
                    help="screen threshold, e.g. -0.30 for 30%% below the 52-week high")
    args = ap.parse_args()

    close = load_universe()

    if args.ticker and args.spreads:
        spread_report(close, args.ticker.upper(), args.expiry, args.budget,
                      kind="put" if args.puts else "call")
        raise SystemExit(0)

    if args.ticker:
        ticker_report(close, args.ticker.upper())
        raise SystemExit(0)

    print(f"universe: {close.shape[1]} tickers, {close.index[0].date()} -> {close.index[-1].date()}")
    print()
    print("=== EVENT STUDY: buying after a >=30% drawdown from the 52-week high ===")
    print("(survivorship-biased upward - see module docstring)")
    print()
    for _, r in dip_study(close).iterrows():
        print(f"  {int(r['horizon_days']):>4}d  n={int(r['events']):<4} "
              f"after dip {r['after_dip_mean']*100:>7.2f}%   "
              f"baseline {r['baseline_mean']*100:>6.2f}%   "
              f"edge {r['edge']*100:>+7.2f}%   t={r['t_stat']:>5.2f}   "
              f"win {r['win_rate']*100:>5.1f}% vs {r['baseline_win']*100:.1f}%")

    print()
    print(f"=== CURRENT SCREEN: >={abs(args.min_drawdown)*100:.0f}% below 52-week high ===")
    print()
    print(f"  {'ticker':<9}{'price':>10}{'drawdown':>11}{'vs 200dma':>12}{'vol ratio':>11}{'mom 12-1':>11}")
    for t, r in screen(close, args.min_drawdown).iterrows():
        print(f"  {t:<9}{r['price']:>10.2f}{r['drawdown']*100:>10.1f}%{r['vs_200dma']*100:>11.1f}%"
              f"{r['vol_ratio']:>11.2f}{r['momentum']*100:>10.1f}%")
