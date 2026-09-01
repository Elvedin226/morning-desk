"""Catalyst calendar: scheduled events, and whether options price them fairly.

This is the tradeable version of the "buy puts on the Boeing headline" idea.
Reacting to unscheduled news is structurally too late — by the time it is a
headline the move and the implied vol have both already happened. Earnings are
different: the DATE is known in advance, only the magnitude is uncertain, so a
position can exist before the event rather than after it.

The analysis that matters is not "when is earnings" — any calendar gives that.
It is whether the option market is charging more or less than the stock has
historically moved:

    implied move    = ATM straddle cost / spot, for the expiry covering the event
    historical move = mean |return| on past earnings reaction days

implied > historical means you are paying above the historical average for the
event. That is the normal state of affairs — earnings straddles are usually rich,
which is why "IV crush" is a cliche — so the interesting rows are the exceptions.

Honest limits: historical realized moves are free, historical IMPLIED moves are
not, so this compares today's pricing against history rather than comparing like
with like. collect.py is accumulating the missing half.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Announcements timestamped at or after this hour are after the close, so the
# market reaction lands on the FOLLOWING session.
AFTER_HOURS_CUTOFF = 12


def earnings_dates(ticker: str, limit: int = 24) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def reaction_moves(ticker: str, close: pd.Series, limit: int = 24) -> pd.Series:
    """Absolute return on each past earnings reaction day."""
    ed = earnings_dates(ticker, limit)
    if ed.empty:
        return pd.Series(dtype=float)

    daily = close.pct_change().abs()
    idx = daily.index.tz_localize(None) if daily.index.tz is not None else daily.index
    daily = pd.Series(daily.to_numpy(), index=idx)

    moves = []
    for ts in ed.index:
        stamp = pd.Timestamp(ts)
        day = stamp.tz_localize(None).normalize() if stamp.tz is not None else stamp.normalize()
        # After-hours release -> the reaction is the next session.
        target = day + pd.Timedelta(days=1) if stamp.hour >= AFTER_HOURS_CUTOFF else day
        future = daily.index[daily.index >= target]
        if len(future) == 0:
            continue
        value = daily.get(future[0], np.nan)
        if np.isfinite(value):
            moves.append(value)
    return pd.Series(moves, dtype=float)


def next_earnings(ticker: str) -> pd.Timestamp | None:
    ed = earnings_dates(ticker)
    if ed.empty:
        return None
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    idx = pd.DatetimeIndex([pd.Timestamp(t).tz_localize(None) for t in ed.index])
    future = idx[idx >= today]
    return future.min() if len(future) else None


def implied_move(ticker: str, spot: float, after: pd.Timestamp) -> dict | None:
    """ATM straddle cost as a share of spot, for the first expiry after `after`.

    A fairly priced ATM straddle costs roughly what the market expects the stock
    to move in absolute terms, so this doubles as the market's forecast.
    """
    try:
        tk = yf.Ticker(ticker)
        expiries = [e for e in tk.options if pd.Timestamp(e) >= after]
        if not expiries:
            return None
        expiry = expiries[0]
        chain = tk.option_chain(expiry)

        total = 0.0
        for side in (chain.calls, chain.puts):
            if side.empty:
                return None
            leg = side.iloc[(side["strike"] - spot).abs().argsort()[:1]].iloc[0]
            bid, ask = float(leg.get("bid", 0) or 0), float(leg.get("ask", 0) or 0)
            if ask <= 0:
                return None
            total += (bid + ask) / 2
        return {"expiry": expiry, "implied_move": total / spot}
    except Exception:
        return None


def calendar(close: pd.DataFrame, tickers: list[str] | None = None,
             within_days: int = 60) -> pd.DataFrame:
    """Upcoming earnings, with implied vs historical move for each."""
    tickers = tickers or list(close.columns)
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()

    rows = []
    for t in tickers:
        if t not in close.columns:
            continue
        nxt = next_earnings(t)
        if nxt is None or (nxt - today).days > within_days:
            continue

        series = close[t].dropna()
        if series.empty:
            continue
        spot = float(series.iloc[-1])
        hist = reaction_moves(t, series)
        imp = implied_move(t, spot, nxt)

        rows.append({
            "ticker": t,
            "earnings": nxt.date(),
            "days_away": (nxt - today).days,
            "spot": spot,
            "hist_move": hist.mean() if len(hist) else np.nan,
            "hist_max": hist.max() if len(hist) else np.nan,
            "n_events": len(hist),
            "expiry": imp["expiry"] if imp else None,
            "implied_move": imp["implied_move"] if imp else np.nan,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # >1 means the option is charging more than the stock has historically moved.
    df["priced_vs_hist"] = df["implied_move"] / df["hist_move"]
    return df.sort_values("days_away").reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    import screener

    ap = argparse.ArgumentParser(description="Upcoming catalysts and how options price them.")
    ap.add_argument("--within", type=int, default=45, help="only events this many days out")
    ap.add_argument("--tickers", nargs="*", help="limit to these tickers")
    args = ap.parse_args()

    close = screener.load_universe()
    df = calendar(close, args.tickers, args.within)
    if df.empty:
        raise SystemExit(f"No earnings within {args.within} days for the selected names.")

    print()
    print(f"=== EARNINGS WITHIN {args.within} DAYS ===")
    print()
    print(f"  {'ticker':<8}{'date':>12}{'in':>5}{'implied':>10}{'historical':>12}"
          f"{'worst':>8}{'n':>4}{'priced/hist':>13}{'expiry':>13}")
    for _, r in df.iterrows():
        ratio = f"{r['priced_vs_hist']:.2f}" if np.isfinite(r["priced_vs_hist"]) else "-"
        hist = f"{r['hist_move']*100:.1f}%" if np.isfinite(r["hist_move"]) else "-"
        worst = f"{r['hist_max']*100:.1f}%" if np.isfinite(r["hist_max"]) else "-"
        imp = f"{r['implied_move']*100:.1f}%" if np.isfinite(r["implied_move"]) else "-"
        print(f"  {r['ticker']:<8}{str(r['earnings']):>12}{int(r['days_away']):>5}"
              f"{imp:>10}{hist:>12}{worst:>8}{int(r['n_events']):>4}{ratio:>13}"
              f"{str(r['expiry']):>13}")

    valid = df.dropna(subset=["priced_vs_hist"])
    if not valid.empty:
        print()
        print(f"  median priced/hist across {len(valid)} names: "
              f"{valid['priced_vs_hist'].median():.2f}")
        print("  >1 = option charges more than the stock has historically moved on earnings")
