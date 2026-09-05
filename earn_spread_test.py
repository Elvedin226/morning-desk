"""Buying an in-the-money call debit spread through earnings. Does it pay?

The pitch: instead of a naked call costing ~$1,000 on a high-priced name, buy a
narrow call debit spread with both strikes already in the money. It costs a
fraction, the loss is capped, and you keep the full width as long as the stock
does not fall below the short strike.

That structure is a genuinely sensible choice and better reasoned than most -
mostly INTRINSIC value, so the implied-vol collapse after earnings barely
touches it, which is the single biggest trap in buying options into an event.

But the structure is not the question. The question is the ONE thing the video
never states: how often does the stock actually hold above the short strike?

An ITM call spread is two bets at once:
    directionally bullish - it needs the stock not to fall
    short volatility     - it needs the move to be SMALLER than priced

So the test is: across real earnings events, how often does a stock hold above
a strike sitting a given distance below spot, and what does that pay against
the price the market charges for it?

METHOD. Real earnings dates, real reactions, no option chain required. The
spread's fair price is (probability it finishes above the short strike) x width,
which is what a market with no edge in it would charge. Measuring the actual
frequency against that price gives the edge - or its absence - directly.

The universe is deliberately broad and never hand-picked, because two studies in
this project were already corrupted by choosing the names after the fact.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import catalysts
import watchlist

# Both strikes in the money, as a fraction below spot. The video's example sat
# roughly 0.6% under, which it described as "deep".
BUFFERS = (0.005, 0.010, 0.020, 0.030, 0.050)
COST = 0.02          # 2% of the debit, round trip. The video says twice that
                     # these spreads have very wide markets, which is correct.


def reactions(tickers: list[str]) -> pd.DataFrame:
    """SIGNED overnight earnings reactions, with their dates.

    catalysts.reaction_moves() cannot be reused here: it returns ABSOLUTE moves
    on a RangeIndex, because it exists to size an implied move. A directional
    spread needs the sign (a call spread cares which way) and the date (to check
    stability by year), so the calculation is redone rather than adapted.
    """
    rows = []
    px = yf.download(tickers, start="2019-01-01", auto_adjust=True,
                     progress=False, group_by="ticker")
    lvl0 = set(px.columns.get_level_values(0)) if isinstance(px.columns, pd.MultiIndex) else set()
    for t in tickers:
        if t not in lvl0:
            continue
        c = px[t]["Close"].dropna()
        if len(c) < 300:
            continue
        idx = c.index.tz_localize(None) if c.index.tz is not None else c.index
        c = pd.Series(c.to_numpy(), index=idx)
        ret = c.pct_change()
        try:
            ed = catalysts.earnings_dates(t, 32)
        except Exception:
            continue
        if ed is None or ed.empty:
            continue
        for ts in ed.index:
            stamp = pd.Timestamp(ts)
            day = stamp.tz_localize(None).normalize() if stamp.tz is not None else stamp.normalize()
            # Released after the bell -> the reaction is the NEXT session.
            target = day + pd.Timedelta(days=1) if stamp.hour >= catalysts.AFTER_HOURS_CUTOFF else day
            fut = ret.index[ret.index >= target]
            if len(fut) == 0:
                continue
            v = ret.get(fut[0], np.nan)
            if np.isfinite(v):
                rows.append({"ticker": t, "date": fut[0], "move": float(v)})
    return pd.DataFrame(rows).drop_duplicates(subset=["ticker", "date"])


def main():
    tickers = sorted(watchlist.SECTORS)
    print(f"\n  collecting earnings reactions for {len(tickers)} names ...")
    df = reactions(tickers)
    if df.empty:
        print("  no earnings data returned")
        return
    mv = df["move"].to_numpy()
    print(f"  {len(df):,} earnings events, {df['ticker'].nunique()} names, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  mean move {mv.mean()*100:+.2f}%   median {np.median(mv)*100:+.2f}%   "
          f"|move| mean {np.abs(mv).mean()*100:.2f}%   up {np.mean(mv>0)*100:.0f}%")

    print("\n  ITM CALL DEBIT SPREAD HELD THROUGH EARNINGS")
    print("  Wins whenever the stock finishes above a short strike sitting")
    print("  `buffer` below spot. Fair price = win rate x width, so a market")
    print("  with no edge charges exactly the win rate.")
    print("  " + "=" * 76)
    print(f"  {'buffer':>8}{'win rate':>11}{'fair debit':>12}{'payoff':>9}"
          f"{'EV @fair':>11}{'EV +2% cost':>13}")
    print("  " + "-" * 76)
    for b in BUFFERS:
        win = float(np.mean(mv > -b))
        debit = win                      # as a fraction of width
        if debit >= 0.995:
            continue
        payoff = (1 - debit) / debit
        ev_fair = win * (1 - debit) - (1 - win) * debit
        d_cost = debit * (1 + COST)
        ev_cost = win * (1 - d_cost) - (1 - win) * d_cost
        print(f"  {b*100:>7.1f}%{win*100:>10.1f}%{debit:>12.3f}{payoff:>8.2f}:1"
              f"{ev_fair*100:>10.2f}%{ev_cost/d_cost*100:>12.1f}%")

    print("\n  EV at a fair price is zero BY CONSTRUCTION - that is the point.")
    print("  The market sets the debit to the win rate, so high probability buys")
    print("  a small payoff and low probability buys a big one, and both land in")
    print("  the same place. Costs then push it negative. The structure cannot")
    print("  create an edge; only a directional view can.")

    # The one place an edge could actually live.
    print("\n  IS THERE A DIRECTIONAL DRIFT TO EXPLOIT?")
    print("  " + "=" * 76)
    t = mv.mean() / (mv.std(ddof=1) / np.sqrt(len(mv)))
    print(f"  mean reaction {mv.mean()*100:+.3f}%   t = {t:.2f}   "
          f"{'significant' if abs(t) > 2 else 'not significant'}")
    print(f"  a bullish structure needs a real upward drift; measured, it is "
          f"{'present' if t > 2 else 'absent'}.")

    print("\n  BY YEAR  (is any of it stable?)")
    print("  " + "=" * 76)
    df["yr"] = pd.to_datetime(df["date"]).dt.year
    for yr, g in df.groupby("yr"):
        m = g["move"].to_numpy()
        if len(m) < 40:
            continue
        print(f"  {yr}   n={len(m):>4}   mean {m.mean()*100:>+6.2f}%   "
              f"up {np.mean(m>0)*100:>4.1f}%   |move| {np.abs(m).mean()*100:>5.2f}%   "
              f"hold -1% {np.mean(m>-0.01)*100:>4.1f}%")


if __name__ == "__main__":
    main()
