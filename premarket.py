"""Pre-market gap scanner.

    .venv/Scripts/python.exe premarket.py
    .venv/Scripts/python.exe premarket.py --tickers FNGR IREN ASTS --min-gap 3

Ranks names by how far they have moved in the pre-market session (04:00-09:30 ET)
against the prior regular-session close, with the context needed to judge the gap
rather than just see it.

THIS IS NOT THE SWING BOT. Different strategy, different horizon, different
evidence base. bot.py trades multi-week holds on tested filters; this scans for
intraday volatility. Keeping them in separate files is deliberate.

WHAT THE GAP DATA ACTUALLY SAYS (measured in screener.py's dip/gap work and the
QuantifiedStrategies figures behind it):

  small gaps  (~0.15%)  fill intraday roughly 92% of the time
  larger gaps (~0.35%)  fill roughly 69%
  BUT when high volatility + a large gap + an open outside the prior range all
  coincide, the fill rate COLLAPSES to about 21%.

That last line is the one that matters here. Every name this scanner surfaces is
by construction in the third bucket. Fading a big pre-market runner is a ~1-in-5
proposition, not the 92% the headline number suggests - and the 79% that do not
fill are exactly the ones that keep running against a short.

BROKER CONSTRAINT: shorting requires $2,000 minimum account equity (Reg T, and
Alpaca/most brokers enforce it). Below that, only the long side is available.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import yfinance as yf

PRE_START, PRE_END = "04:00", "09:30"

# High-beta names that actually gap. Large caps rarely move enough pre-market to
# matter here, so this leans small/mid cap and momentum. Edit freely.
DEFAULT_UNIVERSE = [
    "IREN", "ASTS", "RKLB", "LUNR", "ACHR", "IONQ", "RGTI", "QBTS", "SMCI",
    # KEEL was BITF: Bitfarms redomiciled to the US and rebranded April 2026.
    "SOFI", "COIN", "MARA", "RIOT", "CLSK", "HUT", "KEEL", "CIFR", "WULF",
    "PLTR", "AI", "BBAI", "SOUN", "TSLA", "NVDA", "AMD", "MU", "INTC",
    # NKLA removed: Nikola liquidated in Chapter 11, common stock cancelled at zero.
    "GME", "AMC", "SPCE", "PLUG", "FCEL", "BLNK", "CHPT",
    "UPST", "AFRM", "HOOD", "DKNG", "RBLX", "U", "SNAP", "PINS",
]


def premarket_move(ticker: str) -> dict | None:
    """Pre-market change vs the prior regular-session close."""
    try:
        df = yf.download(ticker, period="2d", interval="5m",
                         prepost=True, progress=False, auto_adjust=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = df.index.tz_convert("America/New_York")

        today = df.index[-1].date()
        # Prior regular session close: the last bar before 16:00 on the previous day.
        prior = df[(df.index.date < today) &
                   (df.index.time <= pd.Timestamp("16:00").time()) &
                   (df.index.time >= pd.Timestamp("09:30").time())]
        pre = df[(df.index.date == today) &
                 (df.index.time >= pd.Timestamp(PRE_START).time()) &
                 (df.index.time < pd.Timestamp(PRE_END).time())]
        if prior.empty or pre.empty:
            return None

        prev_close = float(prior["Close"].iloc[-1])
        last = float(pre["Close"].iloc[-1])
        # Daily bars for the volatility context — a 5% gap means something very
        # different on a name that moves 2% a day than on one that moves 10%.
        daily = yf.download(ticker, period="3mo", interval="1d",
                            progress=False, auto_adjust=True)
        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = daily.columns.get_level_values(0)
        adr = float((daily["High"] / daily["Low"] - 1).tail(20).mean()) if not daily.empty else np.nan

        gap = last / prev_close - 1
        return {
            "ticker": ticker,
            "prev_close": prev_close,
            "premarket": last,
            "gap_pct": gap,
            "pre_high": float(pre["High"].max()),
            "pre_low": float(pre["Low"].min()),
            "pre_volume": int(pre["Volume"].sum()),
            "adr": adr,
            # How many normal days of range this gap represents. Above ~2 means
            # the move is already outsized relative to how the stock usually trades.
            "gap_in_adr": gap / adr if adr and adr > 0 else np.nan,
        }
    except Exception:
        return None


def scan(tickers: list[str], min_gap: float) -> pd.DataFrame:
    rows = []
    for t in tickers:
        r = premarket_move(t)
        if r and abs(r["gap_pct"]) * 100 >= min_gap:
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reindex(
        pd.DataFrame(rows)["gap_pct"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pre-market gap scanner.")
    ap.add_argument("--tickers", nargs="*", help="override the default universe")
    ap.add_argument("--min-gap", type=float, default=3.0, help="minimum abs gap %%")
    args = ap.parse_args()

    universe = args.tickers or DEFAULT_UNIVERSE
    print(f"\nscanning {len(universe)} tickers for pre-market moves >= {args.min_gap:.0f}% ...")
    df = scan(universe, args.min_gap)

    if df.empty:
        print(f"\n  no names gapped {args.min_gap:.0f}%+ pre-market.")
        print("  (outside 04:00-09:30 ET this shows the most recent session's pre-market.)\n")
        raise SystemExit(0)

    print()
    print(f"  {'ticker':<8}{'prev close':>12}{'premarket':>11}{'gap':>9}"
          f"{'ADR':>8}{'gap/ADR':>9}{'pre vol':>12}")
    for _, r in df.iterrows():
        adr = f"{r['adr']*100:.1f}%" if np.isfinite(r["adr"]) else "-"
        gadr = f"{r['gap_in_adr']:.1f}x" if np.isfinite(r["gap_in_adr"]) else "-"
        print(f"  {r['ticker']:<8}{r['prev_close']:>12.2f}{r['premarket']:>11.2f}"
              f"{r['gap_pct']*100:>+8.1f}%{adr:>8}{gadr:>9}{r['pre_volume']:>12,}")

    print()
    print("  gap/ADR is the number that matters: above ~2x the move is already")
    print("  larger than the stock's normal daily range, which is the regime where")
    print("  the intraday fill rate drops from ~92% to roughly 21%. Fading these is")
    print("  a 1-in-5 trade, and the 4-in-5 keep running.")
    print("  Shorting needs $2,000 account equity. Below that, long side only.")
    print()
