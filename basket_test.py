"""Connors RSI-2 across the whole universe, sized like the real account.

Single-symbol RSI-2 fires about 5 times a year. Across 90 names that becomes a
signal most days, which is the only way this strategy can meet a daily target.

This is not a Sharpe study. It answers the practical question directly: starting
at $421, risking a fixed fraction per trade, paying 5bp a side, what does the
equity curve actually do and how often does a day clear $20?

WHAT IS MODELLED
    entry     next open after the signal (never the signal bar's close)
    exit      next open after close > 5sma, or a hard stop
    costs     5bp per side
    sizing    fixed fraction of equity per position, capped by cash
    limits    max concurrent positions, one position per name

WHAT IS NOT
    Intraday path. A stop is checked against the daily low, and filled at the
    stop unless the bar gapped through it. Real fills are worse.
    Survivorship. The universe is today's large caps. Measured elsewhere in this
    project at roughly 5 points of annual overstatement.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

import watchlist

START = "2011-01-01"      # post-publication only. The pre-2011 record is the
                          # author's own sample and not evidence for today.
EQUITY0 = 421.0
RISK_FRAC = 0.20          # fraction of equity per position
MAX_POS = 3
COST = 0.0005
STOP_PCT = 0.08           # hard stop, since RSI-2 has no stop of its own
RSI_ENTRY = 5
TARGET_DAY = 20.0


def rsi(s: pd.Series, n: int = 2) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def panel(tickers: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(tickers, start=START, auto_adjust=True, progress=False,
                      group_by="ticker")
    out = {}
    lvl0 = set(raw.columns.get_level_values(0)) if isinstance(raw.columns, pd.MultiIndex) else set()
    for t in tickers:
        df = raw[t] if t in lvl0 else None
        if df is None:
            continue
        df = df.dropna()
        if len(df) < 260:
            continue
        out[t] = df
    return out


def simulate(px: dict[str, pd.DataFrame], risk_frac=RISK_FRAC, max_pos=MAX_POS,
             rsi_entry=RSI_ENTRY, verbose=False) -> dict:
    # Precompute every indicator once. Signals use only data through bar t and
    # are acted on at bar t+1's open, so nothing here can see its own outcome.
    sig, ma5, closes, opens, lows = {}, {}, {}, {}, {}
    for t, df in px.items():
        c = df["Close"]
        sig[t] = (rsi(c, 2) < rsi_entry) & (c > c.rolling(200).mean())
        ma5[t] = c.rolling(5).mean()
        closes[t], opens[t], lows[t] = c, df["Open"], df["Low"]

    dates = sorted(set().union(*[set(d.index) for d in px.values()]))
    cash, open_pos, closed, curve = EQUITY0, {}, [], []

    for i, day in enumerate(dates):
        if i == 0:
            continue
        prev = dates[i - 1]

        # ---- exits first, at today's open, so capital is free to re-enter ----
        for t in list(open_pos):
            p = open_pos[t]
            if day not in opens[t].index:
                continue
            o = float(opens[t].loc[day])
            lo = float(lows[t].loc[day]) if day in lows[t].index else o

            exit_px, why = None, None
            if lo <= p["stop"]:
                exit_px, why = (min(o, p["stop"]), "stop")
            elif prev in closes[t].index and prev in ma5[t].index:
                cl, m5 = float(closes[t].loc[prev]), float(ma5[t].loc[prev])
                if np.isfinite(m5) and cl > m5:
                    exit_px, why = o, "signal"
            if exit_px is None:
                continue
            proceeds = p["qty"] * exit_px * (1 - COST)
            pnl = proceeds - p["cost"]
            cash += proceeds
            closed.append({"ticker": t, "in": p["entry"], "out": exit_px,
                           "pnl": pnl, "why": why, "opened": p["day"], "closed": day})
            del open_pos[t]

        # ---- entries, on yesterday's signal, at today's open ----
        if len(open_pos) < max_pos:
            fired = [t for t in px
                     if t not in open_pos and prev in sig[t].index and bool(sig[t].loc[prev])]
            # Rank by how oversold, so the choice is deterministic rather than
            # dictionary order - which would silently favour whatever loaded first.
            fired.sort(key=lambda t: float(rsi(closes[t], 2).loc[prev]))
            for t in fired:
                if len(open_pos) >= max_pos or day not in opens[t].index:
                    continue
                o = float(opens[t].loc[day])
                equity_now = cash + sum(q["qty"] * o for q in open_pos.values())
                want = min(equity_now * risk_frac, cash)
                if want < 1.0 or o <= 0:
                    continue
                qty = want / (o * (1 + COST))
                cost = qty * o * (1 + COST)
                cash -= cost
                open_pos[t] = {"qty": qty, "entry": o, "cost": cost,
                               "stop": o * (1 - STOP_PCT), "day": day}

        mv = sum(p["qty"] * float(closes[t].loc[day])
                 for t, p in open_pos.items() if day in closes[t].index)
        curve.append({"date": day, "equity": cash + mv})

    eq = pd.DataFrame(curve).set_index("date")["equity"]
    tr = pd.DataFrame(closed)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    daily = eq.diff().dropna()
    dd = (eq / eq.cummax() - 1).min()
    return {
        "final": float(eq.iloc[-1]), "cagr": (eq.iloc[-1] / EQUITY0) ** (1 / years) - 1,
        "maxdd": float(dd), "trades": len(tr), "years": years,
        "per_year": len(tr) / years,
        "win": float((tr["pnl"] > 0).mean()) if len(tr) else np.nan,
        "avg_win": float(tr.loc[tr["pnl"] > 0, "pnl"].mean()) if len(tr) else np.nan,
        "avg_loss": float(tr.loc[tr["pnl"] <= 0, "pnl"].mean()) if len(tr) else np.nan,
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else np.nan,
        "days_over_target": float((daily >= TARGET_DAY).mean()),
        "median_day": float(daily.median()), "mean_day": float(daily.mean()),
        "best_day": float(daily.max()), "worst_day": float(daily.min()),
        "equity": eq, "tr": tr,
    }


def main() -> None:
    tickers = sorted(watchlist.SECTORS)
    print(f"\n  loading {len(tickers)} names from {START} ...")
    px = panel(tickers)
    print(f"  {len(px)} usable\n")

    r = simulate(px)
    print("  CONNORS RSI-2 ACROSS THE UNIVERSE, from $421")
    print("  " + "=" * 68)
    print(f"    period            {r['years']:.1f} years")
    print(f"    final equity      ${r['final']:,.2f}   ({r['cagr']*100:+.2f}% a year)")
    print(f"    max drawdown      {r['maxdd']*100:.1f}%")
    print(f"    Sharpe            {r['sharpe']:.2f}")
    print(f"    trades            {r['trades']:,}  ({r['per_year']:.0f} a year, "
          f"{r['per_year']/252:.2f} a day)")
    print(f"    win rate          {r['win']*100:.0f}%")
    print(f"    avg win / loss    ${r['avg_win']:+.2f} / ${r['avg_loss']:+.2f}")
    print()
    print(f"    THE ACTUAL QUESTION: how often does a day clear ${TARGET_DAY:.0f}?")
    print(f"    days over target  {r['days_over_target']*100:.1f}% of sessions")
    print(f"    median day        ${r['median_day']:+.2f}")
    print(f"    mean day          ${r['mean_day']:+.2f}")
    print(f"    best / worst day  ${r['best_day']:+.2f} / ${r['worst_day']:+.2f}")

    print("\n\n  SENSITIVITY  (is the result a knife edge?)")
    print("  " + "=" * 68)
    print(f"  {'risk/pos':>9}{'max pos':>9}{'RSI<':>6}{'final':>11}{'CAGR':>9}"
          f"{'maxDD':>8}{'trades':>8}{'>$20 days':>11}")
    for rf, mp, re_ in [(0.20, 3, 5), (0.33, 3, 5), (0.50, 2, 5), (1.00, 1, 5),
                        (0.20, 3, 10), (0.33, 3, 10), (0.20, 5, 10), (0.33, 5, 15)]:
        x = simulate(px, risk_frac=rf, max_pos=mp, rsi_entry=re_)
        print(f"  {rf*100:>8.0f}%{mp:>9}{re_:>6}${x['final']:>10,.0f}"
              f"{x['cagr']*100:>8.2f}%{x['maxdd']*100:>7.1f}%{x['trades']:>8,}"
              f"{x['days_over_target']*100:>10.1f}%")

    # Buy and hold the index over the same window, as the thing to beat.
    spy = yf.download("SPY", start=START, auto_adjust=True, progress=False)
    sc = spy["Close"]
    sc = sc.iloc[:, 0] if hasattr(sc, "columns") else sc
    bh = EQUITY0 * float(sc.iloc[-1] / sc.iloc[0])
    yrs = (sc.index[-1] - sc.index[0]).days / 365.25
    print(f"\n  buy & hold SPY over the same window: ${bh:,.0f} "
          f"({(bh/EQUITY0)**(1/yrs)-1:+.2%} a year)")


if __name__ == "__main__":
    main()
