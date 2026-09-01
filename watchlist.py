"""Weekly swing-trading watchlist — only the filters that survived testing.

Built from the @jv_trading process, with the components stripped out that failed:

  KEPT     market regime filter, sector rotation, momentum ranking, liquidity
  DROPPED  the breakout entry trigger. Tested at -0.76% edge, t = -2.97 -
           significantly NEGATIVE on its own, and adding it to momentum cut the
           edge from +3.61% to +1.74%. See ablate.py.
  DROPPED  the "coil"/contraction, ADR and volume-taper filters. Each cut the
           sample by an order of magnitude while adding nothing measurable.

So this ranks candidates; it does not tell you when to buy. Given the breakout
result, buying into the consolidation is better supported than buying the break.

HONEST STATUS: momentum itself did not clear my permutation null (p = 0.57), but
that null is mis-specified for this question - shuffling preserves each stock's
drift, so it already contains the effect being measured. Momentum is heavily
replicated in the literature; this tool rests on that, not on my test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_YEAR = 252

SECTOR_ETFS = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care",
    "XLE": "Energy", "XLI": "Industrials", "XLY": "Cons Discretionary",
    "XLP": "Cons Staples", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Communication",
}

# Sector per name, so the ranking can favour stocks in leading sectors.
SECTORS = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "MRVL": "XLK",
    "AMD": "XLK", "INTC": "XLK", "MU": "XLK", "QCOM": "XLK", "TXN": "XLK",
    "ADBE": "XLK", "CRM": "XLK", "ORCL": "XLK", "CSCO": "XLK", "IBM": "XLK",
    "NOW": "XLK", "PANW": "XLK", "ARM": "XLK", "DELL": "XLK", "WDC": "XLK",
    "STX": "XLK", "SMCI": "XLK", "PLTR": "XLK", "SHOP": "XLK", "IONQ": "XLK",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",
    "SBUX": "XLY", "TGT": "XLY", "GM": "XLY", "F": "XLY", "ABNB": "XLY",
    "UBER": "XLI", "BA": "XLI", "CAT": "XLI", "DE": "XLI", "GE": "XLI",
    "LMT": "XLI", "RTX": "XLI", "HON": "XLI", "UPS": "XLI", "FDX": "XLI",
    "RKLB": "XLI", "ASTS": "XLI", "LUNR": "XLI", "ACHR": "XLI",
    "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC", "T": "XLC",
    "VZ": "XLC", "CMCSA": "XLC",
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF", "WFC": "XLF",
    "V": "XLF", "MA": "XLF", "AXP": "XLF", "SCHW": "XLF", "BLK": "XLF",
    "COIN": "XLF", "SOFI": "XLF", "PYPL": "XLF",
    "JNJ": "XLV", "PFE": "XLV", "MRK": "XLV", "LLY": "XLV", "ABBV": "XLV",
    "UNH": "XLV", "TMO": "XLV", "ABT": "XLV", "BMY": "XLV", "GILD": "XLV",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "OXY": "XLE",
    "IREN": "XLE",
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP", "COST": "XLP",
}


def _download(tickers: list[str], start: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    return close.dropna(how="all")


def regime(spy: pd.Series) -> dict:
    """The go/no-go gate: 10 > 20 > 50 SMA, with the fast line rising."""
    s10, s20, s50 = (spy.rolling(n).mean() for n in (10, 20, 50))
    stacked = s10.iloc[-1] > s20.iloc[-1] > s50.iloc[-1]
    rising = s10.iloc[-1] > s10.iloc[-6]
    return {
        "green": bool(stacked and rising),
        "stacked": bool(stacked),
        "rising": bool(rising),
        "spy": float(spy.iloc[-1]),
        "s10": float(s10.iloc[-1]), "s20": float(s20.iloc[-1]), "s50": float(s50.iloc[-1]),
    }


def sector_strength(close: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for etf, name in SECTOR_ETFS.items():
        if etf not in close.columns:
            continue
        s = close[etf].dropna()
        if len(s) < 70:
            continue
        rows.append({
            "etf": etf, "sector": name,
            "perf_1w": s.iloc[-1] / s.iloc[-6] - 1,
            "perf_1m": s.iloc[-1] / s.iloc[-22] - 1,
            "perf_3m": s.iloc[-1] / s.iloc[-64] - 1,
        })
    df = pd.DataFrame(rows)
    # 1-month and 3-month averaged: the video's "common denominator between the
    # two", which is just a crude way of asking for persistent strength.
    df["score"] = (df["perf_1m"] + df["perf_3m"]) / 2
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def rank_stocks(close: pd.DataFrame, sectors: pd.DataFrame, top_sectors: int = 4) -> pd.DataFrame:
    leaders = set(sectors.head(top_sectors)["etf"])
    rows = []
    for t, etf in SECTORS.items():
        if t not in close.columns:
            continue
        s = close[t].dropna()
        if len(s) < TRADING_YEAR:
            continue
        s10, s20 = s.rolling(10).mean(), s.rolling(20).mean()
        rows.append({
            "ticker": t,
            "sector": SECTOR_ETFS.get(etf, etf),
            "leading": etf in leaders,
            "price": float(s.iloc[-1]),
            # 12-1 momentum: the standard academic construction, skipping the
            # most recent month to avoid short-term reversal.
            "mom_12_1": float(s.iloc[-22] / s.iloc[-TRADING_YEAR] - 1),
            "perf_3m": float(s.iloc[-1] / s.iloc[-64] - 1),
            "vs_10sma": float(s.iloc[-1] / s10.iloc[-1] - 1),
            "vs_20sma": float(s.iloc[-1] / s20.iloc[-1] - 1),
            "from_high": float(s.iloc[-1] / s.rolling(TRADING_YEAR).max().iloc[-1] - 1),
        })
    df = pd.DataFrame(rows)
    # Rank on momentum, then favour leading sectors — rotation is a tiebreak on
    # top of the factor that actually carries the evidence, not a replacement.
    df["rank"] = df["mom_12_1"].rank(ascending=False) - df["leading"] * 8
    return df.sort_values("rank").reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Weekly swing watchlist.")
    ap.add_argument("--top", type=int, default=15, help="how many names to show")
    ap.add_argument("--sectors", type=int, default=4, help="how many leading sectors count")
    args = ap.parse_args()

    tickers = sorted(set(list(SECTORS) + list(SECTOR_ETFS) + ["SPY"]))
    close = _download(tickers, "2023-01-01")

    r = regime(close["SPY"].dropna())
    flag = "GREEN - conditions support long swings" if r["green"] else \
           "RED - regime filter says stand down"
    print()
    print(f"=== MARKET REGIME: {flag} ===")
    print(f"  SPY {r['spy']:.2f}   10sma {r['s10']:.2f}   20sma {r['s20']:.2f}   50sma {r['s50']:.2f}")
    print(f"  stacked 10>20>50: {r['stacked']}      10sma rising: {r['rising']}")

    sec = sector_strength(close)
    print()
    print("=== SECTOR STRENGTH ===")
    print()
    print(f"  {'#':<3}{'etf':<6}{'sector':<22}{'1w':>8}{'1m':>8}{'3m':>8}")
    for i, row in sec.iterrows():
        mark = "*" if i < args.sectors else " "
        print(f"  {mark}{i+1:<2}{row['etf']:<6}{row['sector']:<22}"
              f"{row['perf_1w']*100:>7.1f}%{row['perf_1m']*100:>7.1f}%{row['perf_3m']*100:>7.1f}%")
    print(f"  (* = leading sectors, used as a ranking tiebreak)")

    df = rank_stocks(close, sec, args.sectors)
    print()
    print(f"=== WATCHLIST: top {args.top} by momentum, leading sectors favoured ===")
    print()
    print(f"  {'ticker':<7}{'sector':<20}{'lead':>5}{'price':>9}{'mom12-1':>10}"
          f"{'3mo':>8}{'v10sma':>8}{'v20sma':>8}{'frm high':>10}")
    for _, row in df.head(args.top).iterrows():
        print(f"  {row['ticker']:<7}{row['sector']:<20}{'*' if row['leading'] else '':>5}"
              f"{row['price']:>9.2f}{row['mom_12_1']*100:>9.1f}%{row['perf_3m']*100:>7.1f}%"
              f"{row['vs_10sma']*100:>7.1f}%{row['vs_20sma']*100:>7.1f}%{row['from_high']*100:>9.1f}%")
    print()
    print("  This ranks candidates. It is NOT an entry signal - the breakout trigger")
    print("  tested negative (t = -2.97), so buying into consolidation is better")
    print("  supported than buying the break. Names near their 10/20 SMA are the")
    print("  ones consolidating rather than extended.")
