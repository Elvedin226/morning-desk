"""Strategy 6: overnight-only reversion. Buy the close after a down day, sell the open.

The project already found that PLAIN overnight drift (buy every close, sell every
open) is real but dies above ~4bp of cost. The question here is narrower: does
conditioning on a down day buy enough extra edge per night to pay for it?

Arithmetic of the problem, before any data: conditioning throws away roughly half
the nights. Cost per night is unchanged. So the conditional EV per night must be
about 2x the unconditional EV just to break even on the SAME cost budget, and
strictly more than that to raise the break-even cost level. That is the bar.

Return convention: buy at close_t, sell at open_{t+1}, so one night's gross
return is open_{t+1}/close_t - 1, and one round trip pays 2 x slippage.
yfinance auto_adjust back-adjusts open and close by the same factor, so the
ratio is a clean total return.

Statistics: t-stats are clustered by DATE. Every name in the universe gaps
together on macro news; treating 93 same-night observations as independent would
inflate t by roughly sqrt(93).
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

import mr_core as C

SLIPS = [0.0, 0.0002, 0.0005, 0.0010, 0.0020, 0.0050]


def nights(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (ticker, night): gross overnight return + the prior-day flags."""
    frames = []
    for t, df in data.items():
        close, open_ = df["close"], df["open"]
        gross = open_.shift(-1) / close - 1.0
        down = close < close.shift(1)
        # 'date' is the day whose CLOSE we buy at; the sale is the next open.
        frames.append(pd.DataFrame({
            "date": df.index, "ticker": t, "gross": gross.to_numpy(),
            "down": down.to_numpy(),
            "day_ret": (close / close.shift(1) - 1.0).to_numpy(),
        }))
    out = pd.concat(frames, ignore_index=True).dropna(subset=["gross"])
    return out


def clustered(x: pd.DataFrame, col: str) -> tuple[float, float, int, int]:
    """(mean, date-clustered t, n obs, n dates)."""
    if x.empty:
        return float("nan"), float("nan"), 0, 0
    daily = x.groupby("date")[col].mean()
    t, _ = stats.ttest_1samp(daily.to_numpy(), 0.0)
    return float(daily.mean()), float(t), len(x), len(daily)


def report(n: pd.DataFrame, slip: float) -> None:
    n = n.copy()
    n["net"] = n["gross"] - 2 * slip
    rows = [("all nights (baseline)", n),
            ("after a DOWN day", n[n["down"]]),
            ("after an UP day", n[~n["down"]]),
            ("after a -2% or worse day", n[n["day_ret"] <= -0.02]),
            ("after a -4% or worse day", n[n["day_ret"] <= -0.04])]
    print(f"\n  slippage {slip*10000:>4.0f} bp/side   (round trip pays {slip*20000:.0f} bp)")
    print("  {:<28}{:>10}{:>9}{:>10}{:>9}{:>12}".format(
        "condition", "nights", "dates", "EV/night", "t(date)", "ann. gross"))
    print("  " + "-" * 78)
    for label, sub in rows:
        ev, t, nn, nd = clustered(sub, "net")
        # Annualized only over the nights actually taken - the honest scaling for a
        # rule that is in the market on a fraction of nights.
        share = nd / n["date"].nunique() if n["date"].nunique() else 0
        ann = (1 + ev) ** (252 * share) - 1 if not np.isnan(ev) else float("nan")
        print("  {:<28}{:>10}{:>9}{:>10}{:>9}{:>12}".format(
            label, nn, nd, C.fmt_pct(ev, 3), C.fmt(t, 2), C.fmt_pct(ann, 1)))


def by_year(n: pd.DataFrame, slip: float) -> None:
    n = n.copy()
    n["net"] = n["gross"] - 2 * slip
    n["year"] = pd.to_datetime(n["date"]).dt.year
    print(f"\n  BY YEAR @ {slip*10000:.0f}bp/side   (down-day nights vs all nights)\n")
    print("  {:<7}{:>9}{:>11}{:>9}{:>13}{:>11}".format(
        "year", "nights", "EV down", "t", "EV all", "EV up"))
    print("  " + "-" * 60)
    pos = 0
    yrs = sorted(n["year"].unique())
    for y in yrs:
        s = n[n["year"] == y]
        d, td, nd, _ = clustered(s[s["down"]], "net")
        a, _, _, _ = clustered(s, "net")
        u, _, _, _ = clustered(s[~s["down"]], "net")
        pos += d > 0
        print("  {:<7}{:>9}{:>11}{:>9}{:>13}{:>11}".format(
            y, nd, C.fmt_pct(d, 3), C.fmt(td, 2), C.fmt_pct(a, 3), C.fmt_pct(u, 3)))
    print(f"\n  down-day EV positive in {pos}/{len(yrs)} years")


def shuffle_null(n: pd.DataFrame, runs: int = 200, seed: int = 0) -> None:
    """Does the DOWN-DAY CONDITION add anything, or is it just overnight drift?

    Null: the flag carries no information. Shuffle the down/up label within each
    date (so each night keeps its real cross-section, and the market-wide
    component the clustering already accounts for is preserved) and re-measure
    the down-minus-up spread. This isolates the conditioning, which is the only
    thing being claimed here - plain overnight drift is already known to exist.
    """
    real_d = n[n["down"]].groupby("date")["gross"].mean()
    real_u = n[~n["down"]].groupby("date")["gross"].mean()
    real = float((real_d - real_u).dropna().mean())

    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(n["date"])
    gross = n["gross"].to_numpy()
    flag = n["down"].to_numpy()
    order = np.argsort(codes, kind="stable")
    codes_s, gross_s, flag_s = codes[order], gross[order], flag[order]
    bounds = np.flatnonzero(np.diff(codes_s)) + 1
    groups = np.split(np.arange(len(codes_s)), bounds)

    out = []
    for _ in range(runs):
        perm = flag_s.copy()
        for g in groups:
            perm[g] = rng.permutation(flag_s[g])
        d = np.bincount(codes_s, gross_s * perm) / np.maximum(np.bincount(codes_s, perm), 1)
        u = np.bincount(codes_s, gross_s * ~perm) / np.maximum(np.bincount(codes_s, ~perm), 1)
        out.append(np.nanmean(d - u))
    null = np.array(out)
    p = float(np.mean(null >= real))
    print(f"\n  CONDITIONING TEST: down-night EV minus up-night EV (gross)")
    print(f"     real spread            {real*100:>8.4f}%")
    print(f"     shuffled mean / 95th   {null.mean()*100:>8.4f}% / {np.percentile(null,95)*100:.4f}%"
          f"   ({runs} runs)")
    print(f"     p-value                {p:>8.3f}"
          f"   {'<-- condition adds nothing' if p > 0.10 else '<-- condition adds signal'}")


def main() -> None:
    data = C.universe()
    n = nights(data)
    print(f"  UNIVERSE: {len(data)} tickers, {len(n):,} ticker-nights, "
          f"{n['date'].nunique():,} distinct dates, "
          f"{pd.to_datetime(n['date']).min().date()} to {pd.to_datetime(n['date']).max().date()}")
    print("  Survivorship-biased (today's large caps).")

    print("\n\n" + "=" * 100)
    print("  OVERNIGHT REVERSION: cost sweep")
    print("=" * 100)
    for s in SLIPS:
        report(n, s)

    by_year(n, 0.0005)
    shuffle_null(n)


if __name__ == "__main__":
    main()
