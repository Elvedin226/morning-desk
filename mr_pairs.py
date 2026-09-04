"""Strategy 5: within-sector pairs / statistical arbitrage.

Pair selection is the whole game, and it is where pairs studies usually cheat.
Rules here, fixed before looking at any result:

  * Candidates are EVERY pair inside a sector bucket from watchlist.SECTORS.
    No hand-picking, no "these two obviously move together".
  * Formation window (2 years) chooses the hedge ratio, the spread mean/sd, and
    which pairs pass an Engle-Granger cointegration test. Trading window (next
    1 year) uses ONLY those formation-period numbers. Then roll.
  * The headline question asked of this study: of the pairs that cointegrate
    in-sample, how many still cointegrate out-of-sample? Compared against the
    same test on the pairs that FAILED in-sample, which is the base rate.

Engle-Granger, implemented here because statsmodels is not installed:
  1. OLS  log(A) = a + b*log(B)  -> residual spread
  2. ADF with a constant and 1 lag on the residual; t-stat on the level term
  3. Compare against MacKinnon critical values for a 2-variable cointegrating
     regression, constant and no trend: 1% -3.90, 5% -3.34, 10% -3.04.
     (Using plain ADF critical values here would be the classic error - they are
     too permissive once the spread has been fitted.)

P&L convention: dollar-neutral, gross exposure 1.0, so weights are
+1/(1+|b|) on A and -b/(1+|b|) on B. A round trip pays 2 x slippage on that
gross notional, same as a long-only round trip.

BIAS TO STATE PLAINLY: the short leg is free here. No borrow fee, no hard-to-
borrow rate, no recall risk, no margin. Real pairs trading pays all of those,
so every number below is optimistic before costs are even charged.
"""

from __future__ import annotations

import warnings
from itertools import combinations

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

import mr_core as C
import watchlist

FORM = 504          # 2 years to fit
TRADE = 252         # 1 year to trade, then roll
ENTRY_Z = 2.0
EXIT_Z = 0.0
STOP_Z = 4.0
# MacKinnon critical values for a 2-variable cointegrating REGRESSION (the
# coefficients were fitted, so the residual test needs the stricter table).
CRIT = {"1%": -3.90, "5%": -3.34, "10%": -3.04}
# Out of sample the hedge ratio is NOT refitted - the spread is a known series -
# so the correct threshold there is the plain ADF one. Both are reported.
CRIT_OOS = -2.86
SLIPS = [0.0, 0.0005, 0.0010, 0.0020, 0.0050]


def adf_t(x: np.ndarray, lags: int = 1) -> float:
    """t-stat on the level coefficient of  dx_t = a + g*x_{t-1} + sum d_i dx_{t-i}."""
    dx = np.diff(x)
    n = len(dx) - lags
    if n < 30:
        return np.nan
    y = dx[lags:]
    cols = [np.ones(n), x[lags:-1]]
    for i in range(1, lags + 1):
        cols.append(dx[lags - i:-i])
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - X.shape[1]
    if dof <= 0:
        return np.nan
    s2 = resid @ resid / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.nan
    se = np.sqrt(cov[1, 1])
    return float(beta[1] / se) if se > 0 else np.nan


def hedge(la: np.ndarray, lb: np.ndarray) -> tuple[float, float]:
    """OLS log(A) = a + b log(B)."""
    X = np.column_stack([np.ones(len(lb)), lb])
    beta, *_ = np.linalg.lstsq(X, la, rcond=None)
    return float(beta[0]), float(beta[1])


def trade_spread(z: np.ndarray, ra: np.ndarray, rb: np.ndarray, b: float,
                 slip: float) -> list[dict]:
    """Walk the trading window, z-score entries, dollar-neutral P&L."""
    w = 1.0 / (1.0 + abs(b))
    daily = w * ra - (b * w) * rb        # long-A/short-B spread return, gross 1.0
    trades, pos, entry_i = [], 0, 0
    for i in range(len(z)):
        if pos == 0:
            # Entry needs ENTRY_Z < |z| < STOP_Z. Without the upper guard the rule
            # opens positions that are ALREADY past the stop - the spread has
            # already broken - and closes them one bar later for the cost. That
            # was 96% of trades on the first pass; it is a bug, not a finding.
            if -STOP_Z < z[i] < -ENTRY_Z:
                pos, entry_i = 1, i
            elif ENTRY_Z < z[i] < STOP_Z:
                pos, entry_i = -1, i
        else:
            close = (pos == 1 and z[i] >= EXIT_Z) or (pos == -1 and z[i] <= EXIT_Z) \
                    or abs(z[i]) > STOP_Z or i == len(z) - 1
            if close:
                seg = daily[entry_i + 1:i + 1]
                ret = pos * float(np.prod(1 + seg) - 1) - 2 * slip
                trades.append({"bars": i - entry_i, "return": ret, "side": pos,
                               "stopped": bool(abs(z[i]) > STOP_Z)})
                pos = 0
    return trades


def run(slip: float, verbose: bool = False) -> dict:
    data = C.universe()
    # Common calendar so a pair is always compared on aligned bars.
    closes = pd.DataFrame({t: d["close"] for t, d in data.items()}).sort_index()

    buckets: dict[str, list[str]] = {}
    for t, sec in watchlist.SECTORS.items():
        if t in closes.columns:
            buckets.setdefault(sec, []).append(t)

    idx = closes.index
    rows, all_trades = [], []
    start = 0
    while start + FORM + TRADE <= len(idx):
        f_slice = closes.iloc[start:start + FORM]
        t_slice = closes.iloc[start + FORM:start + FORM + TRADE]
        year = t_slice.index[0].year

        for sec, names in buckets.items():
            ok = [n for n in names
                  if f_slice[n].notna().all() and t_slice[n].notna().all()]
            for a, b_ in combinations(sorted(ok), 2):
                la, lb = np.log(f_slice[a].to_numpy()), np.log(f_slice[b_].to_numpy())
                alpha, beta = hedge(la, lb)
                spread = la - alpha - beta * lb
                t_is = adf_t(spread)
                if np.isnan(t_is):
                    continue
                mu, sd = spread.mean(), spread.std()
                if sd <= 0:
                    continue

                tla, tlb = np.log(t_slice[a].to_numpy()), np.log(t_slice[b_].to_numpy())
                t_spread = tla - alpha - beta * tlb
                t_oos = adf_t(t_spread)

                coint_is = t_is < CRIT["5%"]
                rec = {"year": year, "sector": sec, "pair": f"{a}/{b_}",
                       "t_is": t_is, "t_oos": t_oos,
                       "coint_is": coint_is,
                       "coint_oos": bool(t_oos < CRIT["5%"]),
                       "coint_oos_plain": bool(t_oos < CRIT_OOS),
                       "beta": beta}
                rows.append(rec)

                if coint_is:
                    z = (t_spread - mu) / sd
                    ra = t_slice[a].pct_change().fillna(0.0).to_numpy()
                    rb = t_slice[b_].pct_change().fillna(0.0).to_numpy()
                    for tr in trade_spread(z, ra, rb, beta, slip):
                        tr.update({"year": year, "pair": f"{a}/{b_}", "sector": sec})
                        all_trades.append(tr)
        start += TRADE
        if verbose:
            print(f"    window ending {year}: {len(rows)} pair-windows so far", flush=True)

    return {"pairs": pd.DataFrame(rows), "trades": pd.DataFrame(all_trades)}


def clustered_t(tr: pd.DataFrame) -> tuple[float, float, int]:
    """t clustered by trading-window year - pairs inside one year share shocks."""
    if tr.empty:
        return np.nan, np.nan, 0
    yearly = tr.groupby("year")["return"].mean()
    if len(yearly) < 3:
        return float(tr["return"].mean()), np.nan, len(yearly)
    t, _ = stats.ttest_1samp(yearly.to_numpy(), 0.0)
    return float(tr["return"].mean()), float(t), len(yearly)


def main() -> None:
    print("  WITHIN-SECTOR PAIRS  ::  formation 504 bars -> trade 252 bars, rolled")
    print("  Every pair inside every watchlist.SECTORS bucket. No hand-picking.")
    print("  Short leg costs nothing here - no borrow modelled. Optimistic by construction.\n")

    base = run(0.0005, verbose=True)
    p, tr = base["pairs"], base["trades"]

    print("\n" + "=" * 92)
    print("  COINTEGRATION: does an in-sample pass survive out of sample?")
    print("=" * 92)
    n_all = len(p)
    is_pass = p[p["coint_is"]]
    is_fail = p[~p["coint_is"]]
    oos_given_pass = is_pass["coint_oos"].mean()
    oos_given_fail = is_fail["coint_oos"].mean()
    print(f"  pair-windows tested                 {n_all:,}")
    print(f"  cointegrated IN-sample (EG 5%)      {len(is_pass):,}  ({len(is_pass)/n_all*100:.1f}%)")
    print(f"  of those, still cointegrated OOS    {is_pass['coint_oos'].sum():,}  "
          f"({oos_given_pass*100:.1f}%)")
    print(f"  BASE RATE: OOS pass among IS-fails  {is_fail['coint_oos'].sum():,}  "
          f"({oos_given_fail*100:.1f}%)")
    lift = oos_given_pass / oos_given_fail if oos_given_fail > 0 else np.nan
    print(f"  lift from selection                 {lift:.2f}x")
    tt, pv = stats.ttest_ind(is_pass["coint_oos"].astype(float),
                             is_fail["coint_oos"].astype(float), equal_var=False)
    print(f"  difference in proportions           t={tt:.2f}  p={pv:.2e}")
    print(f"  mean OOS ADF t | IS pass            {is_pass['t_oos'].mean():.2f}")
    print(f"  mean OOS ADF t | IS fail            {is_fail['t_oos'].mean():.2f}")
    print(f"  (5% critical value is {CRIT['5%']}; by construction ~5% of random pairs "
          f"pass by chance)")
    a = is_pass["coint_oos_plain"].mean()
    b = is_fail["coint_oos_plain"].mean()
    print(f"\n  Same test at the PLAIN ADF 5% value ({CRIT_OOS}), which is the right")
    print("  threshold out of sample because the hedge ratio is not refitted there:")
    print(f"  OOS pass | IS pass                  {is_pass['coint_oos_plain'].sum():,}  ({a*100:.1f}%)")
    print(f"  OOS pass | IS fail (base rate)      {is_fail['coint_oos_plain'].sum():,}  ({b*100:.1f}%)")
    tt2, pv2 = stats.ttest_ind(is_pass["coint_oos_plain"].astype(float),
                               is_fail["coint_oos_plain"].astype(float), equal_var=False)
    print(f"  lift {a/b:.2f}x   t={tt2:.2f}  p={pv2:.2e}")

    print("\n" + "=" * 92)
    print("  P&L OF THE SELECTED PAIRS  (cost sweep)")
    print("=" * 92)
    print("  {:<14}{:>9}{:>11}{:>9}{:>9}{:>10}{:>9}".format(
        "slippage", "trades", "EV/trade", "t(year)", "win", "med bars", "yrs +"))
    print("  " + "-" * 72)
    for slip in SLIPS:
        r = run(slip)["trades"] if slip != 0.0005 else tr
        ev, t, ny = clustered_t(r)
        win = (r["return"] > 0).mean() if not r.empty else np.nan
        yearly = r.groupby("year")["return"].mean()
        print("  {:<14}{:>9}{:>11}{:>9}{:>9}{:>10}{:>9}".format(
            f"{slip*10000:.0f} bp/side", len(r), C.fmt_pct(ev, 3), C.fmt(t, 2),
            C.fmt_pct(win, 0), f"{r['bars'].median():.0f}" if not r.empty else "n/a",
            f"{(yearly > 0).sum()}/{len(yearly)}"))

    print("\n  BY YEAR @ 5bp/side\n")
    print("  {:<7}{:>9}{:>11}{:>9}{:>12}{:>12}".format(
        "year", "trades", "EV/trade", "win", "pairs sel", "OOS coint%"))
    print("  " + "-" * 60)
    for y in sorted(tr["year"].unique()):
        s = tr[tr["year"] == y]
        pw = p[(p["year"] == y) & p["coint_is"]]
        print("  {:<7}{:>9}{:>11}{:>9}{:>12}{:>12}".format(
            y, len(s), C.fmt_pct(s["return"].mean(), 3),
            C.fmt_pct((s["return"] > 0).mean(), 0), len(pw),
            C.fmt_pct(pw["coint_oos"].mean(), 0) if len(pw) else "n/a"))

    stopped = tr["stopped"].mean() if not tr.empty else np.nan
    print(f"\n  stopped out at |z|>{STOP_Z}: {stopped*100:.1f}% of trades")
    print(f"  loss on stopped trades: {C.fmt_pct(tr[tr['stopped']]['return'].mean(), 2)}"
          f"   vs non-stopped {C.fmt_pct(tr[~tr['stopped']]['return'].mean(), 2)}")


if __name__ == "__main__":
    main()
