"""0DTE: buy at the open, settle at the close. What are the actual odds?

I previously said 0DTE could not be honestly tested without intraday option
quotes. That was over-stated, and the correction matters: a 0DTE contract's ONLY
trading day is expiry day, so an open-to-expiry trade needs exactly two things -
an entry price at the open, and settlement, which is just intrinsic value. No
intraday path required.

What is still modelled rather than observed is the ENTRY PREMIUM. Black-Scholes
with one day to expiry is a poor description of a 0DTE option (gamma explodes,
charm dominates, and implied vol collapses into the close), so the premium here
is an estimate and the result is only as good as it. That is why iv_mult is
swept rather than fixed - the honest output is a range, not a number.

SPREADS ARE THE OTHER HALF, and they are measured, not assumed. Live SPY 0DTE
quotes show the bid-ask spread as a share of premium rising sharply as the
contract gets cheaper: roughly 1-1.4% near the money, but 25-40% on the $0.02
to $0.04 contracts a small account can actually buy in quantity. A 30% round
trip tax applied to a lottery ticket is most of the reason far-OTM measured
-99.7% in the earlier study.

WHY THIS IS THE ONLY SIDE TESTED HERE: selling 0DTE is the side with measured
positive expectancy, and it is structurally unavailable at this account size -
credit spreads and naked short options need Level 3 approval and a margin
account, which needs $2,000. A $421 cash account gets Level 2: long calls and
puts only. So the accessible side is the side every study measures as losing.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from ivcheck import _bs_call, _bs_put

START = "2018-01-01"
DAY = 1 / 252

# Spread as a fraction of premium, by how cheap the contract is. Measured from
# live SPY 0DTE quotes: cheap contracts are where the tax is brutal.
def spread_cost(prem: float, spot: float) -> float:
    r = prem / spot
    if r < 0.0005:
        return 0.325      # sub-$0.05 lottery tickets: 25-40% round trip
    if r < 0.0015:
        return 0.15
    if r < 0.004:
        return 0.05
    return 0.014          # near the money


def payoffs(ticker="SPY", moneyness=1.002, iv_mult=1.0, kind="call") -> np.ndarray:
    """Buy at the open on expiry day, settle at the close. Returns per trade.

    SPY has expiries every weekday, so every session is a 0DTE session.
    """
    df = yf.download(ticker, start=START, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    rv = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)

    out = []
    price = _bs_call if kind == "call" else _bs_put
    for i in range(30, len(df)):
        o, c = float(df["Open"].iloc[i]), float(df["Close"].iloc[i])
        vol = float(rv.iloc[i - 1])
        if not (np.isfinite(vol) and vol > 0.03):
            continue
        k = o * moneyness if kind == "call" else o * (2 - moneyness)
        prem = price(o, k, vol * iv_mult, DAY)
        if not np.isfinite(prem) or prem <= 0.005:
            continue
        intrinsic = max(0.0, c - k) if kind == "call" else max(0.0, k - c)
        gross = intrinsic / prem - 1.0
        out.append((1 + gross) * (1 - spread_cost(prem, o)) - 1)
    return np.array(out)


def run_paths(dist, n_days, frac, start=421.0, target=1000.0, ruin=150.0,
              paths=20000, seed=0):
    rng = np.random.default_rng(seed)
    eq = np.full(paths, start)
    hit = np.zeros(paths, bool)
    dead = np.zeros(paths, bool)
    for _ in range(n_days):
        live = ~(hit | dead)
        if not live.any():
            break
        r = rng.choice(dist, size=int(live.sum()))
        eq[live] = eq[live] * (1 - frac) + eq[live] * frac * (1 + r)
        hit |= eq >= target
        dead |= eq < ruin
    return float(hit.mean()), float(dead.mean()), float(np.median(eq))


def main():
    print("\n  0DTE SPY, BOUGHT AT THE OPEN, SETTLED AT THE CLOSE")
    print("  Premium is Black-Scholes (an estimate); the underlying move and the")
    print("  spread haircut are real. iv_mult is swept because BS misprices 0DTE.")
    print("  " + "=" * 76)
    print(f"  {'strike':>9}{'iv_mult':>9}{'n':>7}{'EV/trade':>11}{'win%':>7}"
          f"{'total loss':>12}{'best':>8}")
    print("  " + "-" * 76)

    grid = {}
    for mny in (1.000, 1.002, 1.005, 1.010):
        for ivm in (0.9, 1.0, 1.2):
            r = payoffs(moneyness=mny, iv_mult=ivm)
            if len(r) < 200:
                continue
            grid[(mny, ivm)] = r
            print(f"  {(mny-1)*100:>8.1f}%{ivm:>9.1f}{len(r):>7,}{r.mean()*100:>10.1f}%"
                  f"{np.mean(r > 0)*100:>6.1f}%{np.mean(r <= -0.999)*100:>11.1f}%"
                  f"{r.max():>7.1f}x")

    print("\n  P(REACH $1,000 BY 31 DEC) FROM $421, 85 SESSIONS")
    print("  " + "=" * 76)
    print(f"  {'strike':>9}{'iv_mult':>9}{'per trade':>11}{'P(hit)':>9}"
          f"{'P(<$150)':>11}{'median':>10}")
    print("  " + "-" * 76)
    for (mny, ivm), r in grid.items():
        if ivm != 1.0:
            continue
        for frac in (0.20, 0.50, 1.00):
            h, d, m = run_paths(r, 85, frac)
            print(f"  {(mny-1)*100:>8.1f}%{ivm:>9.1f}{frac*100:>10.0f}%"
                  f"{h*100:>8.1f}%{d*100:>10.1f}%{m:>10,.0f}")

    print("\n  HOW MUCH OF THIS IS THE SPREAD?")
    print("  " + "=" * 76)
    df = yf.download("SPY", start=START, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    rv = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    gross = []
    for i in range(30, len(df)):
        o, c = float(df["Open"].iloc[i]), float(df["Close"].iloc[i])
        v = float(rv.iloc[i - 1])
        if not (np.isfinite(v) and v > 0.03):
            continue
        k = o * 1.002
        prem = _bs_call(o, k, v, DAY)
        # NaN fails every comparison, so `prem <= 0.005` lets it through and
        # poisons the mean. Check finiteness explicitly.
        if not np.isfinite(prem) or prem <= 0.005:
            continue
        gross.append(max(0.0, c - k) / prem - 1.0)
    gross = np.array(gross)
    net = grid.get((1.002, 1.0))
    print(f"     gross of spreads   EV {gross.mean()*100:>7.1f}%   win {np.mean(gross>0)*100:.1f}%")
    if net is not None:
        print(f"     net of spreads     EV {net.mean()*100:>7.1f}%   win {np.mean(net>0)*100:.1f}%")
        print(f"     the spread alone costs {(gross.mean()-net.mean())*100:.1f} points of EV")


if __name__ == "__main__":
    main()
