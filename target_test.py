"""$421 -> $1,000 by 31 Dec. What are the actual odds, per approach?

This asks a different question from every other test in this project, and the
difference is the whole point.

Everything else here measured EXPECTED RETURN - what compounds best over time.
That is the right objective for growing capital, and it is the WRONG objective
for hitting a target by a deadline. The two recommend opposite behaviour:

    maximise expected growth  ->  small positions, diversify, avoid ruin
    maximise P(hit target)    ->  concentrate, accept ruin risk

Dubins & Savage (1965): in a SUBFAIR game, bold play - betting the largest
amount that is useful - maximises the probability of reaching a target before
going broke. Timid play is optimal only when the game is favourable. With an
edge near zero after costs, spreading risk thin does not protect the goal, it
guarantees missing it. Diversification lowers variance, and variance is the only
thing that can carry $421 to $1,000 in four months without an edge.

So this file reports two numbers per approach and refuses to collapse them:
    P(reach $1,000 by 31 Dec)
    P(end under $150)          - the price of the attempt

WHAT IS REAL HERE
    The equity trade distribution is the 1,719 actual Connors RSI-2 trades from
    basket_test.py, resampled. Not assumed, measured.
    The option payoffs are Black-Scholes entries against ACTUAL realised moves
    of real tickers - so the fat right tail and the frequent total loss are both
    empirical, not invented.

WHAT IS NOT
    Option IV is estimated as realised vol times a variance-risk-premium
    multiplier, because historical intraday option quotes are not available for
    free. That multiplier is the single most important assumption in the file
    and it is swept, not fixed, for exactly that reason.
"""

from __future__ import annotations

import warnings
from datetime import date

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from ivcheck import _bs_call

START_EQUITY = 421.0
TARGET = 1000.0
RUIN = 150.0
DEADLINE = date(2026, 12, 31)
TODAY = date(2026, 9, 3)
PATHS = 20000

SIGNALS_PER_DAY = 0.44        # measured: 110 RSI-2 trades a year across 90 names
COST = 0.0005

# Index ETFs, NOT hand-picked single names.
#
# The first version of this file used NVDA/AMD/TSLA/PLTR/COIN/SMCI/ARM/IONQ and
# produced an option EV of +33% per trade - a money printer, and obviously wrong.
# The cause was me: that list is the set of names that went up the most between
# 2018 and 2026, chosen with hindsight. Measured across universes:
#
#     cherry-picked vol names   EV  +33.0%
#     broad 90-name universe    EV  +15.6%   (still today's survivors)
#     SPY + QQQ + IWM           EV  -35.9%
#     SPY alone                 EV  -63.1%
#
# Index ETFs are the only arm with no single-name selection in it, so they are
# what the headline numbers use. Buying OTM calls loses money on average. The
# question this file asks is whether it still maximises P(hit target) anyway -
# which is a genuinely different question, and the answer can be yes.
VOL_NAMES = ["SPY", "QQQ", "IWM"]


def trading_days() -> int:
    return int(np.busday_count(TODAY.isoformat(), DEADLINE.isoformat()))


def option_payoffs(hold_days=21, moneyness=1.05, iv_mult=1.25, n_names=None) -> np.ndarray:
    """Empirical return distribution of buying a short-dated OTM call.

    For every historical date and ticker: price the call with Black-Scholes on
    the trailing realised vol scaled by `iv_mult`, then settle it against what
    the stock ACTUALLY did over the next `hold_days`. The stock move is real, so
    the tail is real; only the entry premium is modelled.

    iv_mult is the variance risk premium - implied vol usually exceeds
    subsequent realised vol, which is precisely why buying options is a losing
    proposition on average. Setting it to 1.0 would quietly hand the buyer free
    money and is the most common way this kind of study lies.
    """
    names = VOL_NAMES if n_names is None else VOL_NAMES[:n_names]
    raw = yf.download(names, start="2018-01-01", auto_adjust=True,
                      progress=False, group_by="ticker")
    lvl0 = set(raw.columns.get_level_values(0)) if isinstance(raw.columns, pd.MultiIndex) else set()

    out = []
    for t in names:
        if t not in lvl0:
            continue
        c = raw[t]["Close"].dropna()
        if len(c) < 300:
            continue
        rv = c.pct_change().rolling(20).std() * np.sqrt(252)
        fwd = c.shift(-hold_days)
        yrs = hold_days / 252
        for i in range(60, len(c) - hold_days):
            spot, vol, end = float(c.iloc[i]), float(rv.iloc[i]), float(fwd.iloc[i])
            if not (np.isfinite(vol) and vol > 0.05 and np.isfinite(end)):
                continue
            strike = spot * moneyness
            prem = _bs_call(spot, strike, vol * iv_mult, yrs)
            if prem <= 0.01:
                continue
            payoff = max(0.0, end - strike)
            out.append(payoff / prem - 1.0)
    return np.array(out)


def simulate(kind, dist, days, frac, paths=PATHS, seed=0,
             signals_per_day=SIGNALS_PER_DAY):
    """Monte Carlo of the account to the deadline.

    `frac` is the fraction of equity committed per trade. It is deliberately
    allowed to reach 1.0 - the whole question is what concentration does to the
    two probabilities, and capping it at a 'prudent' level would assume the
    answer.
    """
    rng = np.random.default_rng(seed)
    eq = np.full(paths, START_EQUITY)
    hit = np.zeros(paths, dtype=bool)
    dead = np.zeros(paths, dtype=bool)

    for _ in range(days):
        live = ~(hit | dead)
        if not live.any():
            break
        # Whether a signal appears today. Options are assumed available daily;
        # equity signals fire at the measured rate.
        if kind == "equity":
            take = live & (rng.random(paths) < signals_per_day)
        else:
            take = live.copy()
        n = int(take.sum())
        if n == 0:
            continue
        r = rng.choice(dist, size=n) - 2 * COST
        stake = eq[take] * frac
        eq[take] = eq[take] - stake + stake * (1 + r)
        hit |= eq >= TARGET
        dead |= eq < RUIN

    return {"p_target": float(hit.mean()), "p_ruin": float(dead.mean()),
            "median": float(np.median(eq)), "mean": float(eq.mean()),
            "p90": float(np.percentile(eq, 90))}


def main() -> None:
    days = trading_days()
    print(f"\n  {TODAY} -> {DEADLINE}   {days} trading days")
    print(f"  ${START_EQUITY:,.0f} -> ${TARGET:,.0f} is {TARGET/START_EQUITY:.2f}x, "
          f"or {(TARGET/START_EQUITY)**(1/days)-1:.3%} a day compounded\n")

    rsi2 = np.load("data_cache/rsi2_returns.npy")
    print(f"  equity engine : {len(rsi2):,} measured RSI-2 trades, "
          f"EV {rsi2.mean():+.3%}/trade, win {np.mean(rsi2>0):.1%}")

    opt = option_payoffs()
    print(f"  option engine : {len(opt):,} modelled 21d 5%-OTM calls, "
          f"EV {opt.mean():+.1%}/trade, win {np.mean(opt>0):.1%}, "
          f"best {opt.max():.0f}x")
    print(f"                  total losses: {np.mean(opt <= -0.999):.1%} of trades")

    print("\n\n  A. EQUITY - Connors RSI-2, varying concentration")
    print("  " + "=" * 74)
    print(f"  {'per trade':>10}{'P(hit $1000)':>15}{'P(<$150)':>11}"
          f"{'median':>10}{'90th pct':>11}")
    for f in (0.10, 0.20, 0.33, 0.50, 0.75, 1.00):
        r = simulate("equity", rsi2, days, f)
        print(f"  {f*100:>9.0f}%{r['p_target']*100:>14.1f}%{r['p_ruin']*100:>10.1f}%"
              f"{r['median']:>10,.0f}{r['p90']:>11,.0f}")

    print("\n  B. OPTIONS - 21-day 5% OTM calls, varying concentration")
    print("  " + "=" * 74)
    print(f"  {'per trade':>10}{'P(hit $1000)':>15}{'P(<$150)':>11}"
          f"{'median':>10}{'90th pct':>11}")
    for f in (0.05, 0.10, 0.20, 0.33, 0.50, 1.00):
        r = simulate("option", opt, days, f)
        print(f"  {f*100:>9.0f}%{r['p_target']*100:>14.1f}%{r['p_ruin']*100:>10.1f}%"
              f"{r['median']:>10,.0f}{r['p90']:>11,.0f}")

    print("\n  C. HOW MUCH THE IV ASSUMPTION MATTERS  (options at 20% per trade)")
    print("  " + "=" * 74)
    print("  iv_mult is what you pay over subsequent realised vol. 1.0 = free")
    print("  options, 1.25 = roughly the historical variance risk premium.")
    print(f"  {'iv_mult':>10}{'EV/trade':>12}{'P(hit $1000)':>15}{'P(<$150)':>11}")
    for m in (1.00, 1.10, 1.25, 1.40):
        o = option_payoffs(iv_mult=m, n_names=8)
        r = simulate("option", o, days, 0.20)
        print(f"  {m:>10.2f}{o.mean()*100:>11.1f}%{r['p_target']*100:>14.1f}%"
              f"{r['p_ruin']*100:>10.1f}%")

    print("\n  D. MIXED - equity core, small convex sleeve")
    print("  " + "=" * 74)
    print("  Run the equity engine at 33% and additionally buy a call with a")
    print("  small slice whenever one is available.")
    rng = np.random.default_rng(7)
    for sleeve in (0.02, 0.05, 0.10):
        eq = np.full(PATHS, START_EQUITY)
        hit = np.zeros(PATHS, bool); dead = np.zeros(PATHS, bool)
        for _ in range(days):
            live = ~(hit | dead)
            if not live.any():
                break
            t = live & (rng.random(PATHS) < SIGNALS_PER_DAY)
            if t.any():
                r = rng.choice(rsi2, size=int(t.sum())) - 2 * COST
                st = eq[t] * 0.33
                eq[t] = eq[t] - st + st * (1 + r)
            ro = rng.choice(opt, size=int(live.sum())) - 2 * COST
            st = eq[live] * sleeve
            eq[live] = eq[live] - st + st * (1 + ro)
            hit |= eq >= TARGET; dead |= eq < RUIN
        print(f"  sleeve {sleeve*100:>3.0f}%   P(hit) {hit.mean()*100:>5.1f}%   "
              f"P(<$150) {dead.mean()*100:>5.1f}%   median ${np.median(eq):>7,.0f}")


if __name__ == "__main__":
    main()
