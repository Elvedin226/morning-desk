"""How should a small account size convex (lottery-shaped) bets?

"We just need a couple of big trades" is a sizing question, not a strategy. A
bet can have positive expected value and still bankrupt you, because you must
survive long enough for the rare winner to arrive. This simulates that.

Payoff model: each bet either loses 100% (the option expires worthless) or pays
`multiple` x. Real option payoffs are continuous, but total-loss-or-jackpot is
the honest caricature of far-OTM calls and it keeps the sizing lesson visible.
"""

from __future__ import annotations

import numpy as np

START = 4_000.0
RUIN = 400.0  # 90% drawdown — below this you cannot meaningfully continue


def simulate(win_prob, multiple, fraction, bets=50, runs=20_000, seed=0):
    """Bet `fraction` of current equity each time, `bets` times, `runs` paths."""
    rng = np.random.default_rng(seed)
    equity = np.full(runs, START)
    alive = np.ones(runs, dtype=bool)

    for _ in range(bets):
        stake = equity * fraction
        won = rng.random(runs) < win_prob
        equity = np.where(alive, equity - stake + np.where(won, stake * multiple, 0.0), equity)
        alive &= equity > RUIN

    ev = win_prob * multiple - 1
    return {
        "ev_per_bet": ev,
        "ruin_rate": float((~alive).mean()),
        "median": float(np.median(equity)),
        "mean": float(equity.mean()),
        "p90": float(np.percentile(equity, 90)),
        "beat_start": float((equity > START).mean()),
    }


def table(win_prob, multiple, label):
    ev = win_prob * multiple - 1
    print(f"\n{label}")
    print(f"  {win_prob:.0%} chance of {multiple}x, else total loss  ->  EV {ev:+.0%} per bet")
    print(f"  50 bets from ${START:,.0f}, ruin = falling below ${RUIN:,.0f}\n")
    print(f"  {'bet size':<12}{'ruin':>8}{'median end':>14}{'90th pct':>14}{'beat start':>12}")
    for fraction in (1.0, 0.5, 0.25, 0.10, 0.05, 0.02):
        r = simulate(win_prob, multiple, fraction)
        print(f"  {fraction:>7.0%}{'':<5}{r['ruin_rate']:>7.1%}{r['median']:>13,.0f}{r['p90']:>14,.0f}{r['beat_start']:>11.1%}")


if __name__ == "__main__":
    # Optimistic: the payoff you would need for "a couple of big trades" to work.
    table(0.10, 15, "SCENARIO A — strongly positive EV (optimistic)")
    # Coval & Shumway put the average far-OTM call return near -96%.
    table(0.04, 15, "SCENARIO B — matching the documented average (-40% EV)")
