"""Strategies 1-4: parameter sweeps, cost sweeps and per-year breakdowns.

Run: python mr_run.py            (everything)
     python mr_run.py cumrsi     (one section)
"""

from __future__ import annotations

import sys
import itertools

import numpy as np
import pandas as pd

import mr_core as C

BASELINE = ("connors_rsi2", {})

SPECS = {
    "cumrsi": ("mr_cumrsi",
               [{"days": d, "entry": e} for d, e in itertools.product([2, 3], [35, 45, 60])],
               {"days": 2, "entry": 35}),
    "cumrsi_nt": ("mr_cumrsi_nt",
                  [{"days": d, "entry": e} for d, e in itertools.product([2, 3], [35, 45, 60])],
                  {"days": 2, "entry": 35}),
    "vwap": ("mr_vwap",
             [{"window": w, "k": k} for w, k in itertools.product([10, 20, 50], [1.0, 1.5, 2.0])],
             {"window": 20, "k": 1.5}),
    "ibs": ("mr_ibs", [{"entry": 0.2, "exit": 0.8}], {"entry": 0.2, "exit": 0.8}),
    "ndown": ("mr_ndown", [{"k": k} for k in (2, 3, 4)], {"k": 3}),
}

HDR = ("  {:<22}{:>7}{:>8}{:>9}{:>9}{:>8}{:>9}{:>8}{:>9}{:>8}"
       .format("config", "trades", "expo", "EV/trd", "t(clust)", "win",
               "portCAGR", "pSharpe", "medCAGR", "beatBH"))


def line(label: str, s: dict) -> str:
    return ("  {:<22}{:>7}{:>8}{:>9}{:>9}{:>8}{:>9}{:>8}{:>9}{:>8}".format(
        label[:22], s["total_trades"], C.fmt_pct(s["med_exposure"], 0),
        C.fmt_pct(s["ev_trade"], 3), C.fmt(s["t_clustered"], 2),
        C.fmt_pct(s["med_win_rate"], 0), C.fmt_pct(s["port_cagr"], 2),
        C.fmt(s["port_sharpe"], 2), C.fmt_pct(s["med_cagr"], 2),
        C.fmt_pct(s["beat_bh"], 0)))


def section(title: str, name: str, combos: list[dict], primary: dict) -> None:
    print(f"\n\n{'='*110}\n  {title}\n{'='*110}")

    print("\n  PARAMETER SWEEP @ 5bp/side  (all configs shown - the best one is a"
          "\n  selection, not a result)\n")
    print(HDR)
    print("  " + "-" * 108)
    for p in combos:
        res = C.evaluate(name, p, 0.0005)
        print(line(", ".join(f"{k}={v}" for k, v in p.items()), res["summary"]))

    print("\n  COST SWEEP, primary config " + str(primary) + "\n")
    print(HDR)
    print("  " + "-" * 108)
    keep = None
    for slip in C.SLIPS:
        res = C.evaluate(name, primary, slip)
        if slip == 0.0005:
            keep = res
        print(line(f"{slip*10000:.0f} bp/side", res["summary"]))
    s = keep["summary"]
    print(f"\n  buy & hold, same window : portfolio CAGR {C.fmt_pct(s['bench_cagr'])}"
          f"   Sharpe {C.fmt(s['bench_sharpe'])}   maxDD {C.fmt_pct(s['bench_maxdd'],1)}")
    print(f"  strategy  @ 5bp         : portfolio CAGR {C.fmt_pct(s['port_cagr'])}"
          f"   Sharpe {C.fmt(s['port_sharpe'])}   maxDD {C.fmt_pct(s['port_maxdd'],1)}")

    print("\n  BY YEAR @ 5bp/side\n")
    print("  {:<7}{:>11}{:>11}{:>9}{:>10}{:>8}".format(
        "year", "port ret", "B&H ret", "trades", "EV/trade", "t"))
    print("  " + "-" * 56)
    yr = C.by_year(keep)
    for _, r in yr.iterrows():
        print("  {:<7}{:>11}{:>11}{:>9}{:>10}{:>8}".format(
            int(r["year"]), C.fmt_pct(r["port_ret"]), C.fmt_pct(r["bh_ret"]),
            int(r["trades"]), C.fmt_pct(r["ev_trade"], 3), C.fmt(r["t"], 2)))
    beat = (yr["port_ret"] > yr["bh_ret"]).mean()
    pos = (yr["ev_trade"] > 0).mean()
    print(f"\n  beat B&H in {beat*100:.0f}% of years   positive EV in {pos*100:.0f}% of years"
          f"   ({len(yr)} years)")


def main() -> None:
    which = sys.argv[1:] or ["baseline"] + list(SPECS)
    data = C.universe()
    print(f"  UNIVERSE: {len(data)} tickers (watchlist.SECTORS + SPY/QQQ/IWM), "
          f"{min(d.index[0] for d in data.values()).date()} to "
          f"{max(d.index[-1] for d in data.values()).date()}")
    print("  Survivorship-biased (today's large caps). ~5pp/yr of overstatement "
          "measured elsewhere in this project.")

    if "baseline" in which:
        section("BASELINE TO BEAT: Connors RSI-2 (already known: p=0.05, "
                "loses to buy & hold)", BASELINE[0], [{}], {})
    for key in which:
        if key in SPECS:
            name, combos, primary = SPECS[key]
            section(f"{key.upper()}  [{name}]", name, combos, primary)


if __name__ == "__main__":
    main()
