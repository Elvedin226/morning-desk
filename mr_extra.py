"""Follow-up cuts that the headline tables cannot answer.

1. ERA SPLIT. The two biases that burned this project were universe selection
   and period bias. The by-year tables rule out "one year carried it"; this
   asks the coarser question - is the edge a pre-2011 artefact? 2011 is the
   Connors publication line, so it doubles as the post-publication split.

2. INDEXES vs SINGLE NAMES. The IBS effect is documented on ETFs. The brief
   asks whether it survives on single names, which is a different claim.

3. THE -4% OVERNIGHT TAIL. The overnight run turned up one condition that
   survives 5bp: buy the close after a day down 4% or more. That condition was
   added by me while looking at the data, so it is exactly the kind of thing
   that needs a period check before anyone calls it a finding.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

import mr_core as C
import mr_overnight

CONFIGS = [
    ("connors_rsi2", {}),
    ("mr_cumrsi", {"days": 2, "entry": 35}),
    ("mr_cumrsi_nt", {"days": 2, "entry": 35}),
    ("mr_vwap", {"window": 20, "k": 1.5}),
    ("mr_ibs", {"entry": 0.2, "exit": 0.8}),
    ("mr_ndown", {"k": 3}),
]
SLIP = 0.0005


def slice_data(data, start=None, end=None, keep=None):
    out = {}
    for t, d in data.items():
        if keep is not None and t not in keep:
            continue
        s = d
        if start:
            s = s[s.index >= start]
        if end:
            s = s[s.index < end]
        if len(s) >= 500:
            out[t] = s
    return out


def table(title, rows):
    print(f"\n\n{'='*98}\n  {title}\n{'='*98}")
    print("  {:<28}{:>8}{:>10}{:>10}{:>9}{:>11}{:>10}".format(
        "strategy / slice", "trades", "EV/trade", "t(clust)", "expo",
        "port CAGR", "B&H CAGR"))
    print("  " + "-" * 96)
    for label, s in rows:
        print("  {:<28}{:>8}{:>10}{:>10}{:>9}{:>11}{:>10}".format(
            label[:28], s["total_trades"], C.fmt_pct(s["ev_trade"], 3),
            C.fmt(s["t_clustered"]), C.fmt_pct(s["med_exposure"], 0),
            C.fmt_pct(s["port_cagr"]), C.fmt_pct(s["bench_cagr"])))


def main() -> None:
    data = C.universe()
    indexes = {"SPY", "QQQ", "IWM"}
    singles = set(data) - indexes

    rows = []
    for name, p in CONFIGS:
        pre = C.evaluate(name, p, SLIP, slice_data(data, end="2011-01-01"))
        post = C.evaluate(name, p, SLIP, slice_data(data, start="2011-01-01"))
        rows.append((f"{name} pre-2011", pre["summary"]))
        rows.append((f"{name} 2011-now", post["summary"]))
    table("ERA SPLIT @ 5bp/side  (2011 = the Connors publication line)", rows)

    rows = []
    for name, p in CONFIGS:
        idx = C.evaluate(name, p, SLIP, slice_data(data, keep=indexes))
        sng = C.evaluate(name, p, SLIP, slice_data(data, keep=singles))
        rows.append((f"{name} indexes(3)", idx["summary"]))
        rows.append((f"{name} singles({len(sng['per_ticker'])})", sng["summary"]))
    table("INDEXES vs SINGLE NAMES @ 5bp/side", rows)

    # 3. the -4% overnight tail, by period
    n = mr_overnight.nights(data)
    n["net"] = n["gross"] - 2 * 0.0005
    n["year"] = pd.to_datetime(n["date"]).dt.year
    tail = n[n["day_ret"] <= -0.04]
    print(f"\n\n{'='*98}\n  OVERNIGHT AFTER A -4% DAY, @5bp/side  (a condition I added "
          f"while looking - check it)\n{'='*98}")
    print("  {:<12}{:>10}{:>10}{:>11}{:>9}".format("period", "nights", "dates", "EV/night", "t(date)"))
    print("  " + "-" * 54)
    for lo, hi, lab in [(2000, 2010, "2000-2010"), (2011, 2026, "2011-2026"),
                        (2000, 2026, "all")]:
        s = tail[(tail["year"] >= lo) & (tail["year"] <= hi)]
        ev, t, nn, nd = mr_overnight.clustered(s, "net")
        print("  {:<12}{:>10}{:>10}{:>11}{:>9}".format(lab, nn, nd, C.fmt_pct(ev, 3), C.fmt(t)))
    # how concentrated is it? drop the single best year and re-measure.
    yearly = tail.groupby("year")["net"].mean()
    best = yearly.idxmax()
    s = tail[tail["year"] != best]
    ev, t, nn, nd = mr_overnight.clustered(s, "net")
    print(f"\n  best year is {best} (EV {yearly.max()*100:.3f}%). Dropping it: "
          f"EV {ev*100:.3f}%  t={t:.2f}  ({nn:,} nights)")
    crisis = tail[tail["year"].isin([2008, 2009, 2020])]
    calm = tail[~tail["year"].isin([2008, 2009, 2020])]
    for lab, s in (("2008/09/2020 only", crisis), ("excluding those 3", calm)):
        ev, t, nn, nd = mr_overnight.clustered(s, "net")
        print(f"  {lab:<20} EV {ev*100:>7.3f}%  t={t:>5.2f}  {nn:>7,} nights")
    yc = yearly.sort_values()
    print(f"  positive in {(yearly > 0).sum()}/{len(yearly)} years; "
          f"worst {yc.index[0]} {yc.iloc[0]*100:.2f}%, best {best} {yc.iloc[-1]*100:.2f}%")


if __name__ == "__main__":
    main()
