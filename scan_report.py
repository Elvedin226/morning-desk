"""Rank the spread scan and expose what ranking by P(win) actually selects for."""

import numpy as np
import pandas as pd

d = pd.read_csv("data_cache/spread_scan.csv")
d = d[np.isfinite(d["pwin"]) & np.isfinite(d["ev"])]
print(f"{len(d)} spreads under $421 across 89 tickers")

print()
print("=== RANKED BY P(win), as asked (slippage capped at 20%) ===")
print()
top = d[d["slip"] <= 0.20].sort_values("pwin", ascending=False).head(15)
# Expiry per row, not just dte: the scan targets ~50 days for each ticker
# separately, so different rows are different dates. A strike pair without its
# expiry is not an order anyone can place.
hdr = (f"  {'ticker':<7}{'type':>5}{'strikes':>12}{'expiry':>13}{'dte':>5}"
       f"{'debit':>7}{'max':>7}{'R:R':>6}{'P(win)':>8}{'EV':>7}{'slip':>6}")
print(hdr)
for _, r in top.iterrows():
    strikes = f"{r['long']:g}/{r['short']:g}"
    print(f"  {r['ticker']:<7}{r['kind']:>5}{strikes:>12}{r['exp']:>13}{r['dte']:>5}"
          f"{r['debit']:>7,.0f}{r['max']:>7,.0f}{r['rr']:>6.2f}{r['pwin']*100:>7.0f}%"
          f"{r['ev']:>7,.0f}{r['slip']*100:>5.0f}%")

print()
print("=== WHAT HIGH P(win) COSTS YOU ===")
print()
d["bucket"] = pd.cut(d["pwin"], [0, .3, .4, .5, .6, .7, 1.0],
                     labels=["<30%", "30-40%", "40-50%", "50-60%", "60-70%", ">70%"])
print(f"  {'P(win)':<10}{'n':>6}{'median R:R':>13}{'median EV':>12}{'median EV%':>13}")
for b, g in d.groupby("bucket", observed=True):
    print(f"  {str(b):<10}{len(g):>6}{g['rr'].median():>13.2f}"
          f"{g['ev'].median():>12,.0f}{(g['ev'] / g['debit']).median() * 100:>12.0f}%")

print()
print(f"  corr(P(win), R:R) = {np.corrcoef(d['pwin'], d['rr'])[0, 1]:+.2f}")
med_ev = d["ev"].median()
print(f"  median EV across ALL {len(d)} spreads: ${med_ev:,.0f} "
      f"({(d['ev'] / d['debit']).median() * 100:.1f}% of debit)")
print(f"  share with positive modelled EV: {(d['ev'] > 0).mean() * 100:.0f}%")

print()
print("=== EXPECTED VALUE, BEST AND WORST ===")
print()
for label, sub in [("best EV", d.nlargest(5, "ev")), ("worst EV", d.nsmallest(5, "ev"))]:
    print(f"  {label}:")
    for _, r in sub.iterrows():
        strikes = f"{r['long']:g}/{r['short']:g}"
        print(f"    {r['ticker']:<7}{r['kind']:>5}{strikes:>12}{r['exp']:>13}"
              f"  debit {r['debit']:>5,.0f}  EV {r['ev']:>+6,.0f}"
              f"  ({r['ev'] / r['debit'] * 100:+.0f}%)  slip {r['slip'] * 100:.0f}%")
