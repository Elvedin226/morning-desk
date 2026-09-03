"""Build the phone dashboard.

    .venv/Scripts/python.exe publish.py

Runs the swing bot and the pre-market scan, then writes dashboard.html - a
self-contained page sized for a phone. Claude publishes that file as a private
Artifact; you add it to your home screen and it behaves like an app.

SNAPSHOT, NOT LIVE. The page shows what your PC computed the last time this ran.
It cannot fetch prices itself. The "as of" stamp is displayed prominently for
exactly that reason - a stale dashboard that looks live is worse than no
dashboard. Re-run this, then ask Claude to republish, to refresh it.
"""

from __future__ import annotations

import html
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import bot
import premarket
import watchlist

OUT = Path(__file__).parent / "dashboard.html"
ACCOUNT = 421.0
RISK = 0.01


# ---------------------------------------------------------------------------
# Charts, drawn as inline SVG from the same data as the numbers.
#
# Not live: an artifact cannot reach an external host, so there is no widget and
# no quote fetch at view time. These are rendered when publish.py runs, which
# means they are exactly as fresh as everything else on the page - and the "as
# of" stamp covers them too.
# ---------------------------------------------------------------------------

def _scale(vals, lo, hi, size, invert=False):
    if hi - lo < 1e-9:
        return [size / 2] * len(vals)
    out = [(v - lo) / (hi - lo) * size for v in vals]
    return [size - o for o in out] if invert else out


def sparkline(series: pd.Series, w: int = 88, h: int = 26) -> str:
    """Tiny price trace. Endpoint marked, because that is where the eye goes."""
    v = series.dropna().to_numpy()[-63:]
    if len(v) < 5:
        return ""
    lo, hi = float(v.min()), float(v.max())
    xs = np.linspace(1, w - 1, len(v))
    ys = _scale(v, lo, hi, h - 4, invert=True)
    ys = [y + 2 for y in ys]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    rising = v[-1] >= v[0]
    stroke = "var(--go)" if rising else "var(--stop)"
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'aria-hidden="true"><polyline points="{pts}" fill="none" '
            f'stroke="{stroke}" stroke-width="1.4" stroke-linejoin="round"/>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="2" fill="{stroke}"/></svg>')


def candidate_chart(series: pd.Series, stop: float, entry: float,
                    w: int = 460, h: int = 170) -> str:
    """Six months of price with both averages and the stop drawn in.

    The stop line is the point of the chart: it shows how far the idea can go
    against you before it is wrong, which is what sets the position size.
    """
    s = series.dropna().iloc[-126:]
    if len(s) < 30:
        return ""
    price = s.to_numpy()
    sma20 = s.rolling(20).mean().to_numpy()
    sma50 = s.rolling(50).mean().to_numpy()

    lo = float(min(np.nanmin(price), stop)) * 0.985
    hi = float(max(np.nanmax(price), entry)) * 1.015
    pad_t, pad_b = 8, 8
    inner = h - pad_t - pad_b
    xs = np.linspace(2, w - 2, len(price))

    def line(vals, color, width, dash=""):
        ok = [(x, v) for x, v in zip(xs, vals) if np.isfinite(v)]
        if len(ok) < 2:
            return ""
        ys = _scale([v for _, v in ok], lo, hi, inner, invert=True)
        pts = " ".join(f"{x:.1f},{y + pad_t:.1f}" for (x, _), y in zip(ok, ys))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="{width}"{d} stroke-linejoin="round"/>')

    def hline(val, color, dash="4 3"):
        y = _scale([val], lo, hi, inner, invert=True)[0] + pad_t
        return (f'<line x1="2" x2="{w-2}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" '
                f'stroke-width="1" stroke-dasharray="{dash}"/>')

    return (
        f'<svg class="chart" viewBox="0 0 {w} {h}" width="100%" height="{h}" '
        f'role="img" aria-label="Six-month price with 20 and 50 day averages and the stop level">'
        + hline(stop, "var(--stop)")
        + line(sma50, "var(--ink-faint)", 1.1)
        + line(sma20, "var(--accent)", 1.2)
        + line(price, "var(--ink)", 1.7)
        + "</svg>")


def gather() -> dict:
    tickers = sorted(set(list(watchlist.SECTORS) + list(watchlist.SECTOR_ETFS) + ["SPY"]))
    close = watchlist._download(tickers, "2023-01-01")
    reg = watchlist.regime(close["SPY"].dropna())

    sectors = watchlist.sector_strength(close)
    leaders = set(sectors.head(4)["etf"])
    rows = []
    for t in watchlist.SECTORS:
        if t not in close.columns:
            continue
        c = bot.evaluate(close, t)
        if c:
            c["leading_sector"] = watchlist.SECTORS[t] in leaders
            rows.append(c)
    passing = sorted([c for c in rows if c["passes"]],
                     key=lambda c: (-c["leading_sector"], -c["momentum"]))

    chosen, blocked = None, []
    for c in passing[:8]:
        days = bot.earnings_days_away(c["ticker"])
        c["earnings_days"] = days
        if days is not None and 0 <= days <= bot.HOLD_DAYS:
            blocked.append((c["ticker"], days))
            continue
        chosen = c
        break

    pos = bot.size_position(chosen, ACCOUNT, RISK) if chosen else None
    gaps = premarket.scan(premarket.DEFAULT_UNIVERSE, 3.0)

    return {
        "stamp": datetime.now().strftime("%a %d %b %Y, %H:%M"),
        "regime": reg, "sectors": sectors, "candidate": chosen,
        "position": pos, "passing": passing, "blocked": blocked, "gaps": gaps,
        "close": close,
    }


def _rows_sectors(sectors: pd.DataFrame) -> str:
    out = []
    for i, r in sectors.head(6).iterrows():
        lead = ' class="lead"' if i < 4 else ""
        out.append(
            f'<tr{lead}><td>{html.escape(r["sector"])}</td>'
            f'<td class="num">{r["perf_1m"]*100:+.1f}%</td>'
            f'<td class="num">{r["perf_3m"]*100:+.1f}%</td></tr>')
    return "\n".join(out)


def _rows_gaps(gaps: pd.DataFrame) -> str:
    if gaps.empty:
        return '<tr><td colspan="4" class="empty">Nothing gapped 3%+ this morning.</td></tr>'
    out = []
    for _, r in gaps.head(8).iterrows():
        cls = "up" if r["gap_pct"] > 0 else "down"
        stretched = ' data-hot="1"' if abs(r.get("gap_in_adr", 0)) >= 2 else ""
        gadr = f'{r["gap_in_adr"]:.1f}x' if np.isfinite(r.get("gap_in_adr", np.nan)) else "-"
        out.append(
            f'<tr{stretched}><td class="tick">{html.escape(r["ticker"])}</td>'
            f'<td class="num">${r["premarket"]:,.2f}</td>'
            f'<td class="num {cls}">{r["gap_pct"]*100:+.1f}%</td>'
            f'<td class="num">{gadr}</td></tr>')
    return "\n".join(out)


def _rows_watch(passing: list, close: pd.DataFrame) -> str:
    if not passing:
        return '<tr><td colspan="4" class="empty">No names pass the checklist.</td></tr>'
    out = []
    for c in passing[:8]:
        t = c["ticker"]
        spark = sparkline(close[t]) if t in close.columns else ""
        out.append(
            f'<tr><td class="tick">{html.escape(t)}</td>'
            f'<td class="sparkcell">{spark}</td>'
            f'<td class="num">${c["price"]:,.2f}</td>'
            f'<td class="num">{c["momentum"]*100:+.0f}%</td></tr>')
    return "\n".join(out)


def render(d: dict) -> str:
    reg, c, pos = d["regime"], d["candidate"], d["position"]
    green = reg["green"]

    if not green:
        verdict, state = "NO TRADE", "stop"
        because = "Market regime is red. This strategy is long-only and only tested in bullish conditions."
    elif c is None:
        verdict, state = "NO TRADE", "stop"
        because = "Regime is clear, but nothing passes the checklist today."
    else:
        verdict, state = "TRADE", "go"
        because = f"{c['ticker']} passes every filter and has no earnings inside the hold window."

    if c and pos:
        card = f"""
      <section class="card position">
        <div class="ticker-row">
          <span class="ticker">{html.escape(c['ticker'])}</span>
          <span class="last">${c['price']:,.2f}</span>
        </div>
        {candidate_chart(d['close'][c['ticker']], c['stop'], c['price'])}
        <div class="legend">
          <span><i class="k-price"></i>price</span>
          <span><i class="k-sma20"></i>20-day</span>
          <span><i class="k-sma50"></i>50-day</span>
          <span><i class="k-stop"></i>stop</span>
        </div>
        <div class="grid">
          <div><span class="lab">Buy</span><span class="val">{pos['shares']:.4f}<small> sh</small></span></div>
          <div><span class="lab">Position</span><span class="val">${pos['position_value']:,.0f}</span></div>
          <div><span class="lab">Stop</span><span class="val stop-c">${c['stop']:,.2f}</span></div>
          <div><span class="lab">Target</span><span class="val go-c">${pos['target']:,.2f}</span></div>
          <div><span class="lab">At risk</span><span class="val">${pos['risk_dollars']:,.2f}</span></div>
          <div><span class="lab">Earnings</span><span class="val">{c['earnings_days'] if c['earnings_days'] is not None else '?'}<small> d</small></span></div>
        </div>
      </section>"""
    else:
        card = ""

    blocked = ""
    if d["blocked"]:
        items = ", ".join(f"{t} ({n}d)" for t, n in d["blocked"])
        blocked = f'<p class="note">Skipped for earnings: {html.escape(items)}</p>'

    return f"""<title>Morning Desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {{
  --ground:#0F1216; --surface:#171B21; --surface-2:#1F252D; --line:#2A313B;
  --ink:#E6EAEF; --ink-soft:#98A2AE; --ink-faint:#6B7482;
  --accent:#C9A227; --go:#3FB27F; --stop:#E06C5A;
  --go-bg:#12271F; --stop-bg:#2A1815;
}}
:root[data-theme="light"] {{
  --ground:#F2F4F7; --surface:#FFFFFF; --surface-2:#E9EDF2; --line:#D3DAE3;
  --ink:#161A1F; --ink-soft:#525C68; --ink-faint:#7C8694;
  --accent:#8A6D0B; --go:#1F7A54; --stop:#B03F2C;
  --go-bg:#DDEDE5; --stop-bg:#F6E1DC;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);
  font-family:Archivo,system-ui,sans-serif;font-size:16px;line-height:1.5;
  -webkit-font-smoothing:antialiased}}
.wrap{{max-width:520px;margin:0 auto;padding:20px 16px 56px;
  display:flex;flex-direction:column;gap:18px}}
header{{display:flex;flex-direction:column;gap:4px}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent)}}
.asof{{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink-faint)}}
.verdict{{border-radius:3px;padding:22px 20px;display:flex;flex-direction:column;gap:8px;
  border:1px solid var(--line)}}
.verdict[data-state="go"]{{background:var(--go-bg);border-color:var(--go)}}
.verdict[data-state="stop"]{{background:var(--stop-bg);border-color:var(--stop)}}
.verdict h1{{margin:0;font-size:clamp(2.2rem,11vw,3rem);line-height:1;font-weight:700;
  letter-spacing:-.02em}}
.verdict[data-state="go"] h1{{color:var(--go)}}
.verdict[data-state="stop"] h1{{color:var(--stop)}}
.verdict p{{margin:0;font-size:14.5px;color:var(--ink-soft);max-width:44ch}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:3px;padding:16px}}
h2{{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent);margin:0 0 12px}}
.ticker-row{{display:flex;align-items:baseline;justify-content:space-between;
  padding-bottom:12px;margin-bottom:12px;border-bottom:1px solid var(--line)}}
.ticker{{font-size:1.7rem;font-weight:700;letter-spacing:-.01em}}
.last{{font-family:"IBM Plex Mono",monospace;font-size:1.15rem;color:var(--ink-soft);
  font-variant-numeric:tabular-nums}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px 10px}}
.grid>div{{display:flex;flex-direction:column;gap:2px}}
.lab{{font-family:"IBM Plex Mono",monospace;font-size:9.5px;letter-spacing:.11em;
  text-transform:uppercase;color:var(--ink-faint)}}
.val{{font-family:"IBM Plex Mono",monospace;font-size:1.02rem;font-weight:500;
  font-variant-numeric:tabular-nums}}
.val small{{font-size:.72em;color:var(--ink-faint)}}
.go-c{{color:var(--go)}} .stop-c{{color:var(--stop)}}
.chart{{display:block;width:100%;height:auto;margin:2px 0 10px;overflow:visible}}
.legend{{display:flex;flex-wrap:wrap;gap:4px 14px;margin-bottom:14px;
  font-family:"IBM Plex Mono",monospace;font-size:9.5px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-faint)}}
.legend span{{display:flex;align-items:center;gap:5px}}
.legend i{{width:12px;height:2px;display:inline-block;border-radius:1px}}
.k-price{{background:var(--ink)}} .k-sma20{{background:var(--accent)}}
.k-sma50{{background:var(--ink-faint)}}
.k-stop{{background:repeating-linear-gradient(90deg,var(--stop) 0 3px,transparent 3px 6px)}}
.spark{{display:block}}
.sparkcell{{width:92px;padding-right:4px}}
table{{width:100%;border-collapse:collapse;font-size:14.5px}}
th{{font-family:"IBM Plex Mono",monospace;font-size:9.5px;letter-spacing:.11em;
  text-transform:uppercase;color:var(--ink-faint);text-align:left;font-weight:500;
  padding-bottom:8px;border-bottom:1px solid var(--line)}}
th.num,td.num{{text-align:right;font-family:"IBM Plex Mono",monospace;
  font-variant-numeric:tabular-nums}}
td{{padding:9px 0;border-bottom:1px solid var(--line)}}
tr:last-child td{{border-bottom:none}}
.tick{{font-weight:600}}
tr.lead td:first-child{{color:var(--accent)}}
td.up{{color:var(--go)}} td.down{{color:var(--stop)}}
tr[data-hot="1"] td.num:last-child{{color:var(--accent);font-weight:600}}
.empty{{color:var(--ink-faint);font-style:italic;padding:14px 0}}
.note{{font-size:13px;color:var(--ink-faint);margin:12px 0 0}}
footer{{font-family:"IBM Plex Mono",monospace;font-size:11px;line-height:1.7;
  color:var(--ink-faint);border-top:1px solid var(--line);padding-top:16px}}
</style>

<div class="wrap">
  <header>
    <span class="eyebrow">Morning Desk &middot; {RISK*100:.0f}% risk per trade</span>
    <span class="asof">as of {d['stamp']}</span>
  </header>

  <section class="verdict" data-state="{state}">
    <h1>{verdict}</h1>
    <p>{html.escape(because)}</p>
  </section>
{card}
  <section class="card">
    <h2>Regime &middot; SPY {reg['spy']:,.2f}</h2>
    <table>
      <tr><td>10-day</td><td class="num">{reg['s10']:,.2f}</td></tr>
      <tr><td>20-day</td><td class="num">{reg['s20']:,.2f}</td></tr>
      <tr><td>50-day</td><td class="num">{reg['s50']:,.2f}</td></tr>
      <tr><td>Stacked / rising</td><td class="num">{reg['stacked']} / {reg['rising']}</td></tr>
    </table>
  </section>

  <section class="card">
    <h2>Pre-market gaps</h2>
    <table>
      <thead><tr><th>Ticker</th><th class="num">Pre</th><th class="num">Gap</th><th class="num">/ADR</th></tr></thead>
      <tbody>
{_rows_gaps(d['gaps'])}
      </tbody>
    </table>
    <p class="note">Gap divided by average daily range. Above 2x the move is already
    outsized &mdash; the regime where intraday fill rates fall to roughly 1 in 5.</p>
  </section>

  <section class="card">
    <h2>Passing the checklist</h2>
    <table>
      <thead><tr><th>Ticker</th><th>3 months</th><th class="num">Price</th><th class="num">Mom</th></tr></thead>
      <tbody>
{_rows_watch(d['passing'], d['close'])}
      </tbody>
    </table>
{blocked}
  </section>

  <section class="card">
    <h2>Sector strength</h2>
    <table>
      <thead><tr><th>Sector</th><th class="num">1mo</th><th class="num">3mo</th></tr></thead>
      <tbody>
{_rows_sectors(d['sectors'])}
      </tbody>
    </table>
  </section>

  <footer>
    Snapshot, not live &mdash; shows the last run on your PC, not current prices.<br>
    Output of fixed rules. Not a recommendation, and not evidence of an edge.
  </footer>
</div>
"""


if __name__ == "__main__":
    data = gather()
    OUT.write_text(render(data), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  regime: {'GREEN' if data['regime']['green'] else 'RED'}")
    print(f"  candidate: {data['candidate']['ticker'] if data['candidate'] else 'none'}")
    print(f"  gaps: {len(data['gaps'])}   passing: {len(data['passing'])}")
