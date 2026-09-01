"""Standalone dashboard builder for the cloud routine.

Single file, no local imports, installs its own dependencies. This is the version
embedded in the scheduled cloud agent's prompt, which runs in a sandbox with no
access to the rest of this project. Kept here so it can be tested locally before
being embedded, and so the two versions can be diffed later.

Logic mirrors bot.py + publish.py. If you change the rules there, change them here.
"""

import json, warnings
from datetime import datetime, timezone
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

ACCOUNT, RISK = 421.0, 0.01
HOLD_DAYS, MAX_FROM_HIGH, MIN_MOM, MAX_EXT, MAX_RISK, BUF, TARGET_R = 21, -0.15, 0.10, 0.10, 0.08, 0.02, 2.0

SECTOR_ETFS = {"XLK": "Technology", "XLF": "Financials", "XLV": "Health Care", "XLE": "Energy",
               "XLI": "Industrials", "XLY": "Cons Discretionary", "XLP": "Cons Staples",
               "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Communication"}
SECTORS = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "MRVL": "XLK", "AMD": "XLK",
    "INTC": "XLK", "MU": "XLK", "QCOM": "XLK", "TXN": "XLK", "ADBE": "XLK", "CRM": "XLK",
    "ORCL": "XLK", "CSCO": "XLK", "IBM": "XLK", "NOW": "XLK", "PANW": "XLK", "ARM": "XLK",
    "DELL": "XLK", "PLTR": "XLK", "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY",
    "NKE": "XLY", "SBUX": "XLY", "GM": "XLY", "F": "XLY", "UBER": "XLI", "BA": "XLI",
    "CAT": "XLI", "DE": "XLI", "GE": "XLI", "LMT": "XLI", "RTX": "XLI", "HON": "XLI",
    "UPS": "XLI", "FDX": "XLI", "RKLB": "XLI", "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC",
    "DIS": "XLC", "T": "XLC", "VZ": "XLC", "CMCSA": "XLC", "JPM": "XLF", "BAC": "XLF",
    "GS": "XLF", "MS": "XLF", "WFC": "XLF", "V": "XLF", "MA": "XLF", "AXP": "XLF",
    "SCHW": "XLF", "BLK": "XLF", "COIN": "XLF", "SOFI": "XLF", "JNJ": "XLV", "PFE": "XLV",
    "MRK": "XLV", "LLY": "XLV", "ABBV": "XLV", "UNH": "XLV", "TMO": "XLV", "ABT": "XLV",
    "BMY": "XLV", "GILD": "XLV", "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE",
    "OXY": "XLE", "IREN": "XLE", "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP", "COST": "XLP",
}
GAPPERS = ["IREN", "ASTS", "RKLB", "LUNR", "ACHR", "IONQ", "QBTS", "SMCI", "SOFI", "COIN",
           "MARA", "RIOT", "CLSK", "WULF", "PLTR", "SOUN", "TSLA", "NVDA", "AMD", "MU",
           "GME", "AMC", "PLUG", "UPST", "AFRM", "HOOD", "DKNG", "RBLX", "SNAP", "KEEL"]


def dl(tickers, start="2023-01-01"):
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    c = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    return c.dropna(how="all")


def evaluate(close, t):
    s = close[t].dropna()
    if len(s) < 252:
        return None
    price = float(s.iloc[-1])
    s20, s50 = (float(s.rolling(n).mean().iloc[-1]) for n in (20, 50))
    hi52 = float(s.rolling(252).max().iloc[-1])
    stop = float(s.iloc[-20:].min()) * (1 - BUF)
    risk = (price - stop) / price if price > stop else np.nan
    mom = float(s.iloc[-22] / s.iloc[-252] - 1)
    ok = (price > s20 and s20 > s50 and price / hi52 - 1 >= MAX_FROM_HIGH and mom >= MIN_MOM
          and price / s20 - 1 <= MAX_EXT and np.isfinite(risk) and 0 < risk <= MAX_RISK)
    return {"ticker": t, "price": price, "stop": stop, "risk": risk, "mom": mom,
            "from_high": price / hi52 - 1, "passes": ok}


def gap_scan(tickers):
    out = []
    for t in tickers:
        try:
            df = yf.download(t, period="2d", interval="5m", prepost=True,
                             progress=False, auto_adjust=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = df.index.tz_convert("America/New_York")
            today = df.index[-1].date()
            prior = df[(df.index.date < today) & (df.index.time >= pd.Timestamp("09:30").time())
                       & (df.index.time <= pd.Timestamp("16:00").time())]
            pre = df[(df.index.date == today) & (df.index.time >= pd.Timestamp("04:00").time())
                     & (df.index.time < pd.Timestamp("09:30").time())]
            if prior.empty or pre.empty:
                continue
            pc, last = float(prior["Close"].iloc[-1]), float(pre["Close"].iloc[-1])
            d = yf.download(t, period="3mo", interval="1d", progress=False, auto_adjust=True)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            adr = float((d["High"] / d["Low"] - 1).tail(20).mean()) if not d.empty else np.nan
            gap = last / pc - 1
            if abs(gap) >= 0.03:
                out.append({"ticker": t, "pre": last, "gap": gap,
                            "gadr": gap / adr if adr and adr > 0 else np.nan})
        except Exception:
            continue
    return sorted(out, key=lambda r: -abs(r["gap"]))


def spark(series, w=88, h=26):
    v = series.dropna().to_numpy()[-63:]
    if len(v) < 5:
        return ""
    lo, hi = float(v.min()), float(v.max())
    rng = hi - lo if hi - lo > 1e-9 else 1
    xs = np.linspace(1, w - 1, len(v))
    ys = [h - 2 - (x - lo) / rng * (h - 4) for x in v]
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    col = "var(--go)" if v[-1] >= v[0] else "var(--stop)"
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" aria-hidden="true">'
            f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.4"/>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="2" fill="{col}"/></svg>')


def chart(series, stop, entry, w=460, h=170):
    s = series.dropna().iloc[-126:]
    if len(s) < 30:
        return ""
    p = s.to_numpy()
    lo = min(float(np.nanmin(p)), stop) * 0.985
    hi = max(float(np.nanmax(p)), entry) * 1.015
    rng = hi - lo if hi - lo > 1e-9 else 1
    inner, pad = h - 16, 8
    xs = np.linspace(2, w - 2, len(p))

    def ln(vals, col, wd):
        ok = [(x, v) for x, v in zip(xs, vals) if np.isfinite(v)]
        if len(ok) < 2:
            return ""
        pts = " ".join(f"{x:.1f},{pad + inner - (v - lo) / rng * inner:.1f}" for x, v in ok)
        return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{wd}"/>'

    y = pad + inner - (stop - lo) / rng * inner
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" '
            f'aria-label="Six-month price with moving averages and the stop level">'
            f'<line x1="2" x2="{w-2}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--stop)" '
            f'stroke-width="1" stroke-dasharray="4 3"/>'
            + ln(s.rolling(50).mean().to_numpy(), "var(--ink-faint)", 1.1)
            + ln(s.rolling(20).mean().to_numpy(), "var(--accent)", 1.2)
            + ln(p, "var(--ink)", 1.7) + "</svg>")


def build():
    close = dl(sorted(set(list(SECTORS) + list(SECTOR_ETFS) + ["SPY"])))
    spy = close["SPY"].dropna()
    a, b, c50 = (spy.rolling(n).mean() for n in (10, 20, 50))
    reg = {"green": bool(a.iloc[-1] > b.iloc[-1] > c50.iloc[-1] and a.iloc[-1] > a.iloc[-6]),
           "spy": float(spy.iloc[-1]), "s10": float(a.iloc[-1]),
           "s20": float(b.iloc[-1]), "s50": float(c50.iloc[-1])}

    secs = []
    for etf, nm in SECTOR_ETFS.items():
        if etf in close.columns:
            s = close[etf].dropna()
            if len(s) > 70:
                secs.append({"sector": nm, "etf": etf, "m1": s.iloc[-1] / s.iloc[-22] - 1,
                             "m3": s.iloc[-1] / s.iloc[-64] - 1})
    secs.sort(key=lambda r: -(r["m1"] + r["m3"]) / 2)
    leaders = {r["etf"] for r in secs[:4]}

    rows = [evaluate(close, t) for t in SECTORS if t in close.columns]
    rows = [r for r in rows if r]
    for r in rows:
        r["lead"] = SECTORS[r["ticker"]] in leaders
    passing = sorted([r for r in rows if r["passes"]], key=lambda r: (-r["lead"], -r["mom"]))

    chosen, blocked = None, []
    for r in passing[:8]:
        try:
            ed = yf.Ticker(r["ticker"]).get_earnings_dates(limit=8)
            today = pd.Timestamp.utcnow().tz_localize(None).normalize()
            fut = [pd.Timestamp(x).tz_localize(None) for x in ed.index]
            fut = [x for x in fut if x >= today]
            days = (min(fut) - today).days if fut else None
        except Exception:
            days = None
        r["edays"] = days
        if days is not None and 0 <= days <= HOLD_DAYS:
            blocked.append(f"{r['ticker']} ({days}d)")
            continue
        chosen = r
        break

    return {"close": close, "reg": reg, "secs": secs, "passing": passing,
            "chosen": chosen, "blocked": blocked, "gaps": gap_scan(GAPPERS),
            "stamp": datetime.now(timezone.utc).strftime("%a %d %b %Y, %H:%M UTC")}


CSS = """:root{--ground:#0F1216;--surface:#171B21;--line:#2A313B;--ink:#E6EAEF;--ink-soft:#98A2AE;
--ink-faint:#6B7482;--accent:#C9A227;--go:#3FB27F;--stop:#E06C5A;--go-bg:#12271F;--stop-bg:#2A1815}
:root[data-theme="light"]{--ground:#F2F4F7;--surface:#FFFFFF;--line:#D3DAE3;--ink:#161A1F;
--ink-soft:#525C68;--ink-faint:#7C8694;--accent:#8A6D0B;--go:#1F7A54;--stop:#B03F2C;
--go-bg:#DDEDE5;--stop-bg:#F6E1DC}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:Archivo,system-ui,sans-serif;
font-size:16px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:520px;margin:0 auto;padding:20px 16px 56px;display:flex;flex-direction:column;gap:18px}
header{display:flex;flex-direction:column;gap:4px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.16em;
text-transform:uppercase;color:var(--accent)}
.asof{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--ink-faint)}
.verdict{border-radius:3px;padding:22px 20px;display:flex;flex-direction:column;gap:8px;
border:1px solid var(--line)}
.verdict[data-state="go"]{background:var(--go-bg);border-color:var(--go)}
.verdict[data-state="stop"]{background:var(--stop-bg);border-color:var(--stop)}
.verdict h1{margin:0;font-size:clamp(2.2rem,11vw,3rem);line-height:1;font-weight:700;letter-spacing:-.02em}
.verdict[data-state="go"] h1{color:var(--go)}
.verdict[data-state="stop"] h1{color:var(--stop)}
.verdict p{margin:0;font-size:14.5px;color:var(--ink-soft);max-width:44ch}
.card{background:var(--surface);border:1px solid var(--line);border-radius:3px;padding:16px}
h2{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;
color:var(--accent);margin:0 0 12px}
.ticker-row{display:flex;align-items:baseline;justify-content:space-between;padding-bottom:12px;
margin-bottom:12px;border-bottom:1px solid var(--line)}
.ticker{font-size:1.7rem;font-weight:700}
.last{font-family:"IBM Plex Mono",monospace;font-size:1.15rem;color:var(--ink-soft);
font-variant-numeric:tabular-nums}
.chart{display:block;width:100%;height:auto;margin:2px 0 10px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin-bottom:14px;font-family:"IBM Plex Mono",monospace;
font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-faint)}
.legend span{display:flex;align-items:center;gap:5px}
.legend i{width:12px;height:2px;display:inline-block;border-radius:1px}
.k-price{background:var(--ink)}.k-sma20{background:var(--accent)}.k-sma50{background:var(--ink-faint)}
.k-stop{background:repeating-linear-gradient(90deg,var(--stop) 0 3px,transparent 3px 6px)}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px 10px}
.grid>div{display:flex;flex-direction:column;gap:2px}
.lab{font-family:"IBM Plex Mono",monospace;font-size:9.5px;letter-spacing:.11em;text-transform:uppercase;
color:var(--ink-faint)}
.val{font-family:"IBM Plex Mono",monospace;font-size:1.02rem;font-weight:500;font-variant-numeric:tabular-nums}
.val small{font-size:.72em;color:var(--ink-faint)}
.go-c{color:var(--go)}.stop-c{color:var(--stop)}
table{width:100%;border-collapse:collapse;font-size:14.5px}
th{font-family:"IBM Plex Mono",monospace;font-size:9.5px;letter-spacing:.11em;text-transform:uppercase;
color:var(--ink-faint);text-align:left;font-weight:500;padding-bottom:8px;border-bottom:1px solid var(--line)}
th.num,td.num{text-align:right;font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
td{padding:9px 0;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:none}
.tick{font-weight:600}
tr.lead td:first-child{color:var(--accent)}
td.up{color:var(--go)}td.down{color:var(--stop)}
tr[data-hot="1"] td.num:last-child{color:var(--accent);font-weight:600}
.spark{display:block}.sparkcell{width:92px;padding-right:4px}
.empty{color:var(--ink-faint);font-style:italic;padding:14px 0}
.note{font-size:13px;color:var(--ink-faint);margin:12px 0 0}
footer{font-family:"IBM Plex Mono",monospace;font-size:11px;line-height:1.7;color:var(--ink-faint);
border-top:1px solid var(--line);padding-top:16px}"""


def render(d):
    reg, c = d["reg"], d["chosen"]
    if not reg["green"]:
        verdict, state = "NO TRADE", "stop"
        why = "Market regime is red. This strategy is long-only and only tested in bullish conditions."
    elif c is None:
        verdict, state = "NO TRADE", "stop"
        why = "Regime is clear, but nothing passes the checklist today."
    else:
        verdict, state = "TRADE", "go"
        why = f"{c['ticker']} passes every filter and has no earnings inside the hold window."

    card = ""
    if c:
        rps = c["price"] - c["stop"]
        sh = (ACCOUNT * RISK) / rps if rps > 0 else 0
        card = f"""<section class="card">
<div class="ticker-row"><span class="ticker">{c['ticker']}</span>
<span class="last">${c['price']:,.2f}</span></div>
{chart(d['close'][c['ticker']], c['stop'], c['price'])}
<div class="legend"><span><i class="k-price"></i>price</span><span><i class="k-sma20"></i>20-day</span>
<span><i class="k-sma50"></i>50-day</span><span><i class="k-stop"></i>stop</span></div>
<div class="grid">
<div><span class="lab">Buy</span><span class="val">{sh:.4f}<small> sh</small></span></div>
<div><span class="lab">Position</span><span class="val">${sh*c['price']:,.0f}</span></div>
<div><span class="lab">Stop</span><span class="val stop-c">${c['stop']:,.2f}</span></div>
<div><span class="lab">Target</span><span class="val go-c">${c['price']+TARGET_R*rps:,.2f}</span></div>
<div><span class="lab">At risk</span><span class="val">${ACCOUNT*RISK:,.2f}</span></div>
<div><span class="lab">Earnings</span><span class="val">{c['edays'] if c.get('edays') is not None else '?'}<small> d</small></span></div>
</div></section>"""

    gaps = "".join(
        f'<tr{" data-hot=\"1\"" if np.isfinite(g["gadr"]) and abs(g["gadr"])>=2 else ""}>'
        f'<td class="tick">{g["ticker"]}</td><td class="num">${g["pre"]:,.2f}</td>'
        f'<td class="num {"up" if g["gap"]>0 else "down"}">{g["gap"]*100:+.1f}%</td>'
        f'<td class="num">{f"{g["gadr"]:.1f}x" if np.isfinite(g["gadr"]) else "-"}</td></tr>'
        for g in d["gaps"][:8]) or '<tr><td colspan="4" class="empty">Nothing gapped 3%+ this morning.</td></tr>'

    watch = "".join(
        f'<tr><td class="tick">{r["ticker"]}</td>'
        f'<td class="sparkcell">{spark(d["close"][r["ticker"]])}</td>'
        f'<td class="num">${r["price"]:,.2f}</td>'
        f'<td class="num">{r["mom"]*100:+.0f}%</td></tr>'
        for r in d["passing"][:8]) or '<tr><td colspan="4" class="empty">No names pass the checklist.</td></tr>'

    sectors = "".join(
        f'<tr{" class=\"lead\"" if i<4 else ""}><td>{s["sector"]}</td>'
        f'<td class="num">{s["m1"]*100:+.1f}%</td><td class="num">{s["m3"]*100:+.1f}%</td></tr>'
        for i, s in enumerate(d["secs"][:6]))

    blocked = (f'<p class="note">Skipped for earnings: {", ".join(d["blocked"])}</p>'
               if d["blocked"] else "")

    return f"""<title>Morning Desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header><span class="eyebrow">Morning Desk &middot; ${ACCOUNT:,.0f} &middot; {RISK*100:.0f}% risk</span>
<span class="asof">as of {d['stamp']}</span></header>
<section class="verdict" data-state="{state}"><h1>{verdict}</h1><p>{why}</p></section>
{card}
<section class="card"><h2>Regime &middot; SPY {reg['spy']:,.2f}</h2><table>
<tr><td>10-day</td><td class="num">{reg['s10']:,.2f}</td></tr>
<tr><td>20-day</td><td class="num">{reg['s20']:,.2f}</td></tr>
<tr><td>50-day</td><td class="num">{reg['s50']:,.2f}</td></tr></table></section>
<section class="card"><h2>Pre-market gaps</h2><table>
<thead><tr><th>Ticker</th><th class="num">Pre</th><th class="num">Gap</th><th class="num">/ADR</th></tr></thead>
<tbody>{gaps}</tbody></table>
<p class="note">Gap divided by average daily range. Above 2x the move is already outsized &mdash;
the regime where intraday fill rates fall to roughly 1 in 5.</p></section>
<section class="card"><h2>Passing the checklist</h2><table>
<thead><tr><th>Ticker</th><th>3 months</th><th class="num">Price</th><th class="num">Mom</th></tr></thead>
<tbody>{watch}</tbody></table>{blocked}</section>
<section class="card"><h2>Sector strength</h2><table>
<thead><tr><th>Sector</th><th class="num">1mo</th><th class="num">3mo</th></tr></thead>
<tbody>{sectors}</tbody></table></section>
<footer>Snapshot, not live &mdash; built when the routine last ran.<br>
Output of fixed rules. Not a recommendation, and not evidence of an edge.</footer>
</div>"""


if __name__ == "__main__":
    d = build()
    open("dashboard.html", "w", encoding="utf-8").write(render(d))
    print(json.dumps({"regime": "GREEN" if d["reg"]["green"] else "RED",
                      "candidate": d["chosen"]["ticker"] if d["chosen"] else None,
                      "passing": len(d["passing"]), "gaps": len(d["gaps"])}))
