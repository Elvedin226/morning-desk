"""Dashboard builder and paper-account driver, run by GitHub Actions each weekday.

This began as a single file with no local imports, because it was meant to be
embedded in a cloud agent's prompt and run in a sandbox with no repo. That is no
longer how it runs: GitHub Actions checks out the whole repo, so it imports
portfolio.py and risk.py directly. Duplicating the account and sizing maths here
would guarantee the two copies drift, and a paper account that disagrees with
itself is worse than none.

The strategy rules ARE still duplicated from bot.py (evaluate / regime / sector
strength). If you change the rules there, change them here.

The workflow commits data_cache/portfolio.json, so the account persists between
runs and the record accumulates in git history.
"""

import json, os, warnings
from datetime import datetime, timezone
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

import portfolio
import risk

ACCOUNT, RISK = 421.0, 0.01

# FORCED MODE - on for the testing period, by request.
#
# When the checklist approves nothing, take the best available name anyway:
# long the strongest uptrend, short the weakest downtrend. The point is to
# generate a readable record during a two-week test instead of fourteen blank
# days, and to measure what the filters are actually worth.
#
# Every such position is tagged forced=True and its stats are reported
# SEPARATELY. That separation is the whole reason this is defensible: without
# it, forced trades would contaminate the record of the strategy they exist to
# be compared against.
FORCE_DAILY = True
FORCED_RISK = 0.005   # half normal size - these are trades the rules rejected
# Forced positions get their own slot budget. risk.MAX_POSITIONS is 3, which is
# right for real capital but would fill in three days and then stop, giving you
# three trades rather than a trade a day. These are half-size and simulated, so
# a wider budget is affordable; the daily-loss and drawdown switches still bind.
MAX_FORCED = 10
FORCED_HOLD_DAYS = 10  # cycle faster than the 40-day default so slots free up

# Connors RSI-2. See rsi2_candidates() for what the testing actually showed.
RSI2_ENABLED = True
RSI2_ENTRY = 5.0      # RSI(2) below this, inside a 200-day uptrend
RSI2_STOP = 0.08      # hard stop the original strategy lacks

# Stated goal: $421 -> $1,000 by 31 Dec 2026. Tracked on the dashboard so the
# gap between the goal and the trajectory stays visible rather than assumed.
# target_test.py measures P(hit) for the current engine at essentially 0% - the
# edge is real (+0.287%/trade) but far too small over ~85 sessions. Reaching it
# requires convex bets whose measured expected value is negative.
GOAL_EQUITY = 1000.0
GOAL_DATE = "2026-12-31"
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


def build(intraday=False):
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
            "chosen": chosen, "blocked": blocked,
            "intraday": intraday,
            # The gap scan compares pre-market to the prior close; at midday it
            # measures nothing. Skipping it also removes ~60 requests per run,
            # which is what makes a half-hourly schedule affordable.
            "gaps": [] if intraday else gap_scan(GAPPERS),
            "stamp": datetime.now(timezone.utc).strftime("%a %d %b %Y, %H:%M UTC")}


def rsi2(series, n=2):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def rsi2_candidates(d, held):
    """Connors RSI-2: a sharp pullback inside a long-term uptrend.

    Added because it is the only strategy tested in this project that cleared
    its own shuffled-bars null (p=0.05 on SPY 2011-now, real OOS Sharpe 0.64
    against a shuffled 95th percentile of 0.64 - marginal, but it cleared).
    Across 90 names it fires about 110 times a year, which is roughly one signal
    every other day; the 21-day swing checklist fires far less often.

    Measured over 2011-2026 from $421: +3.76% a year at 20% per position. That
    is real but it LOSES to buying and holding SPY (+14.19%) over the same
    window. It is here because it trades often enough to be tested and observed,
    not because it beats the index.

    Exit is Connors' own rule - close above the 5-day average - plus a hard 8%
    stop the original does not have. Without a stop, a mean-reversion entry has
    no defined loss, and the 2011-2026 run showed avg loss ($5.19) already
    exceeding avg win ($3.35).
    """
    close = d["close"]
    out = []
    for t in SECTORS:
        if t not in close.columns or t in held:
            continue
        srs = close[t].dropna()
        if len(srs) < 220:
            continue
        r = float(rsi2(srs).iloc[-1])
        sma200 = float(srs.rolling(200).mean().iloc[-1])
        price = float(srs.iloc[-1])
        if not (np.isfinite(r) and price > sma200 and r < RSI2_ENTRY):
            continue
        stop = price * (1 - RSI2_STOP)
        out.append({"ticker": t, "price": price, "stop": stop,
                    "target": price + TARGET_R * (price - stop),
                    "side": "long", "rsi2": r,
                    "exit_ma": float(srs.rolling(5).mean().iloc[-1])})
    return sorted(out, key=lambda x: x["rsi2"])


def fallback(d, held):
    """Best available trade when the checklist approves nothing.

    Long the strongest name trading above its 20- and 50-day; short the weakest
    trading below both. Whichever side has the more extreme momentum wins, so
    the direction is chosen by the tape rather than fixed in advance.

    Stops use the same 20-day extreme + buffer construction as the real rules,
    mirrored for shorts, so a forced trade is risk-managed the same way even
    though it was never approved.
    """
    close = d["close"]
    best = None
    for t in SECTORS:
        if t not in close.columns or t in held:
            continue
        srs = close[t].dropna()
        if len(srs) < 252:
            continue
        price = float(srs.iloc[-1])
        s20, s50 = (float(srs.rolling(n).mean().iloc[-1]) for n in (20, 50))
        mom = float(srs.iloc[-22] / srs.iloc[-252] - 1)
        if price > s20 > s50:
            stop = float(srs.iloc[-20:].min()) * (1 - BUF)
            side, score = "long", mom
        elif price < s20 < s50:
            stop = float(srs.iloc[-20:].max()) * (1 + BUF)
            side, score = "short", -mom
        else:
            continue
        rps = abs(price - stop)
        if rps <= 0 or rps / price > MAX_RISK:
            continue
        target = price + TARGET_R * rps if side == "long" else price - TARGET_R * rps
        if target <= 0:
            continue
        if best is None or score > best["score"]:
            best = {"ticker": t, "price": price, "stop": stop, "target": target,
                    "side": side, "mom": mom, "score": score}
    return best


def trade(d, intraday=False):
    """Advance the paper account one day: resolve open positions, then open the
    day's candidate if the risk layer allows it.

    Exits are settled BEFORE the entry is sized, so freed cash is available and a
    slot released by a stop can be reused the same day.
    """
    book = portfolio.load()
    book, closed = portfolio.update(book, intraday=intraday)

    equity = book["equity"]
    held = [p["ticker"] for p in book["positions"]]
    # Computed AFTER exits settle, so a target hit this run counts toward the
    # day's number and can stop further entries immediately.
    day_pnl = portfolio.day_realised(book)
    c, skip = d["chosen"], None

    # Regime is checked HERE and not left to build(), which fills in `chosen`
    # whatever the regime so the dashboard can show what would have qualified.
    # Reading that field as permission to trade put a position on during a red
    # tape - the one condition this long-only strategy was never tested in.
    forced_note = None
    if not d["reg"]["green"]:
        skip = "regime red - strategy is long-only and untested in this tape"
        forced_note = "regime red"
    elif c is None:
        skip = "nothing passed the checklist"
        forced_note = "no checklist pass"
    else:
        curve = book.get("equity_curve", [])
        day_start = curve[-2]["equity"] if len(curve) >= 2 else equity
        g = risk.gate(equity, book["start_equity"], day_start, len(held), True,
                      day_pnl=day_pnl)
        if not g.allowed:
            skip = g.reason
        elif c["ticker"] in held:
            skip = f"already holding {c['ticker']}"
        else:
            corr = risk.correlation_veto(c["ticker"], held, d["close"])
            if not corr.allowed:
                skip = corr.reason
            else:
                sized = risk.size(equity, c["price"], c["stop"])
                if not sized.allowed:
                    skip = sized.reason
                else:
                    target = c["price"] + TARGET_R * (c["price"] - c["stop"])
                    portfolio.open_position(book, c["ticker"], sized.qty,
                                            c["price"], c["stop"], target)

    # CONNORS RSI-2. Runs before the forced fallback so a tested signal always
    # outranks a manufactured one. Sized at normal risk, not forced risk, and
    # tagged forced=False because it IS a rule - just a different rule from the
    # 21-day swing checklist.
    d["rsi2"] = None
    if RSI2_ENABLED:
        cands = rsi2_candidates(d, held)
        d["rsi2_seen"] = [f"{c['ticker']} ({c['rsi2']:.1f})" for c in cands[:5]]
        g = risk.gate(equity, book["start_equity"],
                      (book.get("equity_curve") or [{}])[-2].get("equity", equity)
                      if len(book.get("equity_curve", [])) >= 2 else equity,
                      len(held), True, day_pnl=day_pnl)
        if not g.allowed:
            d["rsi2"] = f"no RSI-2 entry: {g.reason}"
        elif not cands:
            d["rsi2"] = "no RSI-2 signal today"
        else:
            for c2 in cands:
                corr = risk.correlation_veto(c2["ticker"], held, d["close"])
                if not corr.allowed:
                    continue
                sized = risk.size(equity, c2["price"], c2["stop"])
                if not sized.allowed:
                    continue
                portfolio.open_position(book, c2["ticker"], sized.qty, c2["price"],
                                        c2["stop"], c2["target"], side="long",
                                        forced=False, note=f"RSI-2 {c2['rsi2']:.1f}")
                held.append(c2["ticker"])
                d["rsi2"] = (f"RSI-2 LONG {c2['ticker']} @ ${c2['price']:,.2f} "
                             f"(RSI {c2['rsi2']:.1f})")
                skip = None
                break
            else:
                d["rsi2"] = "RSI-2 signals all vetoed (correlation or size)"

    # FORCED FALLBACK. Only when the rules declined and no slot conflict exists.
    # Deliberately does NOT override the risk gate: position limits, the daily
    # loss halt and the drawdown kill switch still apply. Forcing a trade is a
    # research choice; disabling the thing that bounds losses is not, and the
    # two are easy to conflate.
    d["forced"] = None
    if FORCE_DAILY and skip:
        curve = book.get("equity_curve", [])
        day_start = curve[-2]["equity"] if len(curve) >= 2 else equity
        n_forced = sum(1 for p in book["positions"] if p.get("forced"))
        # Pass the FORCED count against the forced budget. The loss and drawdown
        # switches inside gate() are what actually bound risk and still apply.
        g = risk.gate(equity, book["start_equity"], day_start, n_forced, True,
                      max_positions=MAX_FORCED, day_pnl=day_pnl)
        if not g.allowed:
            d["forced"] = f"not forced: {g.reason}"
        else:
            fb = fallback(d, held)
            if fb is None:
                d["forced"] = "not forced: nothing tradable in either direction"
            else:
                sized = risk.size(equity, fb["price"], fb["stop"],
                                  risk_pct=FORCED_RISK, side=fb["side"])
                if not sized.allowed:
                    d["forced"] = f"not forced: {sized.reason}"
                else:
                    portfolio.open_position(
                        book, fb["ticker"], sized.qty, fb["price"], fb["stop"],
                        fb["target"], side=fb["side"], forced=True,
                        note=forced_note or skip)
                    d["forced"] = (f"forced {fb['side'].upper()} {fb['ticker']} "
                                   f"@ ${fb['price']:,.2f} ({forced_note})")

    portfolio.save(book)
    d["book"], d["closed"], d["skip"] = book, closed, skip
    d["stats"] = portfolio.stats(book)
    return d


def equity_spark(curve, w=220, h=44):
    """The account's own line. Flat until there are trades - which is itself the
    honest picture, and better than hiding the panel until it looks good."""
    if len(curve) < 2:
        return ""
    v = np.array([p["equity"] for p in curve], dtype=float)
    lo, hi = float(v.min()), float(v.max())
    rng = hi - lo if hi - lo > 1e-9 else max(abs(hi), 1.0) * 0.02
    xs = np.linspace(1, w - 1, len(v))
    ys = [h - 4 - (x - lo) / rng * (h - 10) for x in v]
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    col = "var(--go)" if v[-1] >= v[0] else "var(--stop)"
    base = f"{xs[0]:.1f},{h} " + pts + f" {xs[-1]:.1f},{h}"
    return (f'<svg class="eqline" viewBox="0 0 {w} {h}" width="100%" height="{h}" '
            f'preserveAspectRatio="none" role="img" aria-label="Account equity over time">'
            f'<polygon points="{base}" fill="{col}" opacity="0.10"/>'
            f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6"/>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="2.4" fill="{col}"/></svg>')


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

# Account styles kept separate so publish.py can interpolate them into its own
# stylesheet instead of holding a second copy that drifts.
ACCOUNT_CSS = """.acct-top{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:2px}
.equity{font-family:"IBM Plex Mono",monospace;font-size:2.1rem;font-weight:600;letter-spacing:-.02em;
font-variant-numeric:tabular-nums;line-height:1}
.delta{font-family:"IBM Plex Mono",monospace;font-size:1rem;font-variant-numeric:tabular-nums}
.eqline{display:block;width:100%;height:auto;margin:10px 0 14px}
.acct-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px 8px;
padding-top:14px;border-top:1px solid var(--line)}
.acct-grid>div{display:flex;flex-direction:column;gap:2px}
.pos-bar{height:4px;border-radius:2px;background:var(--line);position:relative;overflow:hidden;
margin-top:5px}
.pos-bar i{position:absolute;top:0;bottom:0;left:0;border-radius:2px;background:var(--accent)}
.posname{display:flex;align-items:baseline;gap:7px}
.posmeta{font-family:"IBM Plex Mono",monospace;font-size:10px;color:var(--ink-faint);
letter-spacing:.04em}
.why{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--ink-faint);
margin:12px 0 0;padding-top:10px;border-top:1px solid var(--line)}
.tag{font-family:"IBM Plex Mono",monospace;font-size:9px;letter-spacing:.1em;text-transform:uppercase;
padding:2px 5px;border-radius:2px;border:1px solid var(--line);color:var(--ink-faint)}
.tag[data-r="target"]{color:var(--go);border-color:var(--go)}
.tag[data-r="stop"]{color:var(--stop);border-color:var(--stop)}
.tag[data-s="short"]{color:var(--stop);border-color:var(--stop)}
.tag[data-s="long"]{color:var(--go);border-color:var(--go)}
.tag[data-s="forced"]{color:var(--accent);border-color:var(--accent)}
.target{margin-top:14px;padding-top:14px;border-top:1px solid var(--line)}
.target-top{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:2px}
.arms td:first-child{font-family:"IBM Plex Mono",monospace;font-size:11px;
letter-spacing:.06em;text-transform:uppercase;color:var(--ink-soft)}"""

CSS = CSS + ACCOUNT_CSS


def account_panel(d):
    """The paper account: what it holds, what it closed, what that came to."""
    book, s = d["book"], d["stats"]
    sign = "go-c" if s["total_return"] >= 0 else "stop-c"

    rows = ""
    for p in book["positions"]:
        last = p.get("last", p["entry"])
        pnl = p.get("unrealised", 0.0)
        # How far price has travelled from stop to target, as a bar. Clamped so
        # a gap outside the band cannot render a bar wider than its track.
        span = p["target"] - p["stop"]
        frac = min(max((last - p["stop"]) / span, 0.0), 1.0) if span > 0 else 0.0
        side = p.get("side", "long")
        badge = f'<span class="tag" data-s="{side}">{side}</span>'
        if p.get("forced"):
            badge += '<span class="tag" data-s="forced">forced</span>'
        rows += (f'<tr><td><div class="posname"><span class="tick">{p["ticker"]}</span>'
                 f'{badge}<span class="posmeta">{p["qty"]:.3f} sh</span></div>'
                 f'<div class="posmeta">${p["stop"]:,.2f} &rarr; ${p["target"]:,.2f}</div>'
                 f'<div class="pos-bar"><i style="width:{frac*100:.0f}%"></i></div></td>'
                 f'<td class="num">${p["entry"]:,.2f}</td>'
                 f'<td class="num">${last:,.2f}</td>'
                 f'<td class="num {"up" if pnl >= 0 else "down"}">${pnl:+,.2f}</td></tr>')
    if not rows:
        rows = '<tr><td colspan="4" class="empty">No open positions.</td></tr>'

    done = ""
    for t in reversed(book.get("closed", [])[-8:]):
        mark = '<span class="tag" data-s="forced">forced</span>' if t.get("forced") else ""
        sd = t.get("side", "long")
        done += (f'<tr><td><div class="posname"><span class="tick">{t["ticker"]}</span>'
                 f'<span class="tag" data-s="{sd}">{sd}</span>{mark}'
                 f'<span class="tag" data-r="{t["reason"].split()[0]}">{t["reason"]}</span></div>'
                 f'<div class="posmeta">{t["opened"]} &rarr; {t["closed"]}</div></td>'
                 f'<td class="num">${t["entry"]:,.2f}</td><td class="num">${t["exit"]:,.2f}</td>'
                 f'<td class="num {"up" if t["pnl"] >= 0 else "down"}">${t["pnl"]:+,.2f}<br>'
                 f'<small>{t["pnl_pct"]:+.1f}%</small></td></tr>')
    closed_tbl = ""
    if done:
        closed_tbl = f"""<section class="card"><h2>Closed trades</h2><table>
<thead><tr><th>Trade</th><th class="num">In</th><th class="num">Out</th><th class="num">P&amp;L</th></tr></thead>
<tbody>{done}</tbody></table></section>"""

    wr = f"{s['win_rate']*100:.0f}%" if s["win_rate"] is not None else "&mdash;"

    a = s["arms"]
    def arm_row(label, x):
        w = f"{x['win_rate']*100:.0f}%" if x["win_rate"] is not None else "&mdash;"
        avg = f"{x['avg_pct']:+.1f}%" if x["avg_pct"] is not None else "&mdash;"
        return (f'<tr><td>{label}</td><td class="num">{x["n"]}<small> +{x["open"]} open</small></td>'
                f'<td class="num">{x["longs"]}L / {x["shorts"]}S</td><td class="num">{w}</td>'
                f'<td class="num">{avg}</td>'
                f'<td class="num {"up" if x["pnl"] >= 0 else "down"}">${x["pnl"]:+,.2f}</td></tr>')
    arms_tbl = f"""<section class="card"><h2>Rules vs forced</h2><table class="arms">
<thead><tr><th>Arm</th><th class="num">Closed</th><th class="num">Dir</th>
<th class="num">Win</th><th class="num">Avg</th><th class="num">P&amp;L</th></tr></thead>
<tbody>{arm_row("By the rules", a["qualified"])}{arm_row("Forced", a["forced"])}</tbody></table>
<p class="note">Forced trades are taken at half size on days the checklist declined,
to produce a record during testing. They are scored separately because the point
is to compare them against the rules &mdash; pooling them would answer neither
question.</p></section>"""

    lines = []
    if d.get("rsi2"):
        lines.append(d["rsi2"])
    if d.get("skip"):
        lines.append(f"Swing rules: {d['skip']}")
    if d.get("forced"):
        lines.append(d["forced"])
    why = f'<p class="why">{"<br>".join(lines)}</p>' if lines else ""

    return f"""<section class="card">
<h2>Paper account</h2>
<div class="acct-top"><span class="equity">${s['equity']:,.2f}</span>
<span class="delta {sign}">{s['total_return']*100:+.2f}%</span></div>
<span class="posmeta">from ${s['start_equity']:,.0f} &middot; simulated, no real money</span>
{equity_spark(book.get('equity_curve', []))}
<div class="acct-grid">
<div><span class="lab">Cash</span><span class="val">${s['cash']:,.0f}</span></div>
<div><span class="lab">Realised</span><span class="val {'go-c' if s['realised_pnl'] >= 0 else 'stop-c'}">${s['realised_pnl']:+,.0f}</span></div>
<div><span class="lab">Open P&amp;L</span><span class="val {'go-c' if s['unrealised_pnl'] >= 0 else 'stop-c'}">${s['unrealised_pnl']:+,.0f}</span></div>
<div><span class="lab">Win rate</span><span class="val">{wr}<small> {s['wins']}/{s['closed_trades']}</small></span></div>
</div>
<div class="target"><div class="target-top">
<span class="lab">Goal &middot; ${GOAL_EQUITY:,.0f} by {GOAL_DATE[5:]}</span>
<span class="val">{(s['equity'] - s['start_equity']) / (GOAL_EQUITY - s['start_equity']) * 100:+.1f}%</span></div>
<div class="pos-bar"><i style="width:{min(max((s['equity'] - s['start_equity']) / (GOAL_EQUITY - s['start_equity']), 0), 1)*100:.0f}%"></i></div>
<p class="note">${GOAL_EQUITY - s['equity']:,.0f} to go. Measured probability for the
current engine is near zero over the remaining sessions &mdash; the edge is real but
small. See target_test.py.</p></div>
<div class="target"><div class="target-top"><span class="lab">Today &middot; target ${risk.DAILY_PROFIT_TARGET:,.0f}</span>
<span class="val {'go-c' if s['day_realised'] >= 0 else 'stop-c'}">${s['day_realised']:+,.2f}</span></div>
<div class="pos-bar"><i style="width:{min(max(s['day_realised'] / risk.DAILY_PROFIT_TARGET, 0), 1)*100:.0f}%"></i></div></div>
{why}</section>
<section class="card"><h2>Open positions</h2><table>
<thead><tr><th>Position</th><th class="num">Entry</th><th class="num">Last</th><th class="num">P&amp;L</th></tr></thead>
<tbody>{rows}</tbody></table></section>
{arms_tbl}
{closed_tbl}"""


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
<header><span class="eyebrow">Morning Desk &middot; {RISK*100:.0f}% risk per trade</span>
<span class="asof">as of {d['stamp']}</span></header>
<section class="verdict" data-state="{state}"><h1>{verdict}</h1><p>{why}</p></section>
{account_panel(d)}
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
<footer>Snapshot, not live &mdash; rebuilt each weekday morning before the open.<br>
The account is simulated. Entries fill at the decision price, exits at the stop or
target, 5bp cost each way. Real fills would be worse.<br>
Output of fixed rules. Not a recommendation, and not evidence of an edge.</footer>
</div>"""


def payload(d):
    """Everything the page needs, as plain JSON. No markup built in Python."""
    book, st = d["book"], d["stats"]
    reg, c = d["reg"], d["chosen"]

    if not reg["green"]:
        verdict, why = "NO TRADE", "Regime is red. The long-only swing rules stand down."
    elif c is None:
        verdict, why = "NO TRADE", "Regime is clear, but nothing passes the checklist today."
    else:
        verdict, why = "TRADE", f"{c['ticker']} passes every filter."

    lines = [x for x in (d.get("rsi2"), d.get("forced")) if x]
    if d.get("skip"):
        lines.append("Swing rules: " + d["skip"])

    left = int(np.busday_count(datetime.now(timezone.utc).date().isoformat(),
                               GOAL_DATE))
    return {
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "intraday": bool(d.get("intraday")),
        "equity": st["equity"], "start_equity": st["start_equity"],
        "cash": book["cash"], "curve": book.get("equity_curve", []),
        "positions": book["positions"], "closed": book.get("closed", []),
        "arms": st["arms"], "day_realised": st["day_realised"],
        "day_target": risk.DAILY_PROFIT_TARGET,
        "goal_target": GOAL_EQUITY,
        "goal_date": datetime.strptime(GOAL_DATE, "%Y-%m-%d").strftime("%b %-d")
                     if os.name != "nt" else
                     datetime.strptime(GOAL_DATE, "%Y-%m-%d").strftime("%b %#d"),
        "days_left": max(left, 0),
        "verdict": verdict, "why": why, "lines": lines,
        "regime": {"green": reg["green"], "spy": reg["spy"], "s10": reg["s10"],
                   "s20": reg["s20"], "s50": reg["s50"]},
        "passing": [{"ticker": r["ticker"], "price": r["price"], "mom": r["mom"]}
                    for r in d["passing"][:8]],
        "sectors": [{"sector": x["sector"], "m1": x["m1"], "m3": x["m3"]}
                    for x in d["secs"][:6]],
        "gaps": [{"ticker": g["ticker"], "pre": g["pre"], "gap": g["gap"]}
                 for g in d["gaps"][:8]],
    }


if __name__ == "__main__":
    import sys
    intraday = "--intraday" in sys.argv
    d = trade(build(intraday=intraday), intraday=intraday)
    import ui
    open("dashboard.html", "w", encoding="utf-8").write(ui.render(payload(d)))
    s = d["stats"]
    print(json.dumps({"mode": "intraday" if intraday else "morning",
                      "regime": "GREEN" if d["reg"]["green"] else "RED",
                      "day_realised": s["day_realised"],
                      "candidate": d["chosen"]["ticker"] if d["chosen"] else None,
                      "skip": d["skip"], "passing": len(d["passing"]),
                      "gaps": len(d["gaps"]), "equity": s["equity"],
                      "open": s["open_positions"], "closed": s["closed_trades"],
                      "realised": s["realised_pnl"],
                      "closed_today": [f'{c["ticker"]} {c["reason"]} {c["pnl"]:+.2f}'
                                       for c in d["closed"]]}))
