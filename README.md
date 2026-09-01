# morning-desk

Swing-trading decision tool. Applies a fixed checklist to a universe of liquid
US equities and emits one answer: a named candidate with entry, stop, position
size and target — or an explicit NO TRADE.

**What this is:** consistency and risk control.
**What this is not:** an edge. Of eight strategies tested while building this,
one survived, and it died on transaction costs. Nothing here beats that record.

## How it runs

```
GitHub Actions (13:00 UTC weekdays)
  -> cloud_bot.py fetches data, applies the rules, writes dashboard.html
  -> commits it back to this repo
        |
Claude routine reads raw.githubusercontent.com and publishes it
        |
Phone opens the same private URL, already updated
```

The Actions runner does the data work because the Claude sandbox blocks every
market-data host but can reach GitHub.

## The rules

Kept, because they survived testing or bound losses:

| rule | why |
|---|---|
| regime filter | only trade when SPY's 10 > 20 > 50 SMA and rising |
| momentum | 12-1 momentum, the one factor with real literature behind it |
| trend structure | price above SMA20 > SMA50 |
| not extended | within 15% of the 52-week high, under 10% above the 20-day |
| earnings block | no entry if earnings falls inside the 21-day hold window |
| 1% risk sizing | position derived from the stop, never chosen |

Dropped, because they were tested and failed (see `ablate.py`):

| rule | result |
|---|---|
| breakout entry | **-0.76% edge, t = -2.97** — significantly negative alone, and adding it to momentum cut the edge from +3.61% to +1.74% |
| volatility "coil" | cut the sample 10x, added nothing measurable |
| ADR / volume taper | same |

## Local use

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

.venv/Scripts/python.exe bot.py              # today's decision
.venv/Scripts/python.exe premarket.py        # pre-market gaps
.venv/Scripts/python.exe watchlist.py        # ranked candidates
.venv/Scripts/python.exe main.py --walkforward --compare   # validate a strategy
```

`cloud_bot.py` is a standalone single-file version of `bot.py` + `publish.py`
with no local imports — that is the one Actions runs. Change a rule in `bot.py`
and it does **not** propagate; update both.

## Files

| file | purpose |
|---|---|
| `bot.py` | the decision engine |
| `cloud_bot.py` | standalone version for CI |
| `validate.py` | walk-forward + permutation baseline |
| `screener.py` | dip and volatility-compression studies |
| `flag.py` / `ablate.py` | high-tight-flag test and layer ablation |
| `overnight.py` | the one surviving edge |
| `lottery.py` | convex bet sizing / ruin simulation |
| `ivcheck.py` | implied vol, spreads, option ladders |

Educational research. Not investment advice.
