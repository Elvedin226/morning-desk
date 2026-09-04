# sm_research.md — 0DTE data availability, 0DTE evidence, small-account methods, hard constraints

Research date: 2026-09-03. Account: $421 → $1,000 by 2026-12-31 (~85 sessions, +1.023%/day compounded).

**Reading key:** `MEASURED` = I ran it or a named source published a number from data. `CLAIM` = asserted, unaudited, or marketing. `REGULATORY` = primary rule text.

---

## LEAD FINDINGS (these change the picture most)

1. **The premise that 0DTE cannot be honestly backtested is REFUTED.** A usable 1-minute historical option source exists at **$40/month** (ThetaData Options Value, 4 years back). A **genuinely free** source of 1-minute option bars back to **February 2024** exists (Alpaca, indicative feed). Both are enough to test 0DTE properly. See §1.
2. **The PDT rule no longer exists.** SEC approved elimination of the $25,000 minimum equity requirement and the pattern-day-trader designation itself; effective **June 4, 2026**. The briefing premise ("PDT requires $25k") is out of date. **But it does not help at $421** — the $2,000 Reg-T/FINRA margin minimum still binds, so this account is a cash account regardless. See §4.
3. **The binding constraint at $421 is not the PDT rule — it is cash-account T+1 settlement.** It roughly halves the number of compounding opportunities, raising the required per-trade return from ~1.02% to **~2.08%**. See §4.
4. **I measured live 0DTE/1DTE bid-ask spreads.** They are 1–1.4% of premium near the money but **25–40% of premium** on the cheap far-OTM contracts a $421 account can actually buy in size. This is the mechanical explanation for the already-measured result that far-OTM EV collapses to −99.7%. See §2.
5. **Leveraged ETFs are worse than the index-call path, and fail the same bias test.** MEASURED: bootstrapped TQQQ 2010–2026 gives **P(touch $1,000) = 2.7%** vs 10.0% for index calls — and 13.6% of that is the 2020–21 melt-up again. See §3.

---

# 1. 0DTE data availability — the decisive section

## 1.1 First, a correction to the framing

The claim "a 0DTE contract opens and expires the same session, so daily bars cannot resolve it" is **half right**.

- Correct for *intraday entry and exit* (e.g. enter 10:00, exit 14:00). That genuinely needs intraday marks.
- **Incorrect for open-to-expiry.** A 0DTE contract's *only* trading day is its expiry day, so a **daily OHLC bar for that contract gives you its open price**, and settlement is intrinsic value against the close/settlement print. Open-to-expiry 0DTE is therefore testable from *daily option aggregates* alone — which are far cheaper and more available than intraday quotes.

This matters because it means a first-pass 0DTE test is available at **$0** via daily option aggregates, before spending anything on intraday data. Caveat: a daily bar's open is a *trade* print, not a quote, so it understates the spread you would actually pay — results from this method are optimistic and must be haircut by the spreads measured in §2.3.

## 1.2 Provider-by-provider findings

| Source | Granularity | History | Price | Free tier? | Verdict for 0DTE |
|---|---|---|---|---|---|
| **ThetaData** | 1-min bars (Value); tick (Standard/Pro) | 4y / 8y / 12y | **$40 / $80 / $160 per mo** | No | **BEST value. Under budget. Unblocks full intraday 0DTE testing.** |
| **Alpaca** | 1-min option bars | **since Feb 2024** | **$0** (indicative feed); $99/mo OPRA | **YES** | **Only genuinely free intraday option source found.** ~2.5y of 0DTE. See caveats 1.3. |
| **Databento** | OPRA tick/MBO | up to 10y | $199/mo sub; pay-as-you-go | **$125 signup credit** | Credit alone can pull a meaningful 0DTE sample for $0. |
| **OptionsDX** | EOD → **minutely** | 2010–2023 (SPX) | **$0.00–$50.00** per product | **Partly free** | Cheap; SPX-specific; stops at 2023 (no recent regime). |
| **Polygon.io** (now Massive.com) | daily aggs, tick on higher tiers | 2y free / 5y Starter | Starter $29/mo (options priced separately per asset class) | Free tier = EOD, 5 calls/min | Free tier workable for the §1.1 **daily-bar** method only. |
| **ORATS** | 1-min, full greeks/IV, 5,000+ symbols | ~3 years | Enterprise (not published; well above $50) | No | Overkill and over budget. |
| **CBOE DataShop** | 1-min intraday ad-hoc | long | **$2,500/mo** (1-min); $750/mo (10-min). SPX underlying prints need a CGIF licence from **$1,000/mo** | No | **Far out of budget. Rule it out.** |
| **CBOE free downloads** | daily aggregate volume, put/call ratios only | 2006– | $0 | Yes | **No option prices at all.** Useless for backtesting. |
| **OptionMetrics / WRDS** | EOD IvyDB, no intraday | long | Academic-institution licence only | No | Not retail-purchasable; and EOD-only anyway. |
| **DoltHub** (`post-no-preference/options`) | **EOD chains only** | 2019– | $0 | Yes | Free, but **EOD snapshot — will not resolve 0DTE intraday.** |

## 1.3 Caveats on the free Alpaca route (read before relying on it)

- History starts **Feb 2024**. That is ~2.5 years — a single, mostly-benign regime. Given the already-established finding that option EV was entirely a 2020–21 artefact, **a 2024–2026-only sample carries exactly the same period-bias risk** and cannot be treated as representative.
- The free feed is **"indicative"**, a derivative of OPRA, not the consolidated NBBO. Spreads and marks will not exactly match what you could have traded.
- Requires an Alpaca account and API keys.

## 1.4 Bottom line on §1

The honest statement is **not** "0DTE cannot be backtested." It is:

> 0DTE **can** be backtested. Open-to-expiry can be tested for **$0** from daily option aggregates. Full intraday entry/exit can be tested for **$0** (Alpaca, Feb 2024→, indicative feed, single regime) or properly for **$40/month** (ThetaData, 4 years, 1-minute). The binding limitation is no longer data availability — it is that any sample short enough to be free is too short to escape period bias.

---

# 2. 0DTE — what the evidence actually says

## 2.1 Peer-reviewed / academic (highest weight)

**Beckmeyer, Branger & Gayda — "Retail Traders Love 0DTE Options... But Should They?"** (SSRN 4404704)
- 0DTE trades **lose 4.7%** relative to other option trades; non-0DTE trades earn +0.19%. `MEASURED`
- Retail investors lost **~$184,000 on the average day** in 0DTE S&P 500 options. `MEASURED`
- Average retail losses of **5–9%**. `MEASURED`
- Mechanism: 0DTE have much lower absolute prices and therefore **much larger *relative* bid-ask spreads**. `MEASURED`

**Bogousslavsky & Muravyev — "An Anatomy of Retail Option Trading"**
- Retail option traders are net losers in aggregate, via bid-ask spread, time decay, and adverse selection by high-frequency market makers. `MEASURED`

**Vilkov et al., 0DTE strategy study** (SPXW, Sept 2016 – Jan 2026; Cboe 30-min bars + ThetaData 1-min) — the single most relevant quantitative study found:
- **Buyers lose:** "ATM-to-OTM calls and most puts lose on average." `MEASURED`
- **Sellers win, but barely:** selling slightly-OTM calls/puts positive in **up to 75% of observations** — note this is a *win rate*, not an expectancy. `MEASURED`
- **A positive 0DTE variance risk premium exists, but its economic magnitude is small** — median realised VRP ≈ **0.0011% of underlying** — "difficult to monetize after realistic trading frictions." `MEASURED`
- **Costs destroy most of it:** iron butterfly/condor goes from **+0.77 gross Sharpe to −0.20 net**. Unconditional strategies mostly land between negative and ~0.5 gross Sharpe. `MEASURED`
- Best conditional (ML-filtered, out-of-sample) result: put ratio spreads 1.18 gross / **0.93 net Sharpe**; all-strategies basket 0.81 gross / **0.25 net**. `MEASURED`
- Tail risk: expected shortfall at 1% ran **0.58–1.58% of underlying**, and the authors explicitly refuse to winsorize because "rare tail realizations are central to the risk." `MEASURED`

**Interpretation for this account.** The *only* side with positive measured expectancy is the **seller** side, and a 0DTE seller is short undefined or large-defined risk — which at $421 is either not permitted (see §4) or a single-event wipeout. The side available to a $421 cash account is the **buyer** side, which is the side every study measures as losing. That is the central conflict.

## 2.2 Exchange (CBOE) research — read carefully

- CBOE: 0DTE are "highly liquid with higher trading volumes and very tight bid-ask spreads." `CLAIM (exchange, has a commercial interest in volume)`
- CBOE market-impact study: market-maker net gamma is small, 0.04%–0.17% of daily S&P futures liquidity. `MEASURED`
- ~1.5 million 0DTE options trade daily, ~half of all SPX-linked options trades (2025). `MEASURED`

**Note the tension:** CBOE's "very tight spreads" is true in *absolute cents* and false in *percent of premium* — which is the only thing that matters to a buyer's return. My own measurement below resolves it.

## 2.3 MEASURED BY ME — live SPY spreads, 2026-09-03 snapshot

Method: pulled the live SPY chain via yfinance; derived true spot **$773.08** from put-call parity (the quoted last price was stale by ~1%); computed `(ask − bid) / mid` for OTM contracts only.

| DTE | Side | OTM band | Median spread as % of mid | Median mid | 1 contract costs |
|---|---|---|---|---|---|
| 1 | call | 0–0.5% | **1.4%** | $1.02 | $102 |
| 1 | call | 0.5–1.0% | **4.1%** | $0.26 | $26 |
| 1 | call | 1.0–2.0% | **25.4%** | $0.04 | $4 |
| 1 | put | 0–0.5% | **1.1%** | $1.27 | $127 |
| 1 | put | 0.5–1.0% | **2.6%** | $0.45 | $45 |
| 1 | put | 1.0–2.0% | **9.6%** | $0.10 | $10 |
| 1 | put | 2.0–4.0% | **40.0%** | $0.02 | $2 |
| 3 | call | 1.0–2.0% | 9.9% | $0.10 | $10 |
| 15 | call | 1.0–2.0% | 1.4% | $2.30 | $230 |
| 15 | put | 2.0–4.0% | 1.2% | $1.70 | $170 |

**The finding:** spread-as-%-of-premium is **inversely proportional to premium**. Both CBOE and the academics are right — spreads are a couple of cents (tight in absolute terms) but those same cents are 25–40% of a $0.02–$0.04 contract.

**This independently explains the already-measured result** that 20% OTM = EV −99.7% and 35% OTM = EV −100.0%. It is not only that convexity stops helping — it is that you are paying a 25–40% round-trip transaction tax on every entry. Cheap lottery-ticket contracts are precisely where the frictional cost is maximal.

**Caveats:** single snapshot; taken outside/near the close so spreads are likely *wider* than mid-session; SPY not SPX; yfinance quotes are not guaranteed NBBO. Treat the *pattern* as robust and the *levels* as indicative.

## 2.4 Influencer / marketing claims — named and dismissed

Search results for 0DTE are dominated by SEO content selling something: SpotGamma, CoveredCallCalculator "0DTE SPX Strategies Guide 2026", optionsai.com, tradingview script promos. None publish audited track records, sample sizes, or net-of-cost results. Per the standing rule: anything promising consistent large returns on a small account is selling something. **No weight given.**

---

# 3. Other small-account growth methods, ranked by realistic viability at $421

### Rank 1 — Index options (already the measured best path)
P(hit $1,000) = 10.0%, P(end < $150) = 90.0%, per prior work at 5% OTM, 100% sizing. Nothing found in this research beats it. §2.3 adds the reason not to go further OTM.

### Rank 2 — Leveraged ETFs — MEASURED BY ME, and it underperforms
Bootstrapped 20,000 paths × 85 sessions from real daily returns, $421 start, $1,000 target, broken down by period per the standing bias warning:

| ETF | Period | n days | mean/day | sd/day | **P(touch $1,000)** | P(end<$150) | median end |
|---|---|---|---|---|---|---|---|
| TQQQ | 2010–2015 | 1482 | +0.215% | 3.24% | 1.1% | 0.0% | $487 |
| TQQQ | 2016–2019 | 1006 | +0.201% | 3.17% | 0.8% | 0.0% | $480 |
| TQQQ | **2020–2021 melt-up** | 505 | +0.409% | 5.23% | **13.6%** | 0.9% | $530 |
| TQQQ | **2022 bear** | 251 | −0.440% | 6.01% | 1.8% | 18.3% | $249 |
| TQQQ | 2023–2026 | 919 | +0.302% | 3.78% | 3.8% | 0.0% | $514 |
| **TQQQ** | **FULL 2010–2026** | 4163 | +0.215% | 3.85% | **2.7%** | 0.1% | $479 |
| SPXL | FULL 2010–2026 | 4163 | +0.158% | 3.21% | **0.6%** | 0.1% | $464 |

**Conclusions:** (a) **2.7% vs 10.0% — leveraged ETFs are strictly worse than index calls for this target.** (b) The **same 2020–21 artefact appears again** (13.6% vs 0.8–3.8% elsewhere) — a third independent confirmation of that bias. (c) The tradeoff is inverted: near-zero ruin risk (0.1%) but also near-zero success. Not enough volatility to reach the target in 85 days. (d) Published evidence confirms decay: TQQQ's effective leverage fell to ~2.1× over long horizons; ~2.4× YTD 2026. `MEASURED`

### Rank 3 — Micro futures (MES) — viable to *open*, not to *survive*
- Day-trade margin: **$40–$138 per MES contract** depending on broker (AMP ~$40, StoneX $138). Overnight maintenance ~$1,200–$2,465 — **overnight holding is impossible at $421**. `MEASURED`
- Brokers will let you open with $400–$500. `CLAIM`, but consistently reported.
- **MEASURED BY ME** at SPX 7,747.71: one MES = **$38,739 notional**, i.e. **92× leverage** on $421. A **1% index move = $387 = 92% of the account.** A **1.09% adverse move wipes out the entire $421.**
- SPX moves >1% on a meaningful fraction of days. Ruin probability approaches certainty over 85 sessions.
- Futures are exempt from PDT and from cash-settlement rules — the *only* structural advantage, and it does not compensate.

### Rank 4 — Crypto perpetuals with leverage
No credible peer-reviewed measurement of retail perp outcomes was found. Structurally: 24/7 liquidation, funding-rate carry bleed, exchange counterparty risk, and non-existent US retail regulatory protection. High leverage on $421 replicates the MES ruin arithmetic with worse counterparty risk. **No measured evidence supports it; no basis to rank it above MES.**

### Rank 5 (do not use) — Prop firm challenges
- **No firm publishes audited pass rates.** Every number below comes from marketing/SEO sites or community surveys. `CLAIM` throughout.
- FTMO: no official rate; community estimate **10–12%** combined challenge+verification.
- Topstep: **15–20%** (self-estimated). Apex: **15–20%** first-attempt (self-reported, marketing).
- Industry-wide: **5–10% per attempt**; **5–14%** of purchased challenges reach funding; **~7% of all challenge buyers ever receive a payout**; of those funded, ~45% get at least one payout.
- **Structural problem:** the challenge fee is a **guaranteed 100% loss** on failure, and the payout is not your capital. Taking the most favourable published figure (7% ever get paid), this is a negative-EV product whose vendor is the counterparty to your failure.
- Regulatory note: the sector has drawn enforcement attention (FTC action against My Forex Funds). **This is the clearest "selling something" category in the entire research set. Name it and move on.**

---

# 4. Hard constraints that actually apply at $421

## 4.1 The PDT rule is gone — and it still doesn't help you
`REGULATORY` — FINRA Regulatory Notice 26-10; SEC approval 2026-04-14.
- The **$25,000 minimum equity requirement is eliminated**, as is the **pattern-day-trader designation itself**. Effective **June 4, 2026**.
- Replaced by an **intraday margin deficit** framework: brokers compute the largest deficiency after each transaction; customers must satisfy it "as promptly as possible"; repeated failure to satisfy within 5 business days triggers a **90-day freeze on new short positions**.
- **Brokers have until 2027-10-20 to implement.** So whether this is live *for you* depends entirely on your broker today. **Verify with your broker; do not assume.**

**Why it doesn't help at $421:** day trading without PDT still requires a **margin account**, and FINRA/Reg-T set a **$2,000 minimum equity** to open one and use leverage. At $421 you cannot have a functioning margin account. You are in a **cash account**, where the PDT rule never applied anyway. `REGULATORY`

## 4.2 The constraint that actually binds: T+1 settlement in a cash account
- Options settle **T+1**. In a cash account you may buy with settled cash and sell the same day (fine), but **using unsettled proceeds to buy and then selling before the original trade settles is a Good Faith Violation.**
- Practical effect: after a round trip on day 1, proceeds are unsettled until end of day 2. You can buy on day 2 with unsettled funds but **cannot sell that position on day 2** without a GFV. **You can effectively complete a full round trip only every other session.**
- **Three GFVs in a rolling 12 months ⇒ account restricted to settled-cash-only for 90 days.** At 85 sessions total, a 90-day restriction is a run-ending event.

**MEASURED consequence:** 85 sessions supports only ~**42 round trips**, not 85. Required return per trade rises from **+1.023%/session to ~+2.08%/trade**. Every prior Monte Carlo result assuming one trade per session is therefore **optimistic on trade count**, and the compounding requirement is roughly double what it appears.

## 4.3 Options approval
- **Level 1:** covered calls, cash-secured puts. Both require ~100 shares or full cash securing — **not achievable at $421** (100 SPY ≈ $77,000).
- **Level 2:** buying calls and puts. **This is the level that matters**, and it is the level that is workable in a cash account with no margin.
- **Level 3+ (spreads):** typically **requires a margin account**, and brokers commonly impose account-equity minimums around **$10,000**. **Not available at $421.**
- **Consequence:** defined-risk spreads and every net-credit structure are closed to this account. **Long premium is the only options structure available** — which is exactly the side §2.1 measures as losing. This is the single most important structural constraint in the whole report.

## 4.4 Contract minimums and granularity
- One option contract = 100 shares. Minimum position size is one contract.
- **MEASURED (§2.3):** at $421 you can afford 1 near-ATM 1DTE SPY contract (~$102–$127), or ~3–4 of them, or dozens of far-OTM tickets at $2–$10 each — but the latter carry the 25–40% spread tax.
- **SPX index options are effectively unavailable**: SPX is ~10× SPY, so one near-ATM SPX 0DTE contract costs multiples of the whole account. SPY (or XSP, 1/10 SPX) is the only viable underlying at this size. Note the prior work's "index calls" result was modelled on index moves — **execution must be in SPY/XSP, and §2.3's spread haircut applies.**
- Commissions (~$0.50–$0.65/contract each way at most retail brokers) are **another 1–2% of a $26–$50 premium**, on top of spread.

## 4.5 Summary table of what is and isn't possible at $421

| Capability | Available? | Blocker |
|---|---|---|
| Buy calls/puts (Level 2) | **Yes** | — |
| Sell cash-secured puts / covered calls | No | Needs ~$77k |
| Spreads, iron condors, any credit structure | **No** | Level 3 + margin account + ~$10k |
| Sell 0DTE premium (the only +EV side measured) | **No** | Same as above |
| Margin account / leverage | **No** | $2,000 Reg-T minimum |
| Day trade unrestricted | Yes, in cash account | But T+1 limits to ~every other day |
| Hold MES overnight | **No** | ~$1,200–$2,465 maintenance margin |
| Day trade 1 MES | Technically yes ($40–$138) | 1.09% adverse move = total ruin |
| SPX index options | **No** | One contract > account |
| SPY / XSP options | Yes | — |

---

## Consolidated source list

**Data providers**
- ThetaData pricing — https://www.thetadata.net/pricing
- Alpaca historical option data — https://docs.alpaca.markets/us/docs/historical-option-data
- Databento OPRA pricing — https://databento.com/opra-pricing-preview · https://databento.com/pricing
- OptionsDX — https://www.optionsdx.com/ · https://www.optionsdx.com/product/spx-option-chain/
- CBOE DataShop — https://datashop.cboe.com/option-quote-intervals · https://datashop.cboe.com/faqs
- CBOE free historical data — https://www.cboe.com/us/options/market_statistics/historical_data/
- Polygon/Massive options REST — https://massive.com/docs/rest/options/overview
- ORATS 1-minute data — https://orats.com/one-minute-data
- DoltHub options dataset writeup — https://medium.com/@codythedatainvestor/wow-here-is-a-free-historical-option-prices-database-for-your-backtesting-99046f1dc128

**0DTE evidence**
- Beckmeyer, Branger & Gayda, "Retail Traders Love 0DTE Options... But Should They?" — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4404704
- Bogousslavsky & Muravyev, "An Anatomy of Retail Option Trading" — https://www.lsu.edu/business/files/event-files/2025-finance-mardi-gras/retail_option_trading_v2.pdf
- Vilkov 0DTE strategy study (annotated) — https://github.com/vilkovgr/0dte-strategies/blob/main/docs/paper/paper-annotated.md
- 0DTE option pricing (Northern Finance Assoc.) — https://portal.northernfinanceassociation.org/viewp.php?n=2240145012
- JHU Carey, "Risk and reward: New insights on 0DTE option trading" — https://carey.jhu.edu/news/risk-reward-insights-0dte-option-trading
- CBOE, "Evaluating the Market Impact of SPX 0DTE Options" — https://www.cboe.com/insights/posts/volatility-insights-evaluating-the-market-impact-of-spx-0-dte-options/

**Regulatory**
- FINRA Regulatory Notice 26-10 (PDT elimination) — https://www.finra.org/rules-guidance/notices/26-10
- FINRA intraday margin requirements — https://www.finra.org/investors/insights/intraday-margin-requirements
- FINRA Rule 4210 — https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210
- Schwab, SEC approves scrapping $25,000 minimum — https://www.schwab.com/learn/story/sec-approves-scrapping-25000-day-trader-minimum
- Fidelity, avoiding cash trading violations — https://www.fidelity.com/learning-center/trading-investing/trading/avoiding-cash-trading-violations
- Schwab, cash account violations — https://www.schwab.com/learn/story/avoid-these-violations-when-trading-cash

**Other methods**
- StoneX futures day-trading margins — https://futures.stonex.com/day-trading-margins
- Prop firm statistics (all unaudited) — https://www.quantvps.com/blog/prop-firm-statistics · https://traderssecondbrain.com/guides/prop-firm-pass-rate · https://damnpropfirms.com/trading-guides/prop-firm-evaluation-pass-rates-statistics-reality-check/
- Leveraged ETF decay — https://247wallst.com/investing/etf/2026/08/05/this-39-77-billion-leveraged-etf-jumped-10-09-tuesday-but-volatility-decay-cost-holders-millions/

*This document reports evidence and constraints only. It is not investment advice and recommends no trade.*
