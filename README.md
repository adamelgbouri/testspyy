# S&D — Commodity Supply & Demand Trading Desk

Single-file Streamlit application. 27 contracts across Energy, Metals, Ags and Freight. Live settlement data from Yahoo Finance, real dated forward curves, Black-76 pricing, parametric VaR.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aeg-snd.onrender.com/)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Live:** [https://aeg-snd.onrender.com](https://aeg-snd.onrender.com/) — no install, no credentials. Source available on request.

---

## Read this first: what is real and what is not

Anyone who has been burned by a dashboard passing off a model as a market will want this table before anything else.

| Output | Source | Trust it for |
|---|---|---|
| Spot / front month | Yahoo continuous contract (`CL=F`, `GC=F`…), 5-min cache | Directional reference, marking a blotter |
| Price history (2y daily) | Yahoo daily OHLCV, auto-adjusted, 1-h cache | Realised vol, chart context |
| **Forward curve** | **Individually downloaded dated contracts** (`CLN25.NYM`, `GCZ25.CMX`…) | **Structure, spreads, roll** — these are actual quoted contracts, not an interpolation |
| Curve where Yahoo has no listing | Cost-of-carry, flagged `model (cost-of-carry)` and drawn with different markers | Shape intuition only |
| Options, Greeks, vol surface | Black-76 on the live forward; surface is parametric (ATM + skew + curvature + vov) | Sensitivities and scenario work. **Not a fitted market surface** — no listed option prices are pulled |
| Supply / demand / stocks balances | Model-driven | Scenario framing only. There is no free fundamental API |
| Regional flows | IEA/USDA-style static estimates | Orientation, not decision-grade |
| Macro (GDP, CPI, rate, PMI) | Model-estimated | Overlay context only |
| LME Al/Ni, EUA, API2, BCI/BPI | Hardcoded constants, badged `FALLBACK`. **They do not move.** | Nothing. Placeholders pending a paid feed |

Every page carries a `LIVE yfinance` or `FALLBACK` badge in the header, the curve table has a `Source` column per contract, and the heatmap reports its real-data hit rate. But the badge is coarser than the reality: it keys off the spot ticker only, so **two commodities show `LIVE` while their curve is modelled**. See [Coverage](#coverage--live-vs-fallback-per-output) for the per-contract breakdown, and note the LME Copper defect flagged there.

---

## Curve construction

The forward curve page is the part with real desk value, so the mechanics are worth stating:

- **Expiry-aware.** Contracts are skipped once past expiry (approximated at the 20th of the month preceding delivery), so you get the true front month rather than a stale, thinly-quoted expired ticker.
- **Delivery-cycle aware.** Each commodity carries its own active month string. Gold prices GJMQVZ, Sugar #11 prices HKNV, Soybeans FHKNQUX, WTI all twelve. The app does not fabricate a Jan gold contract.
- **Batched.** All dated tickers for a commodity are pulled in one grouped download.
- **Structure classified** as CONTANGO / BACKWARDATION / FLAT on a ±0.5% front-to-back threshold, with the percentage carry displayed.
- **Degrades explicitly.** If fewer than two dated contracts return, the whole curve falls back to cost-of-carry ($F = S e^{(r + u - y)T}$) using the commodity's stored storage cost and convenience yield — and says so.

---

## Pages

| Page | Content |
|---|---|
| **Dashboard** | Live spot, 1D change vs previous close, 2y price history. Treemap heatmap that fetches **real closes for two user-selected dates** and colours by the actual move between them |
| **Supply & Demand** | Balance series with surplus/deficit bars. ±20% supply, ±20% demand and GDP sliders for scenario work |
| **Regional Flows** | Net exporters vs importers on a world map, bubble scaled to imbalance. Units are per-commodity and physical: mb/d, bcf/d, Mbu/y, M 60-kg bags/y, kt/y, vessels |
| **Futures Curve** | Dated contract prices, structure classification, real-vs-model contract count |
| **Options & Greeks** | Black-76 on the live forward. Price, delta, gamma, vega, theta, rho for calls and puts. Payoff net of premium, Greeks profiles across strike. Put-call parity residual displayed on every run |
| **Vol Surface** | 3D parametric surface, adjustable ATM / skew / curvature / vol-of-vol. Smile by maturity, ATM term structure |
| **Positions & P&L** | Blotter marked to live spot. Gross long, gross short, net exposure, per-line P&L and return |
| **Risk** | Parametric VaR and CVaR at 90/95/99%, 1–30 day horizon, per-position decomposition. Stress grid from −30% to +30% |
| **Monte Carlo** | GBM from the live spot, up to 2 000 paths. P5–P95 fan, terminal distribution |
| **Macro Overlay** | GDP index, CPI YoY, policy rate, PMI across 8 economies |
| **Events** | EIA weekly, WASDE, OPEC+, IEA MOMR, FOMC, ECB, CPI, Baker Hughes, LME Week |

---

## Coverage — live vs fallback, per output

"Live" is not one switch per commodity. **Spot** and **curve** are fed separately: spot comes from the continuous contract (`yf_ticker`), the curve from dated contract tickers (`yf_fmt`). A commodity can have one without the other, so read the two columns independently.

| Contract | Sector | Spot & history | Forward curve |
|---|---|---|---|
| WTI Crude | Energy | 🟢 `CL=F` | 🟢 `CL{M}{YY}.NYM` |
| Brent Crude | Energy | 🟢 `BZ=F` | 🟢 `BZ{M}{YY}.NYM` |
| Henry Hub Nat Gas | Energy | 🟢 `NG=F` | 🟢 `NG{M}{YY}.NYM` |
| RBOB Gasoline | Energy | 🟢 `RB=F` | 🟢 `RB{M}{YY}.NYM` |
| Heating Oil (ULSD) | Energy | 🟢 `HO=F` | 🟢 `HO{M}{YY}.NYM` |
| **ICE Gasoil** | Energy | 🟢 `LGO=F` | 🟡 **model** |
| Carbon EUA | Energy | 🔴 63.0 | 🔴 model |
| Coal API2 | Energy | 🔴 108.0 | 🔴 model |
| Gold | Metals | 🟢 `GC=F` | 🟢 `GC{M}{YY}.CMX` |
| Silver | Metals | 🟢 `SI=F` | 🟢 `SI{M}{YY}.CMX` |
| Copper (COMEX) | Metals | 🟢 `HG=F` | 🟢 `HG{M}{YY}.CMX` |
| Platinum | Metals | 🟢 `PL=F` | 🟢 `PL{M}{YY}.NYM` |
| Palladium | Metals | 🟢 `PA=F` | 🟢 `PA{M}{YY}.NYM` |
| **LME Copper** | Metals | ⚠️ `HG=F` — see below | 🟡 model |
| LME Aluminium | Metals | 🔴 2 390 | 🔴 model |
| LME Nickel | Metals | 🔴 15 800 | 🔴 model |
| Corn | Ags | 🟢 `ZC=F` | 🟢 `ZC{M}{YY}.CBT` |
| Wheat (CBOT) | Ags | 🟢 `ZW=F` | 🟢 `ZW{M}{YY}.CBT` |
| Soybeans | Ags | 🟢 `ZS=F` | 🟢 `ZS{M}{YY}.CBT` |
| Sugar #11 | Ags | 🟢 `SB=F` | 🟢 `SB{M}{YY}.NYB` |
| Coffee (Arabica) | Ags | 🟢 `KC=F` | 🟢 `KC{M}{YY}.NYB` |
| Cocoa | Ags | 🟢 `CC=F` | 🟢 `CC{M}{YY}.NYB` |
| Live Cattle | Ags | 🟢 `LE=F` | 🟢 `LE{M}{YY}.CME` |
| Lean Hogs | Ags | 🟢 `HE=F` | 🟢 `HE{M}{YY}.CME` |
| Capesize BCI 5TC | Freight | 🔴 17 500 | 🔴 model |
| Panamax BPI 4TC | Freight | 🔴 11 800 | 🔴 model |

🟢 live Yahoo · 🟡 live spot, modelled curve · 🔴 hardcoded fallback constant, modelled curve

**Totals: 19/26 live spot · 17/26 live dated curve · 7 fully synthetic.**

### The three combinations, and what each means

1. **🟢 / 🟢 — live spot, live curve (17 contracts).** Both the mark and the term structure are quoted contracts. Contango/backwardation calls, calendar spreads and roll are all reading real prices. This is the tier worth acting on.
2. **🟡 / model — live spot, modelled curve (ICE Gasoil, LME Copper).** The spot ticks with the market; the curve does not. It is generated by cost-of-carry from that spot with a fixed storage and convenience yield, so **it will never show backwardation** unless $y > r + u$ by construction. Do not read structure off these two. The curve chart plots them as grey dashed diamonds rather than amber circles, and the table's `Source` column says `model (cost-of-carry)`.
3. **🔴 / model — fully synthetic (EUA, API2, LME Al/Ni, BCI, BPI, 7 contracts).** The "price" is a constant compiled into the source. It does not move, ever. Its 1-day change is always 0.00%, and it appears in the heatmap as a flat tile. It is a placeholder for a paid feed, nothing more.

### ⚠️ LME Copper is mispriced by construction

LME Copper is wired to `HG=F` — the **COMEX** contract, quoted in **$/lb**, while the app labels LME Copper in **$/mt**. It will therefore display something near `4.55` where the LME three-month is near `9 750`. Roughly a 2 200× error. It is not a unit-converted proxy; it is the wrong contract in the wrong unit.

Until a real LME feed is wired in, either ignore that row or treat it as COMEX copper mislabelled. Everything downstream of it — its VaR notional, any blotter line, its Monte Carlo start — inherits the error.

---

## Analytics

- **Black-76** — European options on futures. Full Greeks; delta per unit move, vega per 1% vol, theta per calendar day. Put-call parity checked against $C - P = e^{-rT}(F-K)$ each computation.
- **Vol surface** — parametric in log-moneyness: $\sigma(K,T) = \sigma_{ATM}(1 + \nu\sqrt{T}) + \beta\ln(K/F) + \gamma\ln(K/F)^2$.
- **Cost-of-carry** — $F = S e^{(r + u - y)T}$, storage $u$ and convenience yield $y$ stored per commodity.
- **Monte Carlo** — GBM, monthly steps, driftless under $Q$.
- **VaR / CVaR** — parametric, $\text{VaR} = |N| \sigma_d z_\alpha \sqrt{h}$, summed across the book. No correlation matrix — this is a conservative, undiversified aggregate.

---

## Stack

`streamlit` · `yfinance` · `pandas` · `numpy` · `scipy` (norm.cdf, brentq) · `plotly`

```bash
pip install streamlit plotly numpy pandas scipy yfinance
streamlit run commodity_trading_desk.py
```

---

## Where this sits

| | Purpose |
|---|---|
| **S&D** (this app) | Desk terminal — screen the market, mark a book, size a risk |
| [**CFCAP**](https://aeg-cfcap.streamlit.app) | Curve analytics — PCA, Schwartz-Smith, convenience/roll yield, 51 signals |
| [**CODAP**](https://aeg-codap.streamlit.app) | Derivatives pricing — vanilla, Asian, crack, calendar spread, swaps, barrier |
| [**Portfolio Optimizer**](https://aeg-markowitz.streamlit.app) | Markowitz allocation, 47 000+ instruments |

S&D screens the market. CFCAP analyses the curve. CODAP prices what is written on it.

---

## Known limitations

- **LME Copper points at COMEX `HG=F`** — wrong contract, wrong unit ($/lb shown as $/mt). Fix before using that row for anything.
- **The `LIVE` header badge is derived from the spot ticker alone**, so ICE Gasoil and LME Copper are badged live while their curves are cost-of-carry models. Trust the curve table's per-contract `Source` column over the header badge.
- **Seven contracts are constants.** EUA, API2, LME Aluminium, LME Nickel, Capesize and Panamax never tick. Their heatmap change is 0.00% by construction, not because the market was flat.
- **Modelled curves cannot backwardate** unless convenience yield exceeds rate plus storage in the registry. Absence of backwardation on a 🟡 or 🔴 row is an artefact, not a signal.
- **No listed option data.** Implied vols are parametric and seeded from a stored historical vol — not marked to a broker sheet. The surface is a shape, not a market.
- **VaR carries no correlation matrix.** Position VaRs are summed, so the book figure is an undiversified upper bound.
- **Supply/demand, regional flows and macro are modelled or static.** Scenario sandbox, not a balance.

---

MIT — © 2026 Adam El Gbouri · [github.com/adamelgbouri](https://github.com/adamelgbouri)
