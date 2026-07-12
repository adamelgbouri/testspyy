# S&D — Commodity Supply & Demand Trading Desk ™ by AEG

> A 12-page commodity trading terminal in a single Python file.
> 27 commodities across Energy, Metals, Agriculture and Freight — live futures data, real dated forward curves, options pricing, portfolio risk.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aeg-snd.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Live Demo

**[aeg-snd.streamlit.app](https://aeg-snd.streamlit.app/)**

The app is live and free to use. No installation required.
Source code available on request.

---

## What is S&D?

S&D is a standalone trading desk dashboard that puts **live market data, fundamentals, derivatives pricing and risk management** in one place. Pick a sector, pick a commodity, and navigate 12 dedicated pages — from a market heatmap to a Monte Carlo fan chart to a marked-to-market trade blotter.

Everything runs off a single Streamlit file. Live prices come from Yahoo Finance and are cached (5 min for spot, 1 h for history), so the app stays responsive without hammering the API. Commodities without a Yahoo feed (LME metals, carbon, coal, freight) fall back gracefully to modelled values, and every page tells you which source it's using with a `LIVE yfinance` / `FALLBACK` badge.

---

## The 12 Pages

| Page | What it does | Who uses it |
|---|---|---|
| 📊 **Dashboard** | Live spot, 1-day change, 2-year price history, fair-value proxy. Market treemap heatmap comparing **real closing prices between any two dates you choose** | Anyone starting their day |
| ⚖️ **Supply & Demand** | Supply/demand/stocks balance with surplus-deficit bars. Sliders to stress-test ±20% supply, ±20% demand, GDP growth | Fundamental analyst |
| 🌍 **Regional Flows** | World map of net exporters (green) vs net importers (red), bubble size = imbalance. Correct physical units per commodity (mb/d, bcf/d, Mbu/y…) | Physical trader |
| 📈 **Futures Curve** | **Real dated contract prices** (`CLN25.NYM`, `GCZ25.CMX`…) — not a modelled curve. Auto-classifies CONTANGO / BACKWARDATION / FLAT | Curve trader |
| 🎯 **Options & Greeks** | Black-76 pricer on the live spot. Full Greeks, payoff diagram, Greeks-vs-strike profiles, put-call parity check displayed live | Options desk |
| 📉 **Vol Surface** | Interactive 3D parametric surface. Adjustable ATM vol, skew, curvature, vol-of-vol. Smile by maturity + ATM term structure | Vol trader |
| 💼 **Positions & P&L** | Trade blotter marked to live prices. Gross long/short, net exposure, per-trade P&L and return, colour-coded | Trader / back office |
| 🛡️ **Risk** | Parametric VaR & CVaR at 90/95/99% over 1–30 days, per-position decomposition, ±5% to ±30% stress scenarios | Risk manager |
| 🎲 **Monte Carlo** | GBM simulation from the live spot. Fan chart (P5–P95), terminal distribution histogram, up to 2 000 paths | Structurer |
| 🌐 **Macro Overlay** | GDP, CPI, policy rate and PMI for 8 countries, side-by-side comparison | Macro strategist |
| 📅 **Events** | Market-moving calendar — EIA, WASDE, OPEC+, FOMC, IEA, CPI, LME Week | Everyone |
| ℹ️ **About** | Data-source status and project links | — |

---

## Key Features

- **Real dated futures contracts** — the curve page doesn't interpolate a model. It builds Yahoo tickers month by month (`CL{M}{YY}.NYM`), skips already-expired contracts (expiry ≈ 20th of the month before delivery), respects each commodity's **active delivery months** (Gold trades GJMQVZ, Sugar trades HKNV), and downloads them all in one batched call.
- **Date-to-date heatmap** — pick any two dates and the treemap fetches real closing prices for both, colouring every commodity by the actual move between them. Tells you how many contracts returned real data (`✅ 18/27`).
- **Honest data labelling** — spot prices, price history and curve points are tagged `real` or `model (cost-of-carry)` / `fallback`. Modelled and real curve points are even plotted with different markers on the same chart. Nothing is passed off as live when it isn't.
- **Live spot everywhere** — the option forward, the Monte Carlo starting point, the VaR notional and the blotter mark all default to the same cached live price. One source of truth across all 12 pages.
- **Graceful degradation** — no yfinance installed, no internet, Yahoo down? Every commodity has a calibrated fallback price and the app runs end to end regardless.
- **Correct units per commodity** — regional flows show barrels per day for crude, billion cubic feet per day for gas, million 60-kg bags for coffee, vessels for freight.

---

## Commodities Covered

| Family | Count | Commodities | Data |
|---|---|---|---|
| Energy | 8 | WTI, Brent, Natural Gas, RBOB, Heating Oil, Gasoil ICE, Carbon EUA, Coal API2 | Yahoo / fallback |
| Metals | 7 | Gold, Silver, Copper COMEX, Platinum, Palladium, LME Copper/Aluminum/Nickel | Yahoo / fallback |
| Agriculture | 10 | Corn, Wheat, Soybeans, Sugar #11, Coffee, Cocoa, Live Cattle, Lean Hogs | Yahoo Finance |
| Freight | 2 | Capesize BCI 5TC, Panamax BPI 4TC | Fallback |
| **Total** | **27** | | |

---

## Analytics Under the Hood

- **Black-76** — European options on futures, full Greeks (delta, gamma, vega, theta, rho), put-call parity verified on every computation
- **Cost-of-carry** — $F = S \cdot e^{(r + u - y)T}$, used only where real dated contracts aren't available
- **Parametric vol surface** — ATM level + skew + curvature + vol-of-vol in log-moneyness
- **GBM Monte Carlo** — up to 2 000 paths, monthly steps, percentile fan
- **Parametric VaR / CVaR** — $\text{VaR} = |N| \cdot \sigma_d \cdot z_\alpha \cdot \sqrt{h}$, aggregated across the book

---

## Dependencies

| Library | Role |
|---|---|
| `streamlit` | Dashboard framework |
| `yfinance` | Live futures prices, history, dated contracts |
| `pandas` | Curve and balance DataFrames |
| `numpy` | Monte Carlo, vol surface grids |
| `scipy` | `norm.cdf` for Black-76, `brentq` for implied vol |
| `plotly` | Charts, treemap heatmap, 3D surface, geo map |

---

## Run Locally

```bash
pip install streamlit plotly numpy pandas scipy yfinance
streamlit run commodity_trading_desk.py
```

---

## The AEG Platform Family

| | Purpose | Typical user |
|---|---|---|
| **S&D** (this app) | Trading desk terminal — fundamentals, curve, options, risk, P&L | Trader, desk analyst |
| [**CODAP**](https://aeg-codap.streamlit.app) | Six derivatives pricing engines — vanilla, Asian, crack, calendar, swaps, barrier | Options desk, structurer |
| [**CFCAP**](https://aeg-cfcap.streamlit.app) | Forward curve analytics — PCA, Schwartz-Smith, 51 trading signals | Curve trader, risk manager |
| [**Portfolio Optimizer**](https://aeg-markowitz.streamlit.app) | Markowitz optimization across 47 000+ instruments | Asset allocator |

S&D is where you *look at the market*. CFCAP is where you *analyse the curve*. CODAP is where you *price what's written on it*.

---

## License

MIT — © 2026 Adam El Gbouri

**Author:** Adam EL GBOURI · [github.com/adamelgbouri](https://github.com/adamelgbouri)

---

*Built with Python · Streamlit · Yahoo Finance · Plotly*
