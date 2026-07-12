# CSDAP — Commodity Supply & Demand Analytics Platform ™ by AEG

> A full trading desk built around the S&D balance. 10 commodities. Live market data.
> From inventory-driven fair value to VaR and Monte Carlo — one desk, eleven dashboards.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aeg-snd.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-Pydantic%20v2-teal?style=flat-square)
![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-Monte%20Carlo-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What is CSDAP?

CSDAP is a commodity trading desk platform that models the **physical supply & demand balance** of 10 major commodities and derives everything a desk needs from it: inventory paths, days-of-cover, an inventory-driven **fair value**, regional trade flows, futures curves, options with full Greeks, portfolio **VaR / CVaR**, stress tests and Monte Carlo scenario distributions.

Every dashboard is wired to the same balance engine, ensuring the fair value you see on the Dashboard is the same one moved by your assumptions on the Supply & Demand page and shocked by the Monte Carlo simulator.

The platform ships in two flavours sharing one analytics engine:

| Flavour | Stack | Entry point |
|---|---|---|
| **Streamlit app** | Single Python file | `streamlit_app.py` |
| **Web platform** | Next.js 14 + FastAPI | `web/frontend` + `web/backend` |

---

## Live Demo

**[aeg-snd.streamlit.app](https://aeg-snd.streamlit.app/)**

The app is live and free to use. No installation required.

---

## Desk Pages

| # | Page | What it does | Typical desk user |
|---|---|---|---|
| 1 | **Dashboard** | Live spot, fair-value deviation, market heatmap, auto trader brief | Everyone, first screen of the day |
| 2 | **Supply & Demand** | Editable balance assumptions → stocks, days-of-cover, fair value, baseline-vs-scenario overlay | Fundamentals analyst |
| 3 | **Regional Flows** | Production vs consumption by region, implied net trade, exporter/importer call-outs | Physical trader |
| 4 | **Futures Curve** | Live curve from Yahoo, contango / backwardation detection | Curve trader |
| 5 | **Options & Greeks** | Black-76 European pricer, full Greeks, implied vol back-solve, payoff visualiser | Options desk |
| 6 | **Positions & P&L** | Trade blotter with live mark-to-market, gross / net exposure | Book runner |
| 7 | **Risk** | Parametric VaR / CVaR per position, portfolio aggregation, stress grid | Risk manager |
| 8 | **Monte Carlo** | Stochastic supply / demand / weather shocks → price & stocks distributions, fan charts | Scenario analyst |
| 9 | **Macro Overlay** | GDP, CPI, policy rate, PMI panels for 8 economies | Macro strategist |
| 10 | **Events** | Auto-rolling 6-week calendar: EIA, WASDE, OPEC, IEA, FOMC, ECB, NFP | Everyone |
| 11 | **About** | Project overview | — |

---

## Features

- **One balance engine, every page** — the S&D identity (`stocks_t = stocks_{t-1} + supply_t − demand_t`, capped at storage capacity) drives fair value, risk and Monte Carlo alike. No page has its own private model.
- **Scenario vs baseline overlay** — moving a supply/demand slider redraws the forecast as a bright solid line next to the dashed zero-adjustment baseline, with a dedicated Δ-impact chart (monthly stocks delta bars + fair-value delta line).
- **Auto trader brief** — the Dashboard turns the raw numbers into a sentence a human would say: *"WTI is up +1.2% at 70.42 $/bbl, fair value 68.50 — rich (+2.8% vs spot), inventory 27.4d vs 30d target reads as balanced."*
- **Live data with deterministic fallback** — Yahoo Finance for spot and curves; when Yahoo is unreachable or rate-limited, a seeded synthetic generator takes over so the app never shows a blank chart.
- **Expiry-aware curve construction** — contract tickers are generated from each commodity's active month codes and skip expired deliveries, so the front month is always the true front month.
- **Session blotter** — positions persist in the session (Streamlit) or in localStorage (web version) and are marked-to-market on every refresh.
- **Sum-of-VaR portfolio risk** — conservative aggregation (ignores diversification), per-position decomposition and a six-scenario stress grid from −35% black swan to +25% squeeze.
- **Bloomberg-style dark UI** — trading-floor blue palette (`#0a1628`), amber accent, monospace numerals, both in Streamlit and in the Next.js version.

---

## Commodities Covered

| Family | Count | Commodities | Source |
|---|---|---|---|
| Energy | 5 | WTI, Brent, Natural Gas (Henry Hub), RBOB Gasoline, Heating Oil (ULSD) | Yahoo Finance |
| Precious | 2 | Gold, Silver | Yahoo Finance |
| Base Metals | 1 | Copper (COMEX) | Yahoo Finance |
| Agriculture | 2 | Wheat (CBOT), Corn (CBOT) | Yahoo Finance |
| **Total** | **10** | Each with its own seasonality profile, storage capacity, days-of-cover target and regional split | |

---

## Theory

**S&D balance identity** — monthly stocks recursion with a soft storage cap:

$$S_t = \operatorname{clip}\bigl(S_{t-1} + \text{supply}_t - \text{demand}_t,\; 0,\; 1.3 \times \text{capacity}\bigr)$$

Days of cover follow directly: $\text{DC}_t = S_t \,/\, \overline{\text{demand}}_t^{\,\text{daily}}$.

**Inventory-driven fair value** — commodity prices are a decreasing convex function of inventory cover. CSDAP fits a log-linear regression on history only:

$$\ln P_t = \alpha + \beta \cdot \text{DC}_t + \varepsilon_t \qquad\Rightarrow\qquad \hat{P}^{\,\text{fair}}_t = e^{\alpha + \beta\,\text{DC}_t}$$

Forecast assumptions (supply %, demand %, GDP, weather) change the stocks path, hence DC, hence fair value — while $\alpha, \beta$ stay frozen on actuals. A ±10% deviation of spot from fair value is flagged over/undervalued.

**Black-76** — the industry standard for European options on commodity futures:

$$C = e^{-rT}\bigl[F\cdot N(d_1) - K\cdot N(d_2)\bigr], \qquad d_1 = \frac{\ln(F/K)+\tfrac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}$$

Full Greeks (delta, gamma, vega, theta, rho) and Brent-root implied vol inversion.

**Parametric VaR / CVaR** — per position, from the return distribution of its price series:

$$\text{VaR}_\alpha = -\bigl(\mu - z_\alpha\,\sigma\bigr)\cdot \text{notional}, \qquad \text{CVaR}_\alpha = -\left(\mu - \sigma\,\frac{\varphi(z_\alpha)}{1-\alpha}\right)\cdot \text{notional}$$

scaled to the chosen horizon by $\sigma\sqrt{h}$. Portfolio VaR is the sum of individual VaRs — conservative by construction (no correlation offset).

**Monte Carlo on the balance** — instead of shocking price directly, CSDAP shocks the *fundamentals*: each path draws supply / demand / weather adjustments from normal distributions plus a random outage event, re-runs the full balance and re-prices via the fair-value map. Result: distributions of end stocks and average forecast price, with P5–P95 fan charts:

$$\text{path}_j:\quad \Delta s \sim \mathcal N(0,\sigma_s),\;\; \Delta d \sim \mathcal N(0,\sigma_d) \;\;\longrightarrow\;\; S_t^{(j)} \longrightarrow \text{DC}_t^{(j)} \longrightarrow \hat P_t^{(j)}$$

---

## Repository Layout

```
testspyy/
├── streamlit_app.py               ← single-file Streamlit app (this demo)
├── requirements.txt
├── DEPLOYMENT.md                  ← free hosting guide (Vercel + Render + Streamlit)
└── web/
    ├── backend/
    │   ├── main.py                ← FastAPI (REST API for the web frontend)
    │   └── commodity_engine/      ← shared analytics engine
    │       ├── balance.py         ← S&D identity, assumptions
    │       ├── fair_value.py      ← log-linear inventory regression
    │       ├── options.py         ← Black-76 + Greeks + implied vol
    │       ├── risk.py            ← VaR / CVaR / stress
    │       ├── monte_carlo.py     ← balance-shock simulation
    │       ├── spreads.py         ← crack spreads
    │       ├── macro.py           ← country macro panels
    │       ├── events.py          ← rolling desk calendar
    │       ├── data.py            ← Yahoo live + synthetic fallback
    │       └── config.py          ← 10 commodity templates
    └── frontend/                  ← Next.js 14 web version (11 pages)
```

Both flavours import the same `commodity_engine` — zero duplicated maths.

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py          # → http://localhost:8501
```

Web version:

```bash
# terminal 1 — API
cd web/backend && pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# terminal 2 — UI
cd web/frontend && npm install --legacy-peer-deps
npm run dev                             # → http://localhost:3000
```

---

## Dependencies

| Library | Role |
|---|---|
| `numpy` | Balance recursion, Monte Carlo, array operations |
| `pandas` | Monthly S&D frames, regional tables, cashflows |
| `scipy` | `norm` for Black-76 & VaR, `brentq` for implied vol |
| `scikit-learn` | Log-linear fair-value regression |
| `yfinance` | Live spot prices and futures curves |
| `streamlit` | Interactive browser dashboard |
| `plotly` | Zoomable charts, treemap heatmap, fan charts |
| `fastapi` / `pydantic` | REST API for the Next.js version |

---

## Relationship with CODAP & CFCAP

CSDAP, CODAP and CFCAP form one desk toolchain:

| | [CFCAP](https://github.com/adamelgbouri/commodity-forward-curve-analytics-platform) | CODAP | CSDAP |
|---|---|---|---|
| **Purpose** | Forward curve analytics | Derivatives pricing | Fundamentals, risk & P&L |
| **Models** | PCA, Schwartz-Smith, convenience yield | Black-76, Kirk, Asian MC, Barrier MC | S&D balance, fair-value regression, VaR/CVaR, balance MC |
| **Output** | 51 trading signals | Option prices, Greeks, swap NPV | Fair value, stocks paths, portfolio risk, MTM P&L |
| **Typical user** | Curve trader | Options desk, structurer | Fundamentals analyst, risk manager, book runner |

CFCAP reads the curve. CODAP prices what's written on it.
**CSDAP explains *why* the curve is where it is — and what your book is worth against it.**

---

## License

MIT — © 2026 Adam El Gbouri

---

*Built with Python · Streamlit · FastAPI · Next.js · Yahoo Finance*
