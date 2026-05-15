# Commodity Supply & Demand Analytics Desk

A modular **Streamlit** application that emulates the analytical workflow used by
commodity desks at hedge funds and trading houses.  It builds Supply & Demand
balances, projects inventories, runs scenarios, estimates fair-value prices and
exposes a Monte-Carlo / sensitivity engine — all over a clean, dark
"trading-desk" UI.

The app ships with **synthetic data generators** for crude oil, natural gas,
copper and wheat so it runs offline.  Real data hooks (yfinance, optional
TradingView) are wired up but degrade gracefully.

---

## Features

| Page | What it does |
| ---- | ------------ |
| 🏠 Dashboard | Cross-module snapshot - price, balance, regional view, telemetry, positioning. |
| ⚖️ Supply & Demand | Balance engine (monthly/quarterly/yearly), seasonality decomposition, elasticity curves, CSV import & Excel export. |
| 🛢️ Inventories | Storage projection with capacity caps, floating storage spillover, days-of-cover, build/draw waterfall, utilisation gauge. |
| 🌪️ Scenarios | Bull / Base / Bear runs, probability-weighted price, scenario summary table. |
| 🌍 Regional Flows | Regional balances + Sankey trade flow diagram + arbitrage signal. |
| 📈 Futures Curve | Synthetic contango / backwardation curves, calendar spreads, storage economics, inventory↔curve relationship. |
| 🏦 Macro | Correlation heatmap, scatter regressions, rolling correlations vs price, multivariate regression. |
| 🎲 Monte Carlo | Probabilistic balance with random supply/demand/weather/outage shocks, fan charts, VaR. |
| 📉 Sensitivities | Tornado chart and 2D stress matrix on chosen metrics. |
| ⚙️ Settings | Save/load parameters JSON, clear cache, view commodity templates. |

---

## Folder Layout

```
commodity_sd_app/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── README.md
├── data/                   # synthetic generators, CSV loaders, sample CSVs
│   ├── synthetic.py
│   ├── loaders.py
│   ├── sample_crude_oil.csv
│   ├── sample_natural_gas.csv
│   ├── sample_copper.csv
│   └── sample_wheat.csv
├── models/                 # modeling layer
│   ├── balance.py          #   S&D balance engine
│   ├── seasonality.py      #   profiles + decomposition
│   ├── inventory.py        #   storage caps + waterfall
│   ├── elasticity.py       #   price elasticity curves
│   ├── lagged.py           #   distributed-lag supply response
│   ├── scenario.py         #   bull/base/bear engine
│   ├── regional.py         #   regional + sankey flows
│   ├── curve.py            #   futures / term structure
│   ├── macro.py            #   correlation / regression
│   ├── fair_value.py       #   fair-value regression + cost curve
│   ├── monte_carlo.py      #   probabilistic engine
│   ├── sensitivity.py      #   tornado + stress matrix
│   └── positioning.py      #   speculative positioning overlay
├── visuals/                # plotly chart builders + theme + reusable UI
│   ├── theme.py
│   ├── charts.py
│   └── ui.py
├── utils/                  # config, cache, logging, io
│   ├── config.py
│   ├── cache.py
│   ├── logging_setup.py
│   └── io_helpers.py
└── pages/                  # Streamlit multipage entries
    ├── 1_Dashboard.py
    ├── 2_Supply_Demand.py
    ├── 3_Inventories.py
    ├── 4_Scenarios.py
    ├── 5_Regional_Flows.py
    ├── 6_Futures_Curve.py
    ├── 7_Macro.py
    ├── 8_Monte_Carlo.py
    ├── 9_Sensitivities.py
    └── 10_Settings.py
```

---

## Run locally

```bash
# 1. Create + activate a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r commodity_sd_app/requirements.txt

# 3. Launch the app from inside the project directory
cd commodity_sd_app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Optional integrations

| Integration | How to enable |
| --- | --- |
| Yahoo Finance front-month price | Already wired via `yfinance`.  Set `COMMODITY_SD_DISABLE_YF=1` to force synthetic. |
| TradingView feeds | `pip install tvdatafeed` and extend `data/loaders.py` with a `get_tv_history` helper. |

---

## Adding a new commodity

1. Add a `CommodityTemplate` entry to `utils/config.py:COMMODITY_TEMPLATES`.
2. (Optional) Drop a `sample_<key>.csv` into `data/`.
3. The dropdown selector and every page picks it up automatically.

---

## Architecture notes

- **Data layer** — `data/synthetic.py` produces deterministic, reproducible
  monthly S&D series and other panels.  `data/loaders.py` adds caching and
  optional live-data hooks.
- **Modeling layer** — every model is a pure function of a DataFrame plus a
  small dataclass of parameters.  No model imports Streamlit, so they are
  trivially testable.
- **Visuals** — every chart builder returns a `plotly.graph_objects.Figure`.
  Pages just feed data to the builders, which keeps the UI thin.
- **Caching** — `utils/cache.py` wraps `st.cache_data` with a 10-minute TTL by
  default; the `Settings` page exposes a manual clear button.
- **Session state** — `BalanceAssumptions` lives in `st.session_state` so the
  sidebar sliders persist across pages.
- **Logging** — `utils/logging_setup.py` returns an idempotent logger;
  override level via `COMMODITY_SD_LOG=DEBUG`.

---

## Disclaimer

This is an educational analytics scaffold.  Synthetic series are intentionally
plausible but **not real market data**.  Do not trade on the outputs.
