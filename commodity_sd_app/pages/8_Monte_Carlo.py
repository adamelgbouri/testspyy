"""Monte Carlo / probabilistic engine."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from data.loaders import get_sd_dataset  # noqa: E402
from models.monte_carlo import MCConfig, run_monte_carlo, value_at_risk  # noqa: E402
from utils.config import COMMODITY_TEMPLATES  # noqa: E402
from visuals.charts import fan_chart, histogram  # noqa: E402
from visuals.theme import register_theme  # noqa: E402
from visuals.ui import apply_page_style, kpi_row, sidebar_controls  # noqa: E402


st.set_page_config(page_title="Monte Carlo", page_icon="🎲", layout="wide")
register_theme()
apply_page_style()
state = sidebar_controls()
tpl = COMMODITY_TEMPLATES[state.commodity_key]
st.title(f"🎲 {tpl.name} - Monte Carlo Engine")

c1, c2, c3 = st.columns(3)
n_paths = c1.slider("Paths", 100, 2000, 500, step=100)
sigma_supply = c2.slider("Supply σ %", 0.5, 5.0, 1.5, 0.1)
sigma_demand = c3.slider("Demand σ %", 0.5, 5.0, 1.2, 0.1)

c1, c2, c3 = st.columns(3)
sigma_weather = c1.slider("Weather σ %", 0.0, 3.0, 1.0, 0.1)
outage_prob = c2.slider("Outage prob/month", 0.0, 0.20, 0.05, 0.01)
outage_size = c3.slider("Outage size %", 0.0, 15.0, 4.0, 0.5)

cfg = MCConfig(
    n_paths=n_paths,
    supply_sigma_pct=sigma_supply,
    demand_sigma_pct=sigma_demand,
    weather_sigma_pct=sigma_weather,
    outage_prob=outage_prob,
    outage_size_pct=outage_size,
)

if st.button("Run Monte Carlo", type="primary"):
    with st.spinner(f"Simulating {n_paths} paths…"):
        df = get_sd_dataset(state.commodity_key, start=state.history_start,
                            forecast_months=state.horizon_months)
        out = run_monte_carlo(df, state.commodity_key,
                              st.session_state["assumptions"], cfg)

    end_stocks = out["end_stocks"]
    avg_price = out["avg_price"]
    bd = out["build_draw"]

    base_price = float(np.median(avg_price))
    losses = np.maximum(0, base_price - avg_price)
    var95 = value_at_risk(losses, 0.95)

    kpi_row([
        ("Median Price", f"{np.median(avg_price):,.2f}", None),
        ("P5 - P95 Price", f"{np.quantile(avg_price, 0.05):,.2f} - "
                            f"{np.quantile(avg_price, 0.95):,.2f}", None),
        ("Median End Stocks", f"{np.median(end_stocks):,.0f} {tpl.inventory_unit}", None),
        ("VaR 95% (price drop)", f"{var95:,.2f}", None),
    ])

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(histogram(avg_price, "Forecast Avg Price Distribution",
                                  x_label="Price"), width="stretch")
    with c2:
        st.plotly_chart(histogram(end_stocks, "End-of-Horizon Stocks Distribution",
                                  x_label=f"Stocks ({tpl.inventory_unit})"),
                        width="stretch")
    st.plotly_chart(histogram(bd, "Cumulative Build/Draw (FC) Distribution",
                              x_label="Δ Stocks"), width="stretch")

    st.subheader("Probabilistic Fan Charts")
    pct = out["percentiles"]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fan_chart(pct, "price"), width="stretch")
    with c2:
        st.plotly_chart(fan_chart(pct, "stocks"), width="stretch")
else:
    st.info("Configure parameters then click **Run Monte Carlo** to start.")
