"""Dashboard - high-level snapshot across all major modules."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from data.loaders import (  # noqa: E402
    get_high_frequency,
    get_positioning,
    get_regional_dataset,
    get_sd_dataset,
)
from models.balance import run_balance  # noqa: E402
from models.fair_value import estimate_fair_value  # noqa: E402
from models.regional import regional_summary  # noqa: E402
from utils.config import COMMODITY_TEMPLATES  # noqa: E402
from visuals.charts import (  # noqa: E402
    fair_value_chart,
    regional_bar,
    supply_demand_chart,
)
from visuals.theme import register_theme  # noqa: E402
from visuals.ui import apply_page_style, kpi_row, sidebar_controls  # noqa: E402


st.set_page_config(page_title="Dashboard", page_icon="🏠", layout="wide")
register_theme()
apply_page_style()
state = sidebar_controls()
tpl = COMMODITY_TEMPLATES[state.commodity_key]

st.title(f"🏠 {tpl.name} Dashboard")

df = get_sd_dataset(state.commodity_key, start=state.history_start,
                    forecast_months=state.horizon_months)
bal = run_balance(df, state.commodity_key, st.session_state["assumptions"])
fv = estimate_fair_value(bal, state.commodity_key)
last_h = bal[~bal["is_forecast"]].iloc[-1]
last_f = bal.iloc[-1]

kpi_row([
    ("Spot", f"{last_h['price']:,.2f}", None),
    ("Forecast Price (avg)", f"{bal.loc[bal['is_forecast'], 'price'].mean():,.2f}", None),
    ("End Stocks", f"{last_f['stocks_model']:,.0f} {tpl.inventory_unit}", None),
    ("Days Cover", f"{last_f['days_cover_model']:.1f}",
     f"target {tpl.days_cover_target:.0f}"),
])

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(supply_demand_chart(bal, unit=tpl.unit), width="stretch")
with c2:
    st.plotly_chart(fair_value_chart(fv), width="stretch")

st.subheader("Regional Snapshot")
reg = get_regional_dataset(state.commodity_key)
rs = regional_summary(reg)
left, right = st.columns([3, 2])
with left:
    st.plotly_chart(regional_bar(reg), width="stretch")
with right:
    st.dataframe(rs[["region", "supply", "demand", "balance", "status"]],
                 width="stretch", hide_index=True)

st.subheader("Daily Telemetry")
hf = get_high_frequency(state.commodity_key)
c1, c2, c3 = st.columns(3)
c1.line_chart(hf["vessels_tracked"].tail(60), height=180)
c2.line_chart(hf["refinery_util_pct"].tail(60), height=180)
c3.line_chart(hf["sat_production_est"].tail(60), height=180)

st.subheader("Speculative Positioning")
pos = get_positioning(state.commodity_key)
c1, c2 = st.columns(2)
c1.line_chart(pos["managed_money_net"].tail(52), height=200)
c2.line_chart(pos["sentiment_score"].tail(52), height=200)
