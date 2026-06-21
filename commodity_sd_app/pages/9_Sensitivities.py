"""Sensitivity analysis - tornado + 2D stress matrix."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from data.loaders import get_sd_dataset  # noqa: E402
from models.sensitivity import SensitivityVar, stress_matrix, tornado  # noqa: E402
from utils.config import COMMODITY_TEMPLATES  # noqa: E402
from visuals.charts import seasonal_heatmap, tornado_chart  # noqa: E402
from visuals.theme import register_theme  # noqa: E402
from visuals.ui import apply_page_style, sidebar_controls  # noqa: E402


st.set_page_config(page_title="Sensitivities", page_icon="📉", layout="wide")
register_theme()
apply_page_style()
state = sidebar_controls()
tpl = COMMODITY_TEMPLATES[state.commodity_key]
st.title(f"📉 {tpl.name} - Sensitivity Analysis")

metric = st.selectbox("Metric", ["end_stocks", "avg_fc_price", "build_draw_sum"], index=0)

variables = [
    SensitivityVar("Supply Δ %", "supply_adj_pct", -3.0, 3.0),
    SensitivityVar("Demand Δ %", "demand_adj_pct", -3.0, 3.0),
    SensitivityVar("Weather Δ %", "weather_pct", -2.0, 2.0),
    SensitivityVar("GDP %", "gdp_growth_pct", 0.5, 4.0),
    SensitivityVar("Imports Δ %", "imports_adj_pct", -10.0, 10.0),
    SensitivityVar("Exports Δ %", "exports_adj_pct", -10.0, 10.0),
]

df = get_sd_dataset(state.commodity_key, start=state.history_start,
                    forecast_months=state.horizon_months)
torn = tornado(df, state.commodity_key, st.session_state["assumptions"], variables, metric=metric)
st.plotly_chart(tornado_chart(torn), width="stretch")
st.dataframe(torn.round(2), hide_index=True, width="stretch")

st.subheader("2D Stress Matrix")
c1, c2 = st.columns(2)
labels = [v.name for v in variables]
with c1:
    a = st.selectbox("Variable A", labels, index=0)
with c2:
    b = st.selectbox("Variable B", labels, index=1)
va = next(v for v in variables if v.name == a)
vb = next(v for v in variables if v.name == b)

mat = stress_matrix(df, state.commodity_key, st.session_state["assumptions"],
                    va, vb, grid=6, metric=metric)
st.plotly_chart(seasonal_heatmap(mat, title=f"{metric} for {a} × {b}"),
                width="stretch")
