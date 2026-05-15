"""Futures curve / term-structure module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from data.loaders import get_futures_curve, get_sd_dataset  # noqa: E402
from models.balance import run_balance  # noqa: E402
from models.curve import (  # noqa: E402
    calendar_spreads,
    classify_structure,
    inventory_curve_relationship,
    storage_economics,
)
from utils.config import COMMODITY_TEMPLATES  # noqa: E402
from visuals.charts import futures_curve_chart  # noqa: E402
from visuals.theme import register_theme  # noqa: E402
from visuals.ui import apply_page_style, kpi_row, sidebar_controls  # noqa: E402


st.set_page_config(page_title="Futures Curve", page_icon="📈", layout="wide")
register_theme()
apply_page_style()
state = sidebar_controls()
tpl = COMMODITY_TEMPLATES[state.commodity_key]
st.title(f"📈 {tpl.name} - Term Structure")

c1, c2, c3 = st.columns(3)
structure_choice = c1.selectbox("Curve shape", ["contango", "backwardation", "flat"])
months = c2.slider("Tenors (months)", 6, 36, 18)
storage_cost = c3.number_input("Storage cost / month (per unit)", value=0.20, step=0.05)
financing_rate = st.slider("Financing rate %", 0.0, 12.0, 5.0, 0.25)

curve = get_futures_curve(state.commodity_key, structure=structure_choice, months=months)
struct_label = classify_structure(curve)
st.plotly_chart(futures_curve_chart(curve, struct_label), width="stretch")

c1, c2 = st.columns([2, 3])
with c1:
    spreads = calendar_spreads(curve)
    st.dataframe(spreads.round(3), width="stretch")
with c2:
    econ = storage_economics(curve, storage_cost, financing_rate)
    st.dataframe(econ[["tenor_month", "price", "contango_premium", "carry",
                       "positive_carry"]].round(3),
                 hide_index=True, width="stretch")

st.subheader("Inventory ↔ Curve Relationship")
df = get_sd_dataset(state.commodity_key, start=state.history_start,
                    forecast_months=state.horizon_months)
bal = run_balance(df, state.commodity_key, st.session_state["assumptions"])
last_dc = float(bal["days_cover_model"].iloc[-1])
label, score = inventory_curve_relationship(curve, last_dc)

kpi_row([
    ("Days Cover", f"{last_dc:.1f}", None),
    ("Tightness Score", f"{score:+.2f}", None),
    ("Curve Read", label, None),
])

with st.expander("Curve data"):
    st.dataframe(curve.round(3), hide_index=True, width="stretch")
