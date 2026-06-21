"""Inventory & storage analytics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from data.loaders import get_sd_dataset  # noqa: E402
from models.balance import run_balance  # noqa: E402
from models.inventory import (  # noqa: E402
    StorageConfig,
    days_of_forward_cover,
    draw_build_waterfall,
    project_inventory,
)
from utils.config import COMMODITY_TEMPLATES  # noqa: E402
from visuals.charts import (  # noqa: E402
    days_cover_chart,
    inventory_chart,
    utilization_gauge,
    waterfall_chart,
)
from visuals.theme import register_theme  # noqa: E402
from visuals.ui import apply_page_style, kpi_row, sidebar_controls  # noqa: E402


st.set_page_config(page_title="Inventories", page_icon="🛢️", layout="wide")
register_theme()
apply_page_style()
state = sidebar_controls()
tpl = COMMODITY_TEMPLATES[state.commodity_key]
st.title(f"🛢️ {tpl.name} - Inventory & Storage")

df = get_sd_dataset(state.commodity_key, start=state.history_start,
                    forecast_months=state.horizon_months)
bal = run_balance(df, state.commodity_key, st.session_state["assumptions"])

c1, c2, c3 = st.columns(3)
cap = c1.number_input("Storage capacity", value=float(tpl.storage_capacity),
                      step=10.0, min_value=10.0)
floating = c2.slider("Floating storage buffer (% of cap)", 0.0, 25.0, 5.0, 0.5)
allow_neg = c3.checkbox("Allow negative inventory (debug)", value=False)
inv = project_inventory(bal, state.commodity_key,
                        StorageConfig(capacity=cap, floating_buffer_pct=floating,
                                      allow_negative=allow_neg))

last = inv.iloc[-1]
kpi_row([
    ("Current Stocks", f"{last['stocks_capped']:,.0f} {tpl.inventory_unit}", None),
    ("Storage Util", f"{last['utilization_pct']:.1f}%", None),
    ("Floating", f"{last['overflow_floating']:,.0f}", None),
    ("Days of Cover", f"{last['days_cover_model']:.1f}", None),
])

st.plotly_chart(inventory_chart(inv, unit=tpl.inventory_unit), width="stretch")
c1, c2 = st.columns([2, 1])
with c1:
    st.plotly_chart(waterfall_chart(draw_build_waterfall(inv, n=12)), width="stretch")
with c2:
    st.plotly_chart(utilization_gauge(last["utilization_pct"]), width="stretch")

st.subheader("Forward Days of Cover")
st.plotly_chart(days_cover_chart(inv, target=tpl.days_cover_target), width="stretch")

st.subheader("Stocks vs Capacity")
st.area_chart(inv[["stocks_capped", "overflow_floating"]].tail(48), height=260)

st.dataframe(
    inv[["stocks_model", "stocks_capped", "overflow_floating", "utilization_pct",
         "days_cover_model"]].tail(24).round(2),
    width="stretch",
)
