"""Regional balances + Sankey trade flow visualisation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from data.loaders import get_regional_dataset  # noqa: E402
from models.regional import (  # noqa: E402
    arbitrage_signals,
    build_trade_flows,
    regional_summary,
)
from utils.config import COMMODITY_TEMPLATES  # noqa: E402
from visuals.charts import regional_bar, sankey_chart  # noqa: E402
from visuals.theme import register_theme  # noqa: E402
from visuals.ui import apply_page_style, sidebar_controls  # noqa: E402


st.set_page_config(page_title="Regional Flows", page_icon="🌍", layout="wide")
register_theme()
apply_page_style()
state = sidebar_controls()
tpl = COMMODITY_TEMPLATES[state.commodity_key]
st.title(f"🌍 {tpl.name} - Regional Flows")

reg = get_regional_dataset(state.commodity_key)
rs = regional_summary(reg)
arb = arbitrage_signals(reg)

c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(regional_bar(reg), width="stretch")
with c2:
    st.dataframe(rs.round(2), hide_index=True, width="stretch")

st.subheader("Implied Trade Flows")
nodes, sources, targets, values = build_trade_flows(reg)
st.plotly_chart(sankey_chart(nodes, sources, targets, values), width="stretch")

st.subheader("Arbitrage Signals")
st.dataframe(arb[["region", "balance", "arb_signal"]].round(2),
             hide_index=True, width="stretch")
st.caption("Net exporter regions (Export Arb) supply net importers (Import Need); "
           "magnitudes drive the Sankey above.")
