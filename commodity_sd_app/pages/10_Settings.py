"""Settings - parameter save/load + cache controls + commodity templates."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from models.balance import BalanceAssumptions  # noqa: E402
from utils.config import COMMODITY_TEMPLATES  # noqa: E402
from utils.io_helpers import params_from_json, params_to_json  # noqa: E402
from visuals.theme import register_theme  # noqa: E402
from visuals.ui import apply_page_style, sidebar_controls  # noqa: E402


st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
register_theme()
apply_page_style()
state = sidebar_controls()

st.title("⚙️ Settings")

st.subheader("Active Assumptions")
a: BalanceAssumptions = st.session_state["assumptions"]
st.json(asdict(a))

st.subheader("Save / Load")
blob = params_to_json(asdict(a))
st.download_button("Download current parameters JSON",
                   blob.encode("utf-8"),
                   file_name="commodity_sd_params.json",
                   mime="application/json")
uploaded = st.file_uploader("Load parameters JSON", type=["json"])
if uploaded is not None:
    try:
        loaded = params_from_json(uploaded.read().decode("utf-8"))
        st.session_state["assumptions"] = BalanceAssumptions(**loaded)
        st.success("Parameters loaded - they apply across all pages.")
    except Exception as exc:
        st.error(f"Failed to load: {exc}")

st.subheader("Cache")
if st.button("Clear cache & rerun"):
    st.cache_data.clear()
    st.rerun()

st.subheader("Commodity Templates")
for k, tpl in COMMODITY_TEMPLATES.items():
    with st.expander(tpl.name):
        st.json({
            "key": tpl.key,
            "unit": tpl.unit,
            "inventory_unit": tpl.inventory_unit,
            "ticker": tpl.ticker,
            "base_supply": tpl.base_supply,
            "base_demand": tpl.base_demand,
            "base_price": tpl.base_price,
            "price_band": tpl.price_band,
            "storage_capacity": tpl.storage_capacity,
            "days_cover_target": tpl.days_cover_target,
            "regions": tpl.regions,
            "elasticity_alpha": tpl.elasticity_alpha,
            "elasticity_beta": tpl.elasticity_beta,
            "supply_lag_months": tpl.supply_lag_months,
        })
