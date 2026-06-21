"""Scenario engine - Bull / Base / Bear comparison."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from data.loaders import get_sd_dataset  # noqa: E402
from models.scenario import (  # noqa: E402
    probability_weighted_price,
    run_scenarios,
    scenario_summary,
)
from utils.config import COMMODITY_TEMPLATES, SCENARIO_PRESETS  # noqa: E402
from visuals.charts import scenario_paths  # noqa: E402
from visuals.theme import register_theme  # noqa: E402
from visuals.ui import apply_page_style, sidebar_controls  # noqa: E402


st.set_page_config(page_title="Scenarios", page_icon="🌪️", layout="wide")
register_theme()
apply_page_style()
state = sidebar_controls()
tpl = COMMODITY_TEMPLATES[state.commodity_key]
st.title(f"🌪️ {tpl.name} - Scenario Engine")

df = get_sd_dataset(state.commodity_key, start=state.history_start,
                    forecast_months=state.horizon_months)

st.markdown("**Scenario presets** (edit probabilities below; deltas in `utils/config.py`).")
c1, c2, c3 = st.columns(3)
new_probs = {}
for col, name in zip([c1, c2, c3], ["Bull", "Base", "Bear"]):
    with col:
        new_probs[name] = st.number_input(
            f"{name} probability",
            min_value=0.0, max_value=1.0,
            value=float(SCENARIO_PRESETS[name]["probability"]),
            step=0.05,
        )

# normalise probabilities
total = sum(new_probs.values()) or 1.0
for n in new_probs:
    SCENARIO_PRESETS[n]["probability"] = new_probs[n] / total

results = run_scenarios(df, state.commodity_key, st.session_state["assumptions"])

st.subheader("Inventory paths by scenario")
st.plotly_chart(scenario_paths(results, "stocks_model", "End Stocks"), width="stretch")

st.subheader("Price paths by scenario")
st.plotly_chart(scenario_paths(results, "fair_value_price", "Fair-Value Price"),
                width="stretch")

st.subheader("Summary")
summary = scenario_summary(results, state.commodity_key)
st.dataframe(summary.round(2), width="stretch")
st.metric("Probability-weighted forecast price",
          f"{probability_weighted_price(results):,.2f}")

with st.expander("Scenario assumption deltas"):
    st.json(SCENARIO_PRESETS)
