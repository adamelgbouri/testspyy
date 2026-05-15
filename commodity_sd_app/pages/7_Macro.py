"""Macro overlay - GDP / PMI / USD / rates and their commodity link."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from data.loaders import get_macro_panel, get_sd_dataset  # noqa: E402
from models.macro import (  # noqa: E402
    align_macro,
    correlation_matrix,
    regression_summary,
    rolling_correlation,
)
from utils.config import COMMODITY_TEMPLATES  # noqa: E402
from visuals.charts import (  # noqa: E402
    correlation_heatmap,
    rolling_corr_chart,
    scatter_with_fit,
)
from visuals.theme import register_theme  # noqa: E402
from visuals.ui import apply_page_style, sidebar_controls  # noqa: E402


st.set_page_config(page_title="Macro", page_icon="🏦", layout="wide")
register_theme()
apply_page_style()
state = sidebar_controls()
tpl = COMMODITY_TEMPLATES[state.commodity_key]
st.title(f"🏦 {tpl.name} - Macro Overlay")

sd = get_sd_dataset(state.commodity_key, start=state.history_start,
                    forecast_months=state.horizon_months)
macro = get_macro_panel(months=84)
joined = align_macro(sd, macro)

cols = ["price", "gdp_index", "pmi", "usd_index", "policy_rate"]
corr = correlation_matrix(joined, cols)
st.plotly_chart(correlation_heatmap(corr), width="stretch")

st.subheader("Scatter Diagnostics")
choices = ["gdp_index", "pmi", "usd_index", "policy_rate"]
c1, c2 = st.columns(2)
with c1:
    x1 = st.selectbox("Macro driver A", choices, index=0, key="ma")
    st.plotly_chart(scatter_with_fit(joined[x1], joined["price"], x1, "Price"),
                    width="stretch")
with c2:
    x2 = st.selectbox("Macro driver B", choices, index=2, key="mb")
    st.plotly_chart(scatter_with_fit(joined[x2], joined["price"], x2, "Price"),
                    width="stretch")

st.subheader("Rolling Correlations vs Price")
c1, c2 = st.columns(2)
with c1:
    rc1 = rolling_correlation(joined["price"], joined["gdp_index"])
    st.plotly_chart(rolling_corr_chart(rc1, "Price ~ GDP"), width="stretch")
with c2:
    rc2 = rolling_correlation(joined["price"], joined["usd_index"])
    st.plotly_chart(rolling_corr_chart(rc2, "Price ~ USD"), width="stretch")

st.subheader("Multivariate Regression: log(price) ~ macro")
import numpy as np  # local

joined["log_price"] = np.log(joined["price"])
try:
    res = regression_summary(joined, "log_price",
                             ["gdp_index", "pmi", "usd_index", "policy_rate"])
    st.json(res)
except ValueError as exc:
    st.warning(str(exc))
