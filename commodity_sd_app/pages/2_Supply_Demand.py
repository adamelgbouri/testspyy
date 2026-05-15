"""Supply & Demand page - balance engine + seasonality + elasticity views."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from data.loaders import get_sd_dataset, load_csv  # noqa: E402
from models.balance import run_balance  # noqa: E402
from models.elasticity import build_curves, ElasticityParams, equilibrium  # noqa: E402
from models.seasonality import (  # noqa: E402
    decompose,
    monthly_profile,
    rolling_seasonal_average,
    year_over_year_pivot,
)
from utils.config import COMMODITY_TEMPLATES  # noqa: E402
from utils.io_helpers import df_to_csv_bytes, df_to_excel_bytes  # noqa: E402
from visuals.charts import (  # noqa: E402
    balance_bars,
    elasticity_chart,
    seasonal_heatmap,
    seasonal_lines,
    supply_demand_chart,
)
from visuals.theme import register_theme  # noqa: E402
from visuals.ui import apply_page_style, kpi_row, sidebar_controls  # noqa: E402


st.set_page_config(page_title="Supply & Demand", page_icon="⚖️", layout="wide")
register_theme()
apply_page_style()
state = sidebar_controls()
tpl = COMMODITY_TEMPLATES[state.commodity_key]

st.title(f"⚖️ {tpl.name} - Balance Engine")

# Optional CSV import
with st.expander("Upload custom CSV (date, supply, demand, ...)"):
    file = st.file_uploader("CSV file", type=["csv"])
    df = load_csv(file) if file is not None else get_sd_dataset(
        state.commodity_key, start=state.history_start,
        forecast_months=state.horizon_months,
    )

freq = st.radio("Frequency", ["Monthly", "Quarterly", "Yearly"], horizontal=True)
freq_map = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
bal = run_balance(df, state.commodity_key, st.session_state["assumptions"],
                  frequency=freq_map[freq])

last = bal.iloc[-1]
kpi_row([
    ("End Stocks", f"{last['stocks_model']:,.0f} {tpl.inventory_unit}", None),
    ("Surplus / Deficit", f"{last['surplus_deficit']:+,.1f}", None),
    ("Days Cover", f"{last['days_cover_model']:.1f}", None),
    ("Storage Util", f"{last['capacity_pct']:.1f}%", None),
])

st.plotly_chart(supply_demand_chart(bal, unit=tpl.unit), width="stretch")
st.plotly_chart(balance_bars(bal), width="stretch")

st.markdown("### Seasonality")
profile = monthly_profile(df["demand"])
piv = year_over_year_pivot(df["demand"])
c1, c2 = st.columns([2, 3])
with c1:
    st.plotly_chart(seasonal_lines(profile), width="stretch")
with c2:
    st.plotly_chart(seasonal_heatmap(piv, title="Demand Heatmap (Year × Month)"),
                    width="stretch")

with st.expander("Seasonal Decomposition (additive)"):
    trend, seasonal, resid = decompose(df["demand"], period=12)
    st.line_chart(trend.dropna(), height=180)
    st.line_chart(seasonal.dropna(), height=180)
    st.line_chart(resid.dropna(), height=180)
    rolling = rolling_seasonal_average(df["demand"])
    st.line_chart(rolling.dropna(), height=180)

st.markdown("### Price Elasticity")
c1, c2 = st.columns(2)
with c1:
    alpha = st.slider("Demand elasticity α", 0.0, 0.5, tpl.elasticity_alpha, 0.005)
with c2:
    beta = st.slider("Supply elasticity β", 0.0, 0.5, tpl.elasticity_beta, 0.005)

curves = build_curves(state.commodity_key, alpha=alpha, beta=beta)
p = ElasticityParams(alpha=alpha, beta=beta, base_price=tpl.base_price,
                     d0=tpl.base_demand, s0=tpl.base_supply)
eq_p, eq_q = equilibrium(p)
st.plotly_chart(elasticity_chart(curves, eq_p, eq_q), width="stretch")
st.caption(f"Equilibrium price ≈ {eq_p:,.2f} | quantity ≈ {eq_q:,.2f} {tpl.unit}")

st.markdown("### Balance Table & Export")
display = bal.tail(36)
st.dataframe(display.round(2), width="stretch")
c1, c2 = st.columns(2)
c1.download_button("Download balance CSV",
                   df_to_csv_bytes(bal),
                   file_name=f"{state.commodity_key}_balance.csv",
                   mime="text/csv")
c2.download_button("Download Excel workbook",
                   df_to_excel_bytes({"balance": bal, "seasonal_profile": profile}),
                   file_name=f"{state.commodity_key}_workbook.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
