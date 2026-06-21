"""
Commodity S&D Analytics Platform - main entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo modules are importable when launched via `streamlit run app.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from data.loaders import (  # noqa: E402
    get_high_frequency,
    get_positioning,
    get_sd_dataset,
)
from models.balance import BalanceAssumptions, run_balance  # noqa: E402
from models.fair_value import estimate_fair_value  # noqa: E402
from utils.config import COMMODITY_TEMPLATES  # noqa: E402
from utils.logging_setup import get_logger  # noqa: E402
from visuals.charts import (  # noqa: E402
    days_cover_chart,
    fair_value_chart,
    inventory_chart,
    supply_demand_chart,
)
from visuals.theme import register_theme  # noqa: E402
from visuals.ui import apply_page_style, kpi_row, sidebar_controls  # noqa: E402


st.set_page_config(
    page_title="Commodity S&D Desk",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)
register_theme()
apply_page_style()

logger = get_logger("commodity_sd.app")
state = sidebar_controls()
tpl = COMMODITY_TEMPLATES[state.commodity_key]

st.title(f"🛢️ {tpl.name} - Supply & Demand Desk")
st.caption("Lightweight commodity analytics platform · synthetic data fallback enabled")

# Pull data + run balance
df = get_sd_dataset(state.commodity_key, start=state.history_start,
                    forecast_months=state.horizon_months)
bal = run_balance(df, state.commodity_key, st.session_state["assumptions"])
fv = estimate_fair_value(bal, state.commodity_key)

# Latest readings
last_hist = bal[~bal["is_forecast"]].iloc[-1]
last_fc = bal.iloc[-1]
spot = last_hist["price"]
fv_now = fv.loc[fv.index == last_hist.name, "fair_value_price"].iloc[0]
delta_fv_pct = (spot - fv_now) / fv_now * 100.0

kpi_row(
    [
        ("Spot Price", f"{spot:,.2f}", f"{(spot - bal['price'].iloc[-13]) / bal['price'].iloc[-13] * 100:+.1f}% YoY"),
        ("Fair Value", f"{fv_now:,.2f}", f"{delta_fv_pct:+.1f}% vs spot"),
        ("End Stocks (FC)", f"{last_fc['stocks_model']:,.0f} {tpl.inventory_unit}", None),
        ("Days of Cover", f"{last_fc['days_cover_model']:.1f}", f"target {tpl.days_cover_target:.0f}"),
        ("Storage Util", f"{last_fc['capacity_pct']:.1f}%", None),
    ]
)

st.markdown("---")
left, right = st.columns([3, 2])
with left:
    st.plotly_chart(supply_demand_chart(bal, unit=tpl.unit), width="stretch",
                    config={"displaylogo": False})
    st.plotly_chart(fair_value_chart(fv), width="stretch",
                    config={"displaylogo": False})
with right:
    st.plotly_chart(inventory_chart(bal, unit=tpl.inventory_unit), width="stretch",
                    config={"displaylogo": False})
    st.plotly_chart(days_cover_chart(bal, target=tpl.days_cover_target),
                    width="stretch", config={"displaylogo": False})

st.markdown("---")
with st.expander("Daily High-Frequency Snapshot"):
    hf = get_high_frequency(state.commodity_key)
    c1, c2, c3 = st.columns(3)
    c1.metric("Vessels Tracked", int(hf["vessels_tracked"].iloc[-1]),
              f"{hf['vessels_tracked'].iloc[-1] - hf['vessels_tracked'].iloc[-8]:+d} 7d")
    c2.metric("Refinery Util %", f"{hf['refinery_util_pct'].iloc[-1]:.1f}",
              f"{hf['refinery_util_pct'].iloc[-1] - hf['refinery_util_pct'].iloc[-8]:+.2f}")
    c3.metric("Sat Prod Est", f"{hf['sat_production_est'].iloc[-1]:,.1f}",
              f"{(hf['sat_production_est'].iloc[-1] / hf['sat_production_est'].iloc[-8] - 1) * 100:+.2f}%")

with st.expander("Speculative Positioning"):
    pos = get_positioning(state.commodity_key)
    c1, c2 = st.columns(2)
    c1.metric("Managed Money Net", f"{pos['managed_money_net'].iloc[-1]:,.0f}")
    c2.metric("Sentiment Score", f"{pos['sentiment_score'].iloc[-1]:.0f}/100")

st.caption(
    "Use the sidebar to switch commodities, change horizon, edit assumptions, "
    "and navigate to the dedicated analytics pages."
)
