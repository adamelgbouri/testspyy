"""
Reusable Streamlit UI snippets - sidebar, KPI cards, and the standard
'commodity selector + horizon' control block used by every page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st

from models.balance import BalanceAssumptions
from utils.config import COMMODITY_TEMPLATES


_PAGE_LINKS = [
    ("pages/1_Dashboard.py", "🏠 Dashboard"),
    ("pages/2_Supply_Demand.py", "⚖️ Supply & Demand"),
    ("pages/3_Inventories.py", "🛢️ Inventories"),
    ("pages/4_Scenarios.py", "🌪️ Scenarios"),
    ("pages/5_Regional_Flows.py", "🌍 Regional Flows"),
    ("pages/6_Futures_Curve.py", "📈 Futures Curve"),
    ("pages/7_Macro.py", "🏦 Macro"),
    ("pages/8_Monte_Carlo.py", "🎲 Monte Carlo"),
    ("pages/9_Sensitivities.py", "📉 Sensitivities"),
    ("pages/10_Settings.py", "⚙️ Settings"),
]


def apply_page_style() -> None:
    """Inject minimal CSS to give the app a trading-desk aesthetic."""
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.5rem; padding-bottom: 1.5rem;}
            section[data-testid="stSidebar"] > div {background: #0b0f14;}
            div[data-testid="stMetric"] {
                background: #161b22;
                border: 1px solid #1f2937;
                border-radius: 8px;
                padding: 0.6rem 0.9rem;
            }
            div[data-testid="stMetricValue"] {color: #e5e7eb;}
            h1, h2, h3, h4 {color: #f3f4f6 !important;}
            .commodity-pill {
                display: inline-block;
                background: #1f2937;
                color: #00d4ff;
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 12px;
                margin-right: 4px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@dataclass
class GlobalState:
    commodity_key: str
    horizon_months: int
    history_start: str


def init_session_defaults() -> None:
    """Populate session_state with sensible defaults on cold start."""
    ss = st.session_state
    ss.setdefault("commodity_key", "crude_oil")
    ss.setdefault("horizon_months", 24)
    ss.setdefault("history_start", "2018-01-01")
    ss.setdefault("seed", 42)
    ss.setdefault("assumptions", BalanceAssumptions())


def sidebar_controls(show_assumptions: bool = True) -> GlobalState:
    """Render the standard sidebar (commodity + horizon + nav) on a page."""
    init_session_defaults()
    with st.sidebar:
        st.markdown("### 🛢️ Commodity S&D Desk")

        keys = list(COMMODITY_TEMPLATES.keys())
        names = [COMMODITY_TEMPLATES[k].name for k in keys]
        sel = st.selectbox(
            "Commodity",
            options=keys,
            format_func=lambda k: COMMODITY_TEMPLATES[k].name,
            index=keys.index(st.session_state["commodity_key"]),
        )
        st.session_state["commodity_key"] = sel

        st.session_state["horizon_months"] = st.slider(
            "Forecast horizon (months)", 6, 36, st.session_state["horizon_months"], step=3
        )
        st.session_state["history_start"] = st.text_input(
            "History start (YYYY-MM-DD)", st.session_state["history_start"]
        )
        st.divider()

        if show_assumptions:
            with st.expander("Assumptions", expanded=False):
                a: BalanceAssumptions = st.session_state["assumptions"]
                a.supply_adj_pct = st.slider("Supply Δ %", -10.0, 10.0, a.supply_adj_pct, 0.1)
                a.demand_adj_pct = st.slider("Demand Δ %", -10.0, 10.0, a.demand_adj_pct, 0.1)
                a.weather_pct = st.slider("Weather Δ %", -5.0, 5.0, a.weather_pct, 0.1)
                a.gdp_growth_pct = st.slider("GDP growth %", -2.0, 6.0, a.gdp_growth_pct or 2.5, 0.1)
                a.imports_adj_pct = st.slider("Imports Δ %", -20.0, 20.0, a.imports_adj_pct, 0.5)
                a.exports_adj_pct = st.slider("Exports Δ %", -20.0, 20.0, a.exports_adj_pct, 0.5)
                if st.session_state["commodity_key"] == "crude_oil":
                    a.refinery_runs_pct = st.slider("Refinery Runs Δ %", -10.0, 10.0,
                                                    a.refinery_runs_pct, 0.1)
                a.forecast_months = st.session_state["horizon_months"]
                st.session_state["assumptions"] = a

        st.divider()
        st.caption("Pages")
        for path, label in _PAGE_LINKS:
            try:
                st.page_link(path, label=label)
            except Exception:
                # st.page_link requires Streamlit ≥1.31; fail gracefully
                st.markdown(f"- {label}")

        st.divider()
        st.caption("Built with Streamlit · synthetic data when offline")

    return GlobalState(
        commodity_key=st.session_state["commodity_key"],
        horizon_months=st.session_state["horizon_months"],
        history_start=st.session_state["history_start"],
    )


def kpi_row(items: list[tuple[str, str, Optional[str]]]) -> None:
    """Render a row of KPI cards. Each item is (label, value_str, delta_str|None)."""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col:
            st.metric(label, value, delta=delta)
