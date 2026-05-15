"""Plotly + Streamlit theming - applied once at app start."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from utils.config import ACCENT, AMBER, DARK_BG, GREEN, GREY, PANEL_BG, RED


COLORS = {
    "supply": "#22c55e",
    "demand": "#ef4444",
    "stocks": "#00d4ff",
    "price": "#f59e0b",
    "fair_value": "#a78bfa",
    "bull": "#22c55e",
    "base": "#00d4ff",
    "bear": "#ef4444",
    "neutral": "#9ca3af",
}


def register_theme() -> None:
    """Register a trading-desk-style Plotly template."""
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(family="Inter, Helvetica Neue, Arial", color="#e5e7eb", size=12),
        xaxis=dict(
            gridcolor="#1f2937",
            zerolinecolor="#1f2937",
            linecolor="#374151",
            ticks="outside",
        ),
        yaxis=dict(
            gridcolor="#1f2937",
            zerolinecolor="#1f2937",
            linecolor="#374151",
            ticks="outside",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#374151",
            borderwidth=0,
            orientation="h",
            y=1.08,
            x=0,
        ),
        margin=dict(l=40, r=20, t=40, b=40),
        colorway=[ACCENT, GREEN, AMBER, "#a78bfa", RED, GREY, "#f97316", "#06b6d4"],
        hoverlabel=dict(bgcolor="#111827", font_size=11),
    )
    pio.templates["trading_desk"] = tpl
    pio.templates.default = "trading_desk"
