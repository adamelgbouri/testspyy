"""
Reusable Plotly chart builders.

Every function takes raw data + optional styling kwargs and returns a
go.Figure ready to feed to st.plotly_chart.  Functions never mutate inputs.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visuals.theme import COLORS


# ---------------------------------------------------------------------------
# Balance / S&D charts
# ---------------------------------------------------------------------------
def supply_demand_chart(df: pd.DataFrame, unit: str = "") -> go.Figure:
    """Stacked area for supply & demand with stocks on a secondary axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["supply_adj"] if "supply_adj" in df else df["supply"],
            mode="lines", name="Supply",
            line=dict(color=COLORS["supply"], width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["demand_adj"] if "demand_adj" in df else df["demand"],
            mode="lines", name="Demand",
            line=dict(color=COLORS["demand"], width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["stocks_model"] if "stocks_model" in df else df["stocks"],
            mode="lines", name="Stocks",
            line=dict(color=COLORS["stocks"], width=1.5, dash="dot"),
        ),
        secondary_y=True,
    )
    if "is_forecast" in df.columns and df["is_forecast"].any():
        fc_start = df.index[df["is_forecast"]][0]
        fig.add_vline(x=fc_start, line=dict(color="#9ca3af", dash="dash", width=1))
        fig.add_annotation(x=fc_start, y=1, yref="paper", showarrow=False,
                           text=" Forecast →", font=dict(color="#9ca3af", size=11))

    fig.update_yaxes(title_text=f"Flow ({unit})", secondary_y=False)
    fig.update_yaxes(title_text="Stocks", secondary_y=True)
    fig.update_layout(title="Supply, Demand & Inventory", height=380)
    return fig


def inventory_chart(df: pd.DataFrame, unit: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["stocks_model"], mode="lines",
            name="Modeled Stocks", line=dict(color=COLORS["stocks"], width=2),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.15)",
        )
    )
    fig.update_layout(title="Inventory Trajectory", yaxis_title=f"Stocks ({unit})", height=320)
    return fig


def balance_bars(df: pd.DataFrame) -> go.Figure:
    """Build/draw bars; positive = build, negative = draw."""
    last = df.tail(24)
    colors = np.where(last["build_draw"] >= 0, COLORS["supply"], COLORS["demand"])
    fig = go.Figure(
        data=[
            go.Bar(x=last.index, y=last["build_draw"], marker_color=colors, name="Build/Draw")
        ]
    )
    fig.update_layout(title="Monthly Build / Draw (Last 24M)", yaxis_title="Δ stocks", height=320)
    return fig


def days_cover_chart(df: pd.DataFrame, target: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df["days_cover_model"], mode="lines",
                   name="Days of Cover", line=dict(color=COLORS["fair_value"], width=2))
    )
    if target is not None:
        fig.add_hline(y=target, line=dict(color=COLORS["price"], dash="dash"),
                      annotation_text=f"Target {target:.0f}d",
                      annotation_position="top right")
    fig.update_layout(title="Days of Forward Cover", yaxis_title="days", height=300)
    return fig


# ---------------------------------------------------------------------------
# Seasonality charts
# ---------------------------------------------------------------------------
def seasonal_lines(profile: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    fig.add_trace(
        go.Scatter(x=months, y=profile["mean"], name="5y Avg",
                   line=dict(color=COLORS["stocks"], width=2))
    )
    if "current" in profile:
        fig.add_trace(
            go.Scatter(x=months, y=profile["current"], name="Current Year",
                       line=dict(color=COLORS["price"], width=2, dash="dot"))
        )
    fig.add_trace(
        go.Scatter(x=months + months[::-1],
                   y=list(profile["max"]) + list(profile["min"][::-1]),
                   fill="toself", fillcolor="rgba(156,163,175,0.12)",
                   line=dict(color="rgba(0,0,0,0)"),
                   name="5y Range", showlegend=True)
    )
    fig.update_layout(title="Monthly Seasonal Profile", height=320)
    return fig


def seasonal_heatmap(pivot: pd.DataFrame, title: str = "Seasonal Heatmap") -> go.Figure:
    fig = px.imshow(
        pivot.values, x=pivot.columns, y=pivot.index, aspect="auto",
        color_continuous_scale="RdBu_r", origin="lower",
        labels=dict(x="Month", y="Year", color="Value"),
    )
    fig.update_layout(title=title, height=340)
    return fig


# ---------------------------------------------------------------------------
# Inventory waterfall + gauge
# ---------------------------------------------------------------------------
def waterfall_chart(deltas: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Waterfall(
            x=deltas["period"], y=deltas["delta"],
            measure=["relative"] * len(deltas),
            connector=dict(line=dict(color="#374151")),
            increasing=dict(marker=dict(color=COLORS["supply"])),
            decreasing=dict(marker=dict(color=COLORS["demand"])),
        )
    )
    fig.update_layout(title="Inventory Build / Draw Waterfall", height=320)
    return fig


def utilization_gauge(util_pct: float, target_pct: float = 80.0) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=util_pct,
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLORS["stocks"]},
                "steps": [
                    {"range": [0, 40], "color": "#1f2937"},
                    {"range": [40, 75], "color": "#0b3b48"},
                    {"range": [75, 100], "color": "#173b1f"},
                ],
                "threshold": {
                    "line": {"color": COLORS["price"], "width": 3},
                    "thickness": 0.75,
                    "value": target_pct,
                },
            },
            title={"text": "Storage Utilisation"},
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
    return fig


# ---------------------------------------------------------------------------
# Elasticity
# ---------------------------------------------------------------------------
def elasticity_chart(df: pd.DataFrame, eq_price: float, eq_qty: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["price"], y=df["demand"], name="Demand",
                   line=dict(color=COLORS["demand"], width=2))
    )
    fig.add_trace(
        go.Scatter(x=df["price"], y=df["supply"], name="Supply",
                   line=dict(color=COLORS["supply"], width=2))
    )
    fig.add_trace(
        go.Scatter(x=[eq_price], y=[eq_qty], name="Equilibrium",
                   mode="markers+text",
                   marker=dict(color=COLORS["price"], size=12, symbol="diamond"),
                   text=[f" ({eq_price:.1f}, {eq_qty:.1f})"], textposition="top right")
    )
    fig.update_layout(title="Price-Elasticity Curves",
                      xaxis_title="Price", yaxis_title="Quantity",
                      height=380)
    return fig


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def scenario_paths(results: dict, value_col: str = "stocks_model",
                   title: str = "Scenario Paths") -> go.Figure:
    fig = go.Figure()
    colors = {"Bull": COLORS["bull"], "Base": COLORS["base"], "Bear": COLORS["bear"]}
    for name, bal in results.items():
        fig.add_trace(
            go.Scatter(x=bal.index, y=bal[value_col], name=name,
                       line=dict(color=colors.get(name, COLORS["neutral"]), width=2))
        )
    fig.update_layout(title=title, height=380)
    return fig


def fan_chart(percentiles: pd.DataFrame, value: str = "price") -> go.Figure:
    p5 = percentiles[f"p5_{value}"]
    p50 = percentiles[f"p50_{value}"]
    p95 = percentiles[f"p95_{value}"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=p95.index.tolist() + p5.index[::-1].tolist(),
                   y=p95.tolist() + p5[::-1].tolist(),
                   fill="toself", fillcolor="rgba(0,212,255,0.18)",
                   line=dict(color="rgba(0,0,0,0)"),
                   name="P5-P95 band")
    )
    fig.add_trace(
        go.Scatter(x=p50.index, y=p50, name="Median",
                   line=dict(color=COLORS["stocks"], width=2))
    )
    fig.update_layout(title=f"Probabilistic {value.title()} Fan Chart", height=380)
    return fig


def tornado_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(y=df["variable"], x=df["low"], orientation="h",
               name="Low", marker_color=COLORS["bear"])
    )
    fig.add_trace(
        go.Bar(y=df["variable"], x=df["high"], orientation="h",
               name="High", marker_color=COLORS["bull"])
    )
    fig.update_layout(barmode="relative", title="Tornado Sensitivity",
                      xaxis_title="Δ vs Base", height=380)
    return fig


# ---------------------------------------------------------------------------
# Regional / Sankey
# ---------------------------------------------------------------------------
def regional_bar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Supply", x=df["region"], y=df["supply"], marker_color=COLORS["supply"])
    )
    fig.add_trace(
        go.Bar(name="Demand", x=df["region"], y=df["demand"], marker_color=COLORS["demand"])
    )
    fig.update_layout(barmode="group", title="Regional Supply vs Demand", height=320)
    return fig


def sankey_chart(nodes: List[str], sources: List[int], targets: List[int],
                 values: List[float]) -> go.Figure:
    if not nodes:
        fig = go.Figure()
        fig.update_layout(title="No trade flow data", height=320)
        return fig

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=14, thickness=20,
                line=dict(color="#374151", width=0.4),
                label=nodes,
                color=["#22c55e"] * (len(nodes) - sum(1 for s in sources)) + ["#ef4444"] * sum(1 for s in sources),
            ),
            link=dict(source=sources, target=targets, value=values,
                      color="rgba(0,212,255,0.35)"),
        )
    )
    fig.update_layout(title="Inter-Regional Trade Flows (Implied)", height=420)
    return fig


# ---------------------------------------------------------------------------
# Curve
# ---------------------------------------------------------------------------
def futures_curve_chart(curve: pd.DataFrame, structure: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=curve["tenor_month"], y=curve["price"], mode="lines+markers",
                   name=structure,
                   line=dict(color=COLORS["price"], width=2))
    )
    fig.update_layout(title=f"Futures Curve - {structure}",
                      xaxis_title="Contract Month",
                      yaxis_title="Price", height=320)
    return fig


def spread_heatmap(spreads: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        spreads.T.values, y=spreads.columns, x=spreads.index, color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    fig.update_layout(title="Spread Heatmap", height=260)
    return fig


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------
def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr.values, x=corr.columns, y=corr.index,
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    text_auto=True)
    fig.update_layout(title="Correlation Matrix", height=340)
    return fig


def scatter_with_fit(x: pd.Series, y: pd.Series, x_label: str, y_label: str) -> go.Figure:
    common = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
    fig = px.scatter(common, x="x", y="y", trendline="ols",
                     labels={"x": x_label, "y": y_label})
    fig.update_layout(title=f"{y_label} vs {x_label}", height=320)
    return fig


def rolling_corr_chart(s: pd.Series, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines",
                             line=dict(color=COLORS["stocks"], width=2), name="rolling corr"))
    fig.add_hline(y=0, line=dict(color="#9ca3af", dash="dot"))
    fig.update_layout(title=f"Rolling Correlation: {label}", yaxis_range=[-1, 1], height=300)
    return fig


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def histogram(arr: np.ndarray, title: str, x_label: str = "value") -> go.Figure:
    fig = go.Figure(
        go.Histogram(x=arr, nbinsx=40, marker_color=COLORS["stocks"],
                     marker_line=dict(color="#1f2937", width=0.4))
    )
    p5, p50, p95 = np.quantile(arr, [0.05, 0.5, 0.95])
    fig.add_vline(x=p5, line=dict(color=COLORS["bear"], dash="dot"), annotation_text="P5")
    fig.add_vline(x=p50, line=dict(color=COLORS["price"], dash="dash"), annotation_text="P50")
    fig.add_vline(x=p95, line=dict(color=COLORS["bull"], dash="dot"), annotation_text="P95")
    fig.update_layout(title=title, xaxis_title=x_label, height=320)
    return fig


# ---------------------------------------------------------------------------
# Fair value
# ---------------------------------------------------------------------------
def fair_value_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["price"], name="Observed Price",
                             line=dict(color=COLORS["price"], width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["fair_value_price"], name="Fair Value",
                             line=dict(color=COLORS["fair_value"], width=2, dash="dot")))
    upper = df["fair_value_price"] * 1.10
    lower = df["fair_value_price"] * 0.90
    fig.add_trace(go.Scatter(
        x=df.index.tolist() + df.index[::-1].tolist(),
        y=upper.tolist() + lower[::-1].tolist(),
        fill="toself", fillcolor="rgba(167,139,250,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="±10% Fair Band"
    ))
    fig.update_layout(title="Fair Value vs Observed Price", height=380)
    return fig


def cost_curve_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(x=df["cum_share_pct"], y=df["marginal_cost"], marker_color=COLORS["fair_value"])
    )
    fig.update_layout(title="Marginal Cost Curve",
                      xaxis_title="Cumulative Supply Share (%)",
                      yaxis_title="Marginal Cost",
                      height=320)
    return fig
