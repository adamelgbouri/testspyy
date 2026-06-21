"""
Commodity Trading Desk - single-file Streamlit app.
by Adam EL GBOURI

Reuses the analytics engine in web/backend/commodity_engine/* so we don't
duplicate ~3000 lines of pricing / risk / fair-value logic.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deploy free on Streamlit Cloud:
    1. https://share.streamlit.io  →  Sign in with GitHub
    2. New app  →  pick this repo, branch `main`, file `streamlit_app.py`
    3. Deploy  →  done.  URL: https://<your-slug>.streamlit.app
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Plug in the analytics engine that already lives in web/backend
_ENGINE_PATH = Path(__file__).parent / "web" / "backend"
if str(_ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(_ENGINE_PATH))

from commodity_engine import (  # noqa: E402
    BalanceAssumptions, Black76, COMMODITY_TEMPLATES, MCConfig,
    estimate_fair_value, get_country_macro, get_futures_curve,
    get_live_spot, get_market_events, get_regional_dataset,
    get_sd_dataset, list_countries, parametric_var, portfolio_var,
    run_balance, run_monte_carlo, stress_scenarios,
)

# ----------------------------------------------------------------------------
# Page configuration & global styles
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Commodity Trading Desk - Adam EL GBOURI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0a1628; color: #dbe4f0; }
    section[data-testid="stSidebar"] { background-color: #0e1f33; }
    h1, h2, h3, h4 { color: #f0f4fa; letter-spacing: -0.01em; }
    .stMetric { background: linear-gradient(180deg,#0e1f33,#0a1628); padding: 14px;
                border-radius: 10px; border: 1px solid #1f3553; }
    [data-testid="stMetricLabel"] { color: #97a8be; font-size: 11px;
                                     letter-spacing: 0.1em; text-transform: uppercase; }
    [data-testid="stMetricValue"] { color: #f0f4fa; font-family: 'JetBrains Mono', monospace; }
    [data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace; }
    div[data-testid="stHorizontalBlock"] { gap: 12px; }
    .badge { display: inline-block; padding: 2px 8px; border: 1px solid #2c4564;
             border-radius: 6px; font-size: 10px; font-weight: 600;
             letter-spacing: 0.1em; text-transform: uppercase; color: #dbe4f0;
             background-color: #152641; font-family: 'JetBrains Mono', monospace; }
    .badge-accent { border-color: rgba(255,184,0,0.4); color: #ffb800; }
    .badge-pos { border-color: rgba(0,209,140,0.4); color: #00d18c; }
    .badge-neg { border-color: rgba(255,71,87,0.4); color: #ff4757; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { background-color: #152641; color: #dbe4f0;
                                    border-radius: 6px; padding: 6px 14px; }
    .stTabs [aria-selected="true"] { background-color: #ffb800 !important; color: #0a1628 !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
COLORS = {
    "supply": "#22c55e", "demand": "#ef4444", "stocks": "#00d4ff",
    "fair_value": "#ffb800", "price": "#f0f4fa", "accent": "#ffb800",
    "pos": "#00d18c", "neg": "#ff4757",
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def fmt_num(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{digits}f}"


def commodity_selector(key_prefix: str = "") -> str:
    """Sidebar commodity dropdown, returns the selected key."""
    labels = {k: f"[{t.sector}] {t.name}" for k, t in COMMODITY_TEMPLATES.items()}
    keys = list(labels.keys())
    default = st.session_state.get("commodity_key", "wti_crude")
    idx = keys.index(default) if default in keys else 0
    picked = st.sidebar.selectbox(
        "Commodity", options=keys, index=idx,
        format_func=lambda k: labels[k],
        key=f"{key_prefix}commodity_picker",
    )
    st.session_state["commodity_key"] = picked
    return picked


def _styled_chart(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10), height=height,
        font=dict(family="Inter, system-ui", size=12, color="#dbe4f0"),
        legend=dict(bgcolor="rgba(14,31,51,0.6)", bordercolor="#1f3553", borderwidth=1),
    )
    fig.update_xaxes(gridcolor="#152641", zerolinecolor="#1f3553")
    fig.update_yaxes(gridcolor="#152641", zerolinecolor="#1f3553")
    return fig


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
def render_header() -> None:
    col1, col2 = st.columns([5, 2])
    with col1:
        st.markdown(
            f"""
            <div style='display:flex; align-items:center; gap:12px; margin-bottom:8px;'>
                <div style='width:34px; height:34px; border-radius:8px;
                            background:linear-gradient(135deg,#ffb800,#00d4ff);
                            display:flex; align-items:center; justify-content:center;
                            color:#0a1628; font-weight:800; font-size:16px;'>C</div>
                <div>
                    <div style='font-size:18px; font-weight:700; color:#f0f4fa;'>
                        Commodity Trading Desk</div>
                    <div style='font-size:10px; color:#97a8be; letter-spacing:0.18em;
                                font-family:JetBrains Mono,monospace; text-transform:uppercase;'>
                        by Adam EL GBOURI · {date.today().year}</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div style='text-align:right; padding-top:8px; color:#97a8be; font-size:11px;"
            f"font-family:JetBrains Mono,monospace;'>"
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')} · session"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("---")


# ----------------------------------------------------------------------------
# Page: Dashboard
# ----------------------------------------------------------------------------
def page_dashboard() -> None:
    key = commodity_selector()
    tpl = COMMODITY_TEMPLATES[key]

    # Headline
    st.markdown(
        f"<span class='badge badge-accent'>LIVE</span> "
        f"<span class='badge'>{tpl.sector}</span> "
        f"<span class='badge'>{tpl.ticker}</span>",
        unsafe_allow_html=True,
    )
    st.title(tpl.name)

    df = get_sd_dataset(key, forecast_months=18)
    bal = run_balance(df, key, BalanceAssumptions(forecast_months=18))
    fv = estimate_fair_value(bal)
    last_hist = fv[~fv["is_forecast"]].iloc[-1]
    spot = get_live_spot(key)
    price_now = float(spot["price"]) if spot else float(last_hist["price"])
    change_pct = float(spot["change_pct"]) if spot else 0.0
    fv_now = float(last_hist["fair_value_price"])
    fv_dev = (price_now - fv_now) / fv_now * 100

    # Trader brief
    direction = "up" if change_pct > 0 else "down" if change_pct < 0 else "flat"
    fv_flag = ("rich" if fv_dev > 12 else "cheap" if fv_dev < -12 else "fairly priced")
    dc_now = float(last_hist["days_cover_model"])
    dc_flag = ("tight" if dc_now < tpl.days_cover_target * 0.85
               else "well-supplied" if dc_now > tpl.days_cover_target * 1.15
               else "balanced")
    st.info(
        f"**{tpl.name}** is {direction} **{change_pct:+.2f}%** at "
        f"`{price_now:,.2f} {tpl.price_unit}`. "
        f"Fair-value model puts it at `{fv_now:,.2f}` - **{fv_flag}** "
        f"({fv_dev:+.1f}% vs spot). Inventory at `{dc_now:.1f}d` cover vs "
        f"`{tpl.days_cover_target:.0f}d` target reads as **{dc_flag}**.",
        icon="🎯",
    )

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spot price", f"{price_now:,.2f} {tpl.price_unit}",
              f"{change_pct:+.2f}% 1D")
    c2.metric("Fair value", f"{fv_now:,.2f} {tpl.price_unit}",
              f"{fv_dev:+.1f}% vs spot")
    c3.metric(f"End stocks ({tpl.inventory_unit})",
              f"{float(last_hist['stocks_model']):,.0f}")
    c4.metric("Days of cover", f"{dc_now:.1f}",
              f"target {tpl.days_cover_target:.0f}d")
    c5.metric("Storage util %", f"{float(last_hist['capacity_pct']):.1f}%",
              f"ideal {tpl.ideal_utilization_pct:.0f}%")

    # Supply / Demand / Stocks chart
    st.subheader("Supply, demand & stocks")
    fig = go.Figure()
    forecast_start = fv[fv["is_forecast"]].index.min() if fv["is_forecast"].any() else None
    fig.add_trace(go.Scatter(x=fv.index, y=fv["supply"], name=f"Supply ({tpl.unit})",
                             line=dict(color=COLORS["supply"], width=2)))
    fig.add_trace(go.Scatter(x=fv.index, y=fv["demand"], name=f"Demand ({tpl.unit})",
                             line=dict(color=COLORS["demand"], width=2)))
    fig.add_trace(go.Scatter(x=fv.index, y=fv["stocks_model"],
                             name=f"Stocks ({tpl.inventory_unit})",
                             line=dict(color=COLORS["stocks"], width=1.5),
                             yaxis="y2",
                             fill="tozeroy", fillcolor="rgba(0,212,255,0.10)"))
    fig.update_layout(yaxis2=dict(title="Stocks", overlaying="y", side="right",
                                  showgrid=False, color=COLORS["stocks"]))
    if forecast_start is not None:
        fig.add_vline(x=forecast_start, line=dict(color="#6b7280", dash="dash"),
                      annotation_text="Forecast", annotation_position="top right")
    st.plotly_chart(_styled_chart(fig, 380), use_container_width=True)

    # Market heatmap
    st.subheader("Market heatmap")
    spots = []
    for k, t in COMMODITY_TEMPLATES.items():
        s = get_live_spot(k)
        if s:
            spots.append({"key": k, "name": t.name, "sector": t.sector,
                          "price": s["price"], "change_pct": s["change_pct"]})
        else:
            d = get_sd_dataset(k, forecast_months=3)
            spots.append({"key": k, "name": t.name, "sector": t.sector,
                          "price": float(d["price"].iloc[-1]), "change_pct": 0.0})
    sp_df = pd.DataFrame(spots)
    fig_hm = px.treemap(
        sp_df, path=[px.Constant("Markets"), "sector", "name"],
        values=[1] * len(sp_df), color="change_pct",
        color_continuous_scale=[(0, "#ff4757"), (0.5, "#152641"), (1, "#00d18c")],
        color_continuous_midpoint=0,
        custom_data=["price", "change_pct"],
    )
    fig_hm.update_traces(
        textinfo="label+text",
        texttemplate="<b>%{label}</b><br>%{customdata[0]:.2f}<br>%{customdata[1]:+.2f}%",
        hovertemplate="<b>%{label}</b><br>Price: %{customdata[0]:.2f}"
                      "<br>Change: %{customdata[1]:+.2f}%<extra></extra>",
    )
    st.plotly_chart(_styled_chart(fig_hm, 420), use_container_width=True)


# ----------------------------------------------------------------------------
# Page: Supply & Demand
# ----------------------------------------------------------------------------
def page_balance() -> None:
    key = commodity_selector("bal_")
    tpl = COMMODITY_TEMPLATES[key]
    st.title(f"Supply & Demand - {tpl.name}")
    st.caption("Editable assumptions feed the forecast portion of the balance.")

    c1, c2, c3, c4 = st.columns(4)
    sup_adj = c1.slider("Supply adj %", -20, 20, 0, 1, key="bal_sup")
    dem_adj = c2.slider("Demand adj %", -20, 20, 0, 1, key="bal_dem")
    gdp = c3.slider("GDP growth %", -2.0, 6.0, 2.5, 0.1, key="bal_gdp")
    horizon = c4.slider("Forecast months", 6, 36, 18, 3, key="bal_h")

    df = get_sd_dataset(key, forecast_months=horizon)
    bal = run_balance(df, key, BalanceAssumptions(
        supply_adj_pct=sup_adj, demand_adj_pct=dem_adj, gdp_growth_pct=gdp,
        forecast_months=horizon,
    ))
    fv = estimate_fair_value(bal)
    last_hist = fv[~fv["is_forecast"]].iloc[-1]
    last_fc = fv.iloc[-1] if fv["is_forecast"].any() else last_hist

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("End stocks (hist.)", f"{float(last_hist['stocks_model']):,.0f}")
    k2.metric("End stocks (forecast)", f"{float(last_fc['stocks_model']):,.0f}",
              f"{float(last_fc['stocks_model']) - float(last_hist['stocks_model']):+,.0f}")
    k3.metric("Fair value (forecast)", f"{float(last_fc['fair_value_price']):,.2f}")
    k4.metric("Avg surplus/deficit",
              f"{float(fv[fv['is_forecast']]['surplus_deficit'].mean()):+.2f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fv.index, y=fv["supply"], name="Supply",
                             line=dict(color=COLORS["supply"], width=2)))
    fig.add_trace(go.Scatter(x=fv.index, y=fv["demand"], name="Demand",
                             line=dict(color=COLORS["demand"], width=2)))
    fig.add_trace(go.Bar(x=fv.index, y=fv["surplus_deficit"], name="Surplus / Deficit",
                         marker_color=np.where(fv["surplus_deficit"] >= 0,
                                               COLORS["pos"], COLORS["neg"]),
                         opacity=0.6, yaxis="y2"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
    forecast_start = fv[fv["is_forecast"]].index.min() if fv["is_forecast"].any() else None
    if forecast_start is not None:
        fig.add_vline(x=forecast_start, line=dict(color="#6b7280", dash="dash"))
    st.plotly_chart(_styled_chart(fig), use_container_width=True)

    st.subheader("Fair-value model")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fv.index, y=fv["price"], name="Spot",
                              line=dict(color=COLORS["price"], width=2)))
    fig2.add_trace(go.Scatter(x=fv.index, y=fv["fair_value_price"], name="Fair value",
                              line=dict(color=COLORS["fair_value"], width=2, dash="dot")))
    st.plotly_chart(_styled_chart(fig2, 320), use_container_width=True)


# ----------------------------------------------------------------------------
# Page: Regional flows
# ----------------------------------------------------------------------------
def page_regional() -> None:
    key = commodity_selector("reg_")
    tpl = COMMODITY_TEMPLATES[key]
    st.title(f"Regional flows - {tpl.name}")

    reg = get_regional_dataset(key)
    reg = reg.assign(status=np.where(reg["net_trade"] > 0.5, "exporter",
                              np.where(reg["net_trade"] < -0.5, "importer", "balanced")))
    world_s = float(reg["supply"].sum())
    world_d = float(reg["demand"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"World supply ({tpl.unit})", f"{world_s:,.1f}")
    c2.metric(f"World demand ({tpl.unit})", f"{world_d:,.1f}")
    c3.metric("Balance", f"{world_s - world_d:+,.2f}",
              "surplus" if world_s > world_d else "deficit")
    c4.metric("Regions tracked", f"{len(reg)}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=reg["region"], y=reg["supply"], name="Supply",
                         marker_color=COLORS["supply"]))
    fig.add_trace(go.Bar(x=reg["region"], y=reg["demand"], name="Demand",
                         marker_color=COLORS["demand"]))
    fig.update_layout(barmode="group")
    st.plotly_chart(_styled_chart(fig), use_container_width=True)

    fig2 = go.Figure(go.Bar(
        x=reg["region"], y=reg["net_trade"],
        marker_color=np.where(reg["net_trade"] >= 0, COLORS["pos"], COLORS["neg"]),
        text=[f"{v:+,.1f}" for v in reg["net_trade"]], textposition="outside",
    ))
    fig2.update_layout(title="Net trade (supply − demand)")
    st.plotly_chart(_styled_chart(fig2, 320), use_container_width=True)

    st.dataframe(
        reg.style.format({"supply": "{:,.2f}", "demand": "{:,.2f}",
                          "net_trade": "{:+,.2f}",
                          "supply_share_pct": "{:.1f}%",
                          "demand_share_pct": "{:.1f}%"}),
        use_container_width=True,
    )


# ----------------------------------------------------------------------------
# Page: Futures curve
# ----------------------------------------------------------------------------
def page_curve() -> None:
    key = commodity_selector("curve_")
    tpl = COMMODITY_TEMPLATES[key]
    st.title(f"Futures curve - {tpl.name}")

    curve = get_futures_curve(key)
    if curve.empty:
        st.warning("No curve data available.")
        return
    spot = get_live_spot(key)
    spot_p = float(spot["price"]) if spot else float(curve["price"].iloc[0])

    front, back = float(curve["price"].iloc[0]), float(curve["price"].iloc[-1])
    structure = "contango" if back > front * 1.005 else "backwardation" if back < front * 0.995 else "flat"
    c1, c2, c3 = st.columns(3)
    c1.metric("Spot", f"{spot_p:,.2f} {tpl.price_unit}")
    c2.metric("Front month", f"{front:,.2f}")
    c3.metric("Curve structure", structure.upper(),
              f"{(back - front) / front * 100:+.2f}% front→back")

    fig = go.Figure(go.Scatter(
        x=curve["label"], y=curve["price"], mode="lines+markers",
        line=dict(color=COLORS["accent"], width=2),
        marker=dict(size=8, color=COLORS["accent"]),
    ))
    fig.add_hline(y=spot_p, line=dict(color=COLORS["price"], dash="dot"),
                  annotation_text="Spot", annotation_position="right")
    st.plotly_chart(_styled_chart(fig), use_container_width=True)
    st.dataframe(curve, use_container_width=True)


# ----------------------------------------------------------------------------
# Page: Options
# ----------------------------------------------------------------------------
def page_options() -> None:
    key = commodity_selector("opt_")
    tpl = COMMODITY_TEMPLATES[key]
    st.title(f"Options & Greeks - {tpl.name}")
    st.caption("Black-76 European pricer on futures.")

    spot = get_live_spot(key)
    F_default = float(spot["price"]) if spot else tpl.base_price

    c1, c2, c3, c4, c5 = st.columns(5)
    F = c1.number_input("Forward (F)", value=F_default, step=0.5)
    K = c2.number_input("Strike (K)", value=F_default, step=0.5)
    T = c3.slider("Maturity (years)", 0.05, 3.0, 0.5, 0.05)
    sigma = c4.slider("Volatility σ", 0.05, 1.5, 0.30, 0.01)
    r = c5.slider("Rate r", 0.0, 0.10, 0.045, 0.001)

    call = Black76(F=F, K=K, T=T, r=r, sigma=sigma, option_type="call")
    put = Black76(F=F, K=K, T=T, r=r, sigma=sigma, option_type="put")
    cg, pg = call.greeks(), put.greeks()

    st.subheader("Pricing & Greeks")
    cols = st.columns(6)
    labels = ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"]
    for col, lab, c_val, p_val in zip(cols, labels,
                                      cg.values(), pg.values()):
        col.metric(f"Call {lab}", f"{c_val:,.4f}")
    cols2 = st.columns(6)
    for col, lab, p_val in zip(cols2, labels, pg.values()):
        col.metric(f"Put {lab}", f"{p_val:,.4f}")

    # Payoff
    strikes = np.linspace(F * 0.6, F * 1.4, 60)
    call_payoff = np.maximum(strikes - K, 0) - call.price()
    put_payoff = np.maximum(K - strikes, 0) - put.price()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=call_payoff, name="Long call",
                             line=dict(color=COLORS["pos"], width=2)))
    fig.add_trace(go.Scatter(x=strikes, y=put_payoff, name="Long put",
                             line=dict(color=COLORS["neg"], width=2)))
    fig.add_hline(y=0, line=dict(color="#6b7280", dash="dash"))
    fig.add_vline(x=K, line=dict(color=COLORS["accent"], dash="dot"),
                  annotation_text="Strike", annotation_position="top right")
    fig.update_layout(title="Payoff at expiry (net of premium)")
    st.plotly_chart(_styled_chart(fig, 360), use_container_width=True)


# ----------------------------------------------------------------------------
# Page: Positions & P&L
# ----------------------------------------------------------------------------
def page_positions() -> None:
    st.title("Positions & P&L")
    st.caption("Trade blotter with live mark-to-market. Lives in your session only.")

    if "positions" not in st.session_state:
        st.session_state["positions"] = []

    with st.expander("➕ Add a position", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        labels = {k: t.name for k, t in COMMODITY_TEMPLATES.items()}
        keys = list(labels.keys())
        ck = c1.selectbox("Commodity", keys, format_func=lambda k: labels[k])
        side = c2.selectbox("Side", ["Long", "Short"])
        qty = c3.number_input("Quantity", value=100, step=10)
        entry = c4.number_input("Entry price",
                                value=float(COMMODITY_TEMPLATES[ck].base_price),
                                step=0.5)
        if c5.button("Add", use_container_width=True, type="primary"):
            st.session_state["positions"].append({
                "commodity_key": ck, "direction": side,
                "quantity": float(qty), "entry_price": float(entry),
            })
            st.success(f"Added {side} {qty} × {labels[ck]} @ {entry}")
            st.rerun()

    positions = st.session_state["positions"]
    if not positions:
        st.info("No positions yet. Add one above to start.")
        return

    rows, total_pnl, total_long, total_short = [], 0.0, 0.0, 0.0
    for i, p in enumerate(positions):
        tpl = COMMODITY_TEMPLATES[p["commodity_key"]]
        spot = get_live_spot(p["commodity_key"])
        mark = float(spot["price"]) if spot else float(p["entry_price"])
        sign = 1 if p["direction"] == "Long" else -1
        pnl_unit = sign * (mark - p["entry_price"])
        pnl_total = pnl_unit * p["quantity"]
        total_pnl += pnl_total
        if p["direction"] == "Long":
            total_long += mark * p["quantity"]
        else:
            total_short += mark * p["quantity"]
        rows.append({
            "Commodity": tpl.name, "Side": p["direction"],
            "Qty": p["quantity"], "Entry": p["entry_price"], "Mark": mark,
            "P&L/unit": pnl_unit, "P&L total": pnl_total,
            "Return %": (pnl_unit / p["entry_price"] * 100) if p["entry_price"] else 0.0,
        })

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross long ($)", f"{total_long:,.0f}")
    c2.metric("Gross short ($)", f"{total_short:,.0f}")
    c3.metric("Net exposure ($)", f"{total_long - total_short:+,.0f}")
    c4.metric("Total P&L ($)", f"{total_pnl:+,.0f}", f"{len(positions)} positions")

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({"Entry": "{:.2f}", "Mark": "{:.2f}", "P&L/unit": "{:+.3f}",
                         "P&L total": "{:+,.0f}", "Return %": "{:+.1f}%"})
                .apply(lambda x: [
                    f"color: {'#00d18c' if v >= 0 else '#ff4757'}" if isinstance(v, (int, float)) else ""
                    for v in x], subset=["P&L/unit", "P&L total", "Return %"]),
        use_container_width=True,
    )

    if st.button("🗑️ Clear all positions"):
        st.session_state["positions"] = []
        st.rerun()


# ----------------------------------------------------------------------------
# Page: Risk
# ----------------------------------------------------------------------------
def page_risk() -> None:
    st.title("Risk Dashboard")
    positions = st.session_state.get("positions", [])
    if not positions:
        st.warning("Add positions on the Positions page first.")
        return

    c1, c2 = st.columns(2)
    conf = c1.selectbox("Confidence", [0.90, 0.95, 0.99], index=1)
    horizon = c2.slider("Horizon (days)", 1, 30, 1)

    risk = portfolio_var(positions, confidence=conf, horizon_days=horizon)
    k1, k2, k3 = st.columns(3)
    k1.metric(f"VaR {int(conf*100)}%", f"${abs(risk['total_var']):,.0f}",
              f"{horizon}d horizon")
    k2.metric(f"CVaR {int(conf*100)}%", f"${abs(risk['total_cvar']):,.0f}",
              "Expected shortfall")
    k3.metric("Positions", f"{len(risk['rows'])}")

    st.subheader("Per-position decomposition")
    rdf = pd.DataFrame(risk["rows"])
    st.dataframe(
        rdf.style.format({"vol_pct": "{:.1f}%", "var": "${:,.0f}",
                          "cvar": "${:,.0f}", "quantity": "{:.0f}"}),
        use_container_width=True,
    )

    st.subheader("Stress scenarios (first position)")
    first = positions[0]
    tpl = COMMODITY_TEMPLATES[first["commodity_key"]]
    spot = get_live_spot(first["commodity_key"])
    base = float(spot["price"]) if spot else tpl.base_price
    scenarios = stress_scenarios(base, first["quantity"], first["direction"])
    sdf = pd.DataFrame(scenarios)
    st.dataframe(
        sdf.style.format({"shock_pct": "{:+.1f}%", "new_price": "{:.2f}",
                          "pnl_impact": "${:+,.0f}"}),
        use_container_width=True,
    )


# ----------------------------------------------------------------------------
# Page: Monte Carlo
# ----------------------------------------------------------------------------
def page_monte_carlo() -> None:
    key = commodity_selector("mc_")
    tpl = COMMODITY_TEMPLATES[key]
    st.title(f"Monte Carlo - {tpl.name}")
    st.caption("Stochastic balance shocks → distribution of forecast prices & stocks.")

    c1, c2, c3, c4 = st.columns(4)
    n_paths = c1.slider("Number of paths", 100, 2000, 500, 50)
    s_sig = c2.slider("Supply σ %", 0.5, 6.0, 1.5, 0.1)
    d_sig = c3.slider("Demand σ %", 0.5, 6.0, 1.2, 0.1)
    horizon = c4.slider("Forecast months", 6, 36, 18, 3)

    cfg = MCConfig(n_paths=n_paths, supply_sigma_pct=s_sig,
                   demand_sigma_pct=d_sig, forecast_months=horizon)
    with st.spinner(f"Running {n_paths} simulations…"):
        res = run_monte_carlo(key, cfg)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Median avg price",
              f"{res['median_price']:,.2f} {res['price_unit']}")
    k2.metric("P5 avg price", f"{res['p5_price_avg']:,.2f}")
    k3.metric("P95 avg price", f"{res['p95_price_avg']:,.2f}")
    k4.metric("Median end stocks",
              f"{res['median_end_stocks']:,.0f} {res['inventory_unit']}")

    fan = pd.DataFrame(res["fan_chart"])
    if not fan.empty:
        fan["date"] = pd.to_datetime(fan["date"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fan["date"], y=fan["p95"], name="P95",
                                 line=dict(color=COLORS["pos"], width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=fan["date"], y=fan["p5"], name="P5",
                                 line=dict(color=COLORS["neg"], width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(255,184,0,0.10)"))
        fig.add_trace(go.Scatter(x=fan["date"], y=fan["p50"], name="Median",
                                 line=dict(color=COLORS["accent"], width=2.5)))
        fig.update_layout(title="Fair-value price fan chart")
        st.plotly_chart(_styled_chart(fig), use_container_width=True)

    st.subheader("Distribution of avg forecast price")
    hist = pd.DataFrame(res["histogram_price"])
    fig_h = go.Figure(go.Bar(x=hist["x"], y=hist["count"],
                             marker_color=COLORS["accent"], opacity=0.75))
    st.plotly_chart(_styled_chart(fig_h, 280), use_container_width=True)


# ----------------------------------------------------------------------------
# Page: Macro
# ----------------------------------------------------------------------------
def page_macro() -> None:
    st.title("Macro overlay")
    countries = list_countries()
    c1, c2 = st.columns([2, 5])
    primary = c1.selectbox("Country", countries, index=0)
    defaults = [c for c in countries if c != primary][:2]
    compare = c2.multiselect("Compare with", [c for c in countries if c != primary],
                             default=defaults)

    series = [(primary, get_country_macro(primary))]
    series.extend((c, get_country_macro(c)) for c in compare)

    metrics = ["gdp_index", "cpi_yoy", "policy_rate", "pmi"]
    labels = ["GDP index", "CPI YoY %", "Policy rate %", "PMI"]

    snap = series[0][1].iloc[-1]
    cols = st.columns(4)
    for col, m, lab in zip(cols, metrics, labels):
        if m in snap.index:
            suffix = "%" if "%" in lab else ""
            col.metric(lab, f"{float(snap[m]):.2f}{suffix}")

    tabs = st.tabs(labels)
    for tab, m, lab in zip(tabs, metrics, labels):
        with tab:
            fig = go.Figure()
            for name, df in series:
                if m in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df[m], name=name, mode="lines"))
            st.plotly_chart(_styled_chart(fig, 360), use_container_width=True)


# ----------------------------------------------------------------------------
# Page: Events
# ----------------------------------------------------------------------------
def page_events() -> None:
    st.title("Market events calendar")
    st.caption("Auto-rolling 6-week window of EIA / WASDE / OPEC / IEA / central-bank releases.")
    events = get_market_events()
    if not events:
        st.info("No upcoming events.")
        return
    df = pd.DataFrame(events)
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.day_name()
    df["tags"] = df["tags"].apply(lambda t: ", ".join(t))

    today = pd.Timestamp(date.today())
    df["is_today"] = df["date"] == today

    def _row_style(r):
        if r["is_today"]:
            return ["background-color: #1f3553"] * len(r)
        return [""] * len(r)

    st.dataframe(
        df.drop(columns=["is_today"]).style.apply(_row_style, axis=1),
        use_container_width=True, hide_index=True,
    )


# ----------------------------------------------------------------------------
# Page: About
# ----------------------------------------------------------------------------
def page_about() -> None:
    st.title("About")
    st.markdown("""
    **Commodity Trading Desk** - solo-built portfolio project showcasing
    full-stack quantitative finance:

    - **Analytics** (Python · NumPy · SciPy · scikit-learn) - S&D balance,
      log-linear fair value, Black-76 option pricing & Greeks, parametric VaR
      / CVaR, Monte Carlo with fan charts, futures curve construction.
    - **Live data** - Yahoo Finance via `yfinance`, with deterministic
      synthetic fallback so the app never breaks when Yahoo is rate-limited.
    - **10 commodities tracked** - WTI, Brent, Henry Hub, RBOB, ULSD, Gold,
      Silver, Copper, Wheat, Corn.

    The same analytics engine also powers a Next.js + FastAPI version at
    `aeg-snd.vercel.app`.

    ---
    **Author:** Adam EL GBOURI
    [GitHub](https://github.com/adamelgbouri) ·
    [Source](https://github.com/adamelgbouri/testspyy)
    """)


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
PAGES = {
    "📊 Dashboard": page_dashboard,
    "⚖️ Supply & Demand": page_balance,
    "🌍 Regional flows": page_regional,
    "📈 Futures curve": page_curve,
    "🎯 Options & Greeks": page_options,
    "💼 Positions & P&L": page_positions,
    "🛡️ Risk": page_risk,
    "🎲 Monte Carlo": page_monte_carlo,
    "🌐 Macro overlay": page_macro,
    "📅 Events": page_events,
    "ℹ️ About": page_about,
}


def main() -> None:
    render_header()
    st.sidebar.markdown(
        "<div style='font-size:11px; color:#97a8be; letter-spacing:0.15em;"
        " text-transform:uppercase; font-family:JetBrains Mono,monospace;"
        " margin-bottom:8px;'>Navigation</div>",
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio("Pages", list(PAGES.keys()), label_visibility="collapsed")
    st.sidebar.markdown("---")
    PAGES[page]()
    st.sidebar.markdown(
        f"<div style='font-size:9px; color:#6c809b; font-family:JetBrains Mono,monospace;"
        f" letter-spacing:0.1em; text-transform:uppercase; margin-top:20px;'>"
        f"by Adam EL GBOURI · {date.today().year}<br>"
        f"FastAPI/Next.js version: aeg-snd.vercel.app</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
