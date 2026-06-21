"""
Commodity Trading Desk — standalone single-file Streamlit app
by Adam EL GBOURI

No external backend required. All analytics (Black-76, Monte Carlo,
vol surface, S&D balance, VaR) are self-contained.

Run:
    pip install streamlit plotly numpy pandas scipy
    streamlit run commodity_trading_desk.py
"""
from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
from scipy.optimize import brentq

# ══════════════════════════════════════════════════════════════════════════════
#  COLOR PALETTE  (from tst3.py)
# ══════════════════════════════════════════════════════════════════════════════
AMBER  = "#F0A500"
BLUE   = "#58A6FF"
GREEN  = "#3FB950"
RED    = "#FF7B72"
GRAY   = "#8B949E"
PURPLE = "#BC8CFF"
TEAL   = "#39D0D8"
PANEL  = "#161B22"
BG     = "#0D1117"
BORDER = "#30363D"
TEXT   = "#E6EDF3"

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="S&D - Supply & Demand",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

.stApp {{ background-color:{BG}; color:{TEXT}; font-family:'Inter',system-ui; }}
section[data-testid="stSidebar"] {{ background-color:{PANEL}; border-right:1px solid {BORDER}; }}
h1,h2,h3,h4 {{ color:{TEXT}; font-family:'Inter',system-ui; letter-spacing:-0.02em; }}
[data-testid="stMetricLabel"] {{ color:{GRAY}; font-size:11px; font-family:'JetBrains Mono',monospace;
    letter-spacing:0.12em; text-transform:uppercase; }}
[data-testid="stMetricValue"] {{ color:{TEXT}; font-family:'JetBrains Mono',monospace; font-size:1.4rem; }}
[data-testid="stMetricDelta"] {{ font-family:'JetBrains Mono',monospace; font-size:0.75rem; }}
.stMetric {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; padding:14px 16px; }}
div[data-testid="stHorizontalBlock"] {{ gap:10px; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; background:{PANEL}; border-radius:8px; padding:4px; }}
.stTabs [data-baseweb="tab"] {{ background:transparent; color:{GRAY}; border-radius:6px; padding:6px 16px;
    font-family:'JetBrains Mono',monospace; font-size:0.78rem; letter-spacing:0.05em; }}
.stTabs [aria-selected="true"] {{ background:{AMBER} !important; color:{BG} !important; font-weight:700; }}
.badge {{ display:inline-block; padding:2px 8px; border:1px solid {BORDER}; border-radius:5px;
    font-size:10px; font-weight:600; letter-spacing:0.12em; text-transform:uppercase;
    color:{TEXT}; background:{PANEL}; font-family:'JetBrains Mono',monospace; margin-right:4px; }}
.badge-amber {{ border-color:rgba(240,165,0,0.5); color:{AMBER}; }}
.badge-green {{ border-color:rgba(63,185,80,0.5); color:{GREEN}; }}
.badge-red   {{ border-color:rgba(255,123,114,0.5); color:{RED}; }}
.kpi-card {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; padding:14px 16px;
    border-left:3px solid {AMBER}; }}
.kpi-label {{ font-size:10px; color:{GRAY}; font-family:'JetBrains Mono',monospace;
    text-transform:uppercase; letter-spacing:0.12em; margin-bottom:4px; }}
.kpi-value {{ font-size:1.3rem; color:{TEXT}; font-family:'JetBrains Mono',monospace; font-weight:600; }}
.kpi-sub   {{ font-size:0.68rem; color:{GRAY}; margin-top:3px; }}
hr {{ border-color:{BORDER}; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  COMMODITY REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
COMMODITIES: Dict[str, dict] = {
    # Energy
    "WTI Crude Oil":          dict(sector="Energy",       unit="$/bbl",       spot=67.50, vol=0.32, storage=0.096, conv=0.08,  ticker="CL"),
    "Brent Crude Oil":        dict(sector="Energy",       unit="$/bbl",       spot=71.20, vol=0.30, storage=0.096, conv=0.07,  ticker="BZ"),
    "Natural Gas":            dict(sector="Energy",       unit="$/MMBtu",     spot=3.15,  vol=0.55, storage=0.120, conv=0.10,  ticker="NG"),
    "RBOB Gasoline":          dict(sector="Energy",       unit="$/gallon",    spot=2.48,  vol=0.36, storage=0.084, conv=0.07,  ticker="RB"),
    "Heating Oil (ULSD)":     dict(sector="Energy",       unit="$/gallon",    spot=2.62,  vol=0.34, storage=0.084, conv=0.07,  ticker="HO"),
    "Gasoil ICE":             dict(sector="Energy",       unit="$/mt",        spot=770.0, vol=0.32, storage=0.072, conv=0.07,  ticker="GO"),
    "European Carbon (EUA)":  dict(sector="Energy",       unit="EUR/tCO2",    spot=63.0,  vol=0.35, storage=0.024, conv=0.02,  ticker="EUA"),
    "Coal API2":              dict(sector="Energy",       unit="$/mt",        spot=108.0, vol=0.30, storage=0.048, conv=0.04,  ticker="MTF"),
    # Metals
    "Gold":                   dict(sector="Metals",       unit="$/troy oz",   spot=3310., vol=0.15, storage=0.024, conv=0.005, ticker="GC"),
    "Silver":                 dict(sector="Metals",       unit="$/troy oz",   spot=32.8,  vol=0.28, storage=0.036, conv=0.010, ticker="SI"),
    "Copper (COMEX)":         dict(sector="Metals",       unit="$/lb",        spot=4.55,  vol=0.22, storage=0.048, conv=0.030, ticker="HG"),
    "Platinum":               dict(sector="Metals",       unit="$/troy oz",   spot=1010., vol=0.20, storage=0.030, conv=0.015, ticker="PL"),
    "Palladium":              dict(sector="Metals",       unit="$/troy oz",   spot=1090., vol=0.30, storage=0.030, conv=0.020, ticker="PA"),
    "LME Copper":             dict(sector="Metals",       unit="$/mt",        spot=9750., vol=0.22, storage=0.048, conv=0.030, ticker="LP"),
    "LME Aluminum":           dict(sector="Metals",       unit="$/mt",        spot=2390., vol=0.20, storage=0.048, conv=0.025, ticker="LA"),
    "LME Nickel":             dict(sector="Metals",       unit="$/mt",        spot=15800.,vol=0.30, storage=0.048, conv=0.035, ticker="LN"),
    # Agriculture
    "Corn":                   dict(sector="Agriculture",  unit="c/bushel",    spot=468.,  vol=0.25, storage=0.060, conv=0.04,  ticker="ZC"),
    "Wheat (CBOT)":           dict(sector="Agriculture",  unit="c/bushel",    spot=558.,  vol=0.28, storage=0.060, conv=0.04,  ticker="ZW"),
    "Soybeans":               dict(sector="Agriculture",  unit="c/bushel",    spot=1002., vol=0.23, storage=0.060, conv=0.05,  ticker="ZS"),
    "Sugar #11":              dict(sector="Agriculture",  unit="c/lb",        spot=18.9,  vol=0.30, storage=0.048, conv=0.04,  ticker="SB"),
    "Coffee (Arabica)":       dict(sector="Agriculture",  unit="c/lb",        spot=345.,  vol=0.35, storage=0.048, conv=0.05,  ticker="KC"),
    "Cocoa":                  dict(sector="Agriculture",  unit="$/mt",        spot=7850., vol=0.32, storage=0.048, conv=0.04,  ticker="CC"),
    "Live Cattle":            dict(sector="Agriculture",  unit="c/lb",        spot=183.,  vol=0.18, storage=0.036, conv=0.03,  ticker="LE"),
    "Lean Hogs":              dict(sector="Agriculture",  unit="c/lb",        spot=91.5,  vol=0.25, storage=0.036, conv=0.03,  ticker="HE"),
    # Freight
    "Capesize (BCI 5TC)":     dict(sector="Freight",      unit="$/day",       spot=17500.,vol=0.55, storage=0.0,   conv=0.00,  ticker="BCI"),
    "Panamax (BPI 4TC)":      dict(sector="Freight",      unit="$/day",       spot=11800.,vol=0.50, storage=0.0,   conv=0.00,  ticker="BPI"),
}

ALL_SECTORS = sorted({v["sector"] for v in COMMODITIES.values()})

REGIONAL_DATA = {
    "WTI Crude Oil": [
        dict(region="North America", supply=15.2, demand=10.8, lat=45, lon=-100),
        dict(region="Middle East",   supply=28.1, demand=8.2,  lat=25, lon=50),
        dict(region="Russia/FSU",    supply=13.5, demand=5.1,  lat=60, lon=70),
        dict(region="Europe",        supply=3.4,  demand=12.6, lat=52, lon=10),
        dict(region="Asia Pacific",  supply=8.1,  demand=34.5, lat=30, lon=115),
        dict(region="Africa",        supply=7.9,  demand=4.3,  lat=5,  lon=20),
        dict(region="Latin America", supply=5.8,  demand=6.2,  lat=-15,lon=-60),
    ],
    "Gold": [
        dict(region="China",        supply=370, demand=950,  lat=35, lon=105),
        dict(region="Australia",    supply=330, demand=30,   lat=-25,lon=133),
        dict(region="Russia",       supply=295, demand=90,   lat=60, lon=70),
        dict(region="Canada",       supply=190, demand=50,   lat=60, lon=-95),
        dict(region="USA",          supply=170, demand=230,  lat=38, lon=-97),
        dict(region="S. Africa",    supply=120, demand=45,   lat=-30,lon=25),
        dict(region="India",        supply=35,  demand=800,  lat=20, lon=80),
        dict(region="Europe",       supply=30,  demand=280,  lat=50, lon=15),
    ],
    "Corn": [
        dict(region="USA",          supply=387, demand=295, lat=38,  lon=-97),
        dict(region="China",        supply=277, demand=305, lat=35,  lon=105),
        dict(region="Brazil",       supply=137, demand=78,  lat=-15, lon=-55),
        dict(region="EU",           supply=62,  demand=71,  lat=50,  lon=15),
        dict(region="Argentina",    supply=55,  demand=16,  lat=-35, lon=-65),
        dict(region="Ukraine",      supply=27,  demand=12,  lat=49,  lon=32),
        dict(region="Mexico",       supply=28,  demand=45,  lat=24,  lon=-102),
        dict(region="South Africa", supply=16,  demand=14,  lat=-29, lon=25),
    ],
}

EVENTS = [
    dict(date=date.today()+timedelta(days=3),  event="EIA Weekly Petroleum Status Report", tags=["Energy","Oil"]),
    dict(date=date.today()+timedelta(days=5),  event="WASDE Monthly Supply & Demand",      tags=["Agriculture"]),
    dict(date=date.today()+timedelta(days=7),  event="OPEC+ Ministerial Meeting",           tags=["Energy","OPEC"]),
    dict(date=date.today()+timedelta(days=10), event="FOMC Interest Rate Decision",         tags=["Macro","Rates"]),
    dict(date=date.today()+timedelta(days=12), event="IEA Oil Market Report",               tags=["Energy"]),
    dict(date=date.today()+timedelta(days=14), event="US CPI Data Release",                 tags=["Macro","Inflation"]),
    dict(date=date.today()+timedelta(days=18), event="USDA Crop Progress Report",           tags=["Agriculture"]),
    dict(date=date.today()+timedelta(days=21), event="LME Week – London Metal Exchange",    tags=["Metals"]),
    dict(date=date.today()+timedelta(days=25), event="ECB Monetary Policy Decision",        tags=["Macro","Rates"]),
    dict(date=date.today()+timedelta(days=28), event="Baker Hughes Rig Count",              tags=["Energy"]),
    dict(date=date.today()+timedelta(days=32), event="OPEC Monthly Oil Market Report",      tags=["Energy","OPEC"]),
]

COUNTRIES = ["USA", "China", "Germany", "Japan", "UK", "Brazil", "India", "France"]

# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def black76(F: float, K: float, T: float, r: float, sigma: float,
            option_type: str = "call") -> dict:
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return dict(price=0, delta=0, gamma=0, vega=0, theta=0, rho=0)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    if option_type == "call":
        price = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
        delta = disc * norm.cdf(d1)
    else:
        price = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        delta = -disc * norm.cdf(-d1)
    gamma = disc * norm.pdf(d1) / (F * sigma * math.sqrt(T))
    vega  = disc * F * norm.pdf(d1) * math.sqrt(T) / 100
    theta = (-(disc * F * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * price) / 365
    rho   = -T * price / 100
    return dict(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_vol(F: float, K: float, T: float, r: float,
                market_price: float, option_type: str = "call") -> Optional[float]:
    try:
        f = lambda s: black76(F, K, T, r, s, option_type)["price"] - market_price
        return brentq(f, 1e-4, 10.0, xtol=1e-6)
    except Exception:
        return None


def forward_curve(spot: float, r: float, storage: float, conv: float,
                  vol: float, months: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    today = date.today()
    for i in range(1, months + 1):
        T = i / 12
        F = spot * math.exp((r + storage - conv) * T)
        noise = rng.normal(0, vol * math.sqrt(T / 12) * 0.3) * spot * 0.01
        F += noise
        label = (today.replace(day=1) + timedelta(days=32 * i)).strftime("%b %y")
        rows.append(dict(month=i, label=label, T=T, price=F))
    return pd.DataFrame(rows)


def vol_surface(F: float, atm_vol: float, skew: float = -0.05,
                curv: float = 0.02, vov: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    maturities = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])
    moneyness  = np.linspace(-0.40, 0.40, 25)
    Kgrid = F * np.exp(moneyness)
    Z = np.zeros((len(maturities), len(Kgrid)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(Kgrid):
            x = math.log(K / F)
            v = atm_vol * (1 + vov * math.sqrt(T)) + skew * x + curv * x**2
            Z[i, j] = max(v, 0.01)
    return maturities, Kgrid, Z


def sd_dataset(commodity: str, months: int = 36) -> pd.DataFrame:
    rng = np.random.default_rng(hash(commodity) % 2**32)
    c   = COMMODITIES[commodity]
    spot = c["spot"]
    today = date.today().replace(day=1)
    dates = [today - timedelta(days=30*(months - i)) for i in range(months)]
    hist  = int(months * 0.65)
    sup = spot * (1 + rng.normal(0, 0.02, months)).cumprod()
    dem = spot * (1 + rng.normal(0, 0.018, months)).cumprod()
    sto = np.zeros(months)
    sto[0] = spot * 0.15
    for i in range(1, months):
        sto[i] = max(sto[i-1] + (sup[i] - dem[i]) * 0.05, spot * 0.03)
    prices = [spot * math.exp(rng.normal(-0.002, c["vol"]/12, 1)[0] * (j+1)) for j in range(months)]
    fv     = [spot * (sto[j] / sto[0]) ** (-0.3) * (dem[j] / dem[0]) ** 0.5 for j in range(months)]
    is_fc  = [j >= hist for j in range(months)]
    return pd.DataFrame(dict(date=dates, supply=sup, demand=dem,
                             stocks=sto, price=prices, fair_value=fv,
                             surplus=sup-dem, is_forecast=is_fc)).set_index("date")


def run_mc(commodity: str, n_paths: int = 500, horizon: int = 18,
           sup_sig: float = 1.5, dem_sig: float = 1.2) -> dict:
    rng = np.random.default_rng(0)
    c   = COMMODITIES[commodity]
    S0  = c["spot"]
    mu  = 0.0
    sigma = c["vol"]
    dt  = 1 / 12
    paths = np.zeros((n_paths, horizon + 1))
    paths[:, 0] = S0
    for t in range(1, horizon + 1):
        z = rng.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*math.sqrt(dt)*z)
    today = date.today()
    fan_dates = [today + timedelta(days=30*i) for i in range(horizon+1)]
    pcts = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fan = pd.DataFrame(dict(date=fan_dates, p5=pcts[0], p25=pcts[1],
                            p50=pcts[2], p75=pcts[3], p95=pcts[4]))
    hist_prices = paths[:, 1:].mean(axis=1)
    hist_bins   = np.histogram(hist_prices, bins=40)
    return dict(paths=paths[:10], fan=fan,
                median=float(np.median(paths[:,-1])),
                p5=float(np.percentile(paths[:,-1],5)),
                p95=float(np.percentile(paths[:,-1],95)),
                hist_x=hist_bins[1][:-1].tolist(),
                hist_y=hist_bins[0].tolist(),
                unit=c["unit"])


def portfolio_var(positions: list, conf: float = 0.95, horizon: int = 1) -> dict:
    z = norm.ppf(conf)
    rows, total_var, total_cvar = [], 0.0, 0.0
    for p in positions:
        c   = COMMODITIES.get(p["commodity"], {})
        vol = c.get("vol", 0.30)
        spot = c.get("spot", p.get("entry_price", 100))
        sign = 1 if p.get("side") == "Long" else -1
        notional = spot * p.get("quantity", 0) * sign
        daily_vol = vol / math.sqrt(252)
        var  = abs(notional) * daily_vol * z * math.sqrt(horizon)
        cvar = abs(notional) * daily_vol * norm.pdf(z) / (1 - conf) * math.sqrt(horizon)
        total_var  += var
        total_cvar += cvar
        rows.append(dict(commodity=p["commodity"], side=p.get("side"),
                         quantity=p.get("quantity"), spot=spot,
                         vol_pct=vol*100, var=var, cvar=cvar))
    return dict(total_var=total_var, total_cvar=total_cvar, rows=rows)


def macro_data(country: str, months: int = 48) -> pd.DataFrame:
    rng   = np.random.default_rng(hash(country) % 2**32)
    today = date.today().replace(day=1)
    dates = [today - timedelta(days=30*(months-i)) for i in range(months)]
    base  = dict(USA=dict(gdp=100, cpi=3.2, rate=5.25, pmi=52.1),
                 China=dict(gdp=100, cpi=0.8, rate=3.45, pmi=50.4),
                 Germany=dict(gdp=100, cpi=2.9, rate=3.50, pmi=47.2),
                 Japan=dict(gdp=100, cpi=2.5, rate=0.50, pmi=49.8),
                 UK=dict(gdp=100, cpi=3.4, rate=5.25, pmi=51.3),
                 Brazil=dict(gdp=100, cpi=4.8, rate=10.75, pmi=50.9),
                 India=dict(gdp=100, cpi=4.5, rate=6.50, pmi=57.5),
                 France=dict(gdp=100, cpi=2.7, rate=3.50, pmi=48.6))
    b = base.get(country, dict(gdp=100, cpi=2.5, rate=4.0, pmi=50.0))
    gdp  = b["gdp"] + np.cumsum(rng.normal(0.5, 0.3, months))
    cpi  = b["cpi"] + np.cumsum(rng.normal(0.0, 0.08, months))
    rate = b["rate"] + np.cumsum(rng.normal(0.0, 0.05, months))
    pmi  = b["pmi"] + rng.normal(0, 1.2, months)
    return pd.DataFrame(dict(date=dates, gdp_index=gdp,
                             cpi_yoy=np.clip(cpi, -2, 20),
                             policy_rate=np.clip(rate, 0, 20),
                             pmi=np.clip(pmi, 30, 70))).set_index("date")

# ══════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _styled(fig: go.Figure, h: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10), height=h,
        font=dict(family="Inter,system-ui", size=12, color=TEXT),
        legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor=BORDER, borderwidth=1),
    )
    fig.update_xaxes(gridcolor=PANEL, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=PANEL, zerolinecolor=BORDER)
    return fig


def kpi(label: str, value: str, sub: str = "", accent: str = AMBER) -> str:
    return (
        f'<div class="kpi-card" style="border-left-color:{accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar() -> str:
    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;padding:4px 0 16px;">'
        f'<div style="width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,{AMBER},{TEAL});'
        f'display:flex;align-items:center;justify-content:center;color:{BG};font-weight:800;font-size:15px;">C</div>'
        f'<div style="font-size:14px;font-weight:700;color:{TEXT};">Trading Desk</div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'<div style="font-size:10px;color:{GRAY};font-family:JetBrains Mono,monospace;'
        f'letter-spacing:0.15em;text-transform:uppercase;margin-bottom:6px;">Navigation</div>',
        unsafe_allow_html=True,
    )
    pages = [
        "📊 Dashboard", "⚖️ Supply & Demand", "🌍 Regional Flows",
        "📈 Futures Curve", "🎯 Options & Greeks", "📉 Vol Surface",
        "💼 Positions & P&L", "🛡️ Risk", "🎲 Monte Carlo",
        "🌐 Macro Overlay", "📅 Events", "ℹ️ About",
    ]
    page = st.sidebar.radio("Pages", pages, label_visibility="collapsed")
    st.sidebar.markdown("---")

    st.sidebar.markdown(
        f'<div style="font-size:10px;color:{GRAY};font-family:JetBrains Mono,monospace;'
        f'letter-spacing:0.15em;text-transform:uppercase;margin-bottom:6px;">Select Commodity</div>',
        unsafe_allow_html=True,
    )
    sector = st.sidebar.selectbox("Sector", ALL_SECTORS, key="sidebar_sector")
    names_in_sector = [k for k, v in COMMODITIES.items() if v["sector"] == sector]
    commodity = st.sidebar.selectbox("Commodity", names_in_sector, key="sidebar_commodity")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div style="font-size:10px;color:{GRAY};font-family:JetBrains Mono,monospace;'
        f'letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;">Option Parameters</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.slider("Maturity T (months)", 1, 36, 6, key="opt_T_months")
    st.sidebar.slider("Strike %F", 70, 130, 100, key="opt_K_pct")
    st.sidebar.slider("Volatility σ %", 5, 120, int(COMMODITIES[commodity]["vol"]*100), key="opt_vol_pct")
    st.sidebar.slider("Rate r %", 0, 10, 5, key="opt_r_pct")
    st.sidebar.slider("Curve months", 3, 36, 18, key="curve_months")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.sidebar.markdown(
        f'<div style="font-size:9px;color:{GRAY};font-family:JetBrains Mono,monospace;'
        f'letter-spacing:0.08em;text-transform:uppercase;margin-top:20px;">'
        f'{now}<br>by Adam EL GBOURI · {date.today().year}<br>'
        f'aeg-snd.streamlit.app</div>',
        unsafe_allow_html=True,
    )
    return page, commodity

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
def render_header(page: str, commodity: str) -> None:
    c = COMMODITIES[commodity]
    col1, col2 = st.columns([5, 2])
    with col1:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">'
            f'<div style="width:36px;height:36px;border-radius:9px;'
            f'background:linear-gradient(135deg,{AMBER},{TEAL});display:flex;'
            f'align-items:center;justify-content:center;color:{BG};font-weight:800;font-size:17px;">C</div>'
            f'<div><div style="font-size:19px;font-weight:700;color:{TEXT};">Commodity Trading Desk</div>'
            f'<div style="font-size:9px;color:{GRAY};font-family:JetBrains Mono,monospace;'
            f'letter-spacing:0.18em;text-transform:uppercase;">by Adam EL GBOURI · {date.today().year}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div style="text-align:right;padding-top:10px;color:{GRAY};'
            f'font-size:11px;font-family:JetBrains Mono,monospace;">'
            f'{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<span class="badge badge-amber">LIVE</span>'
        f'<span class="badge">{c["sector"]}</span>'
        f'<span class="badge">{c["ticker"]}</span>'
        f'<span class="badge">{commodity}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard(commodity: str) -> None:
    c    = COMMODITIES[commodity]
    spot = c["spot"]
    rng  = np.random.default_rng(int(datetime.now().minute) + hash(commodity) % 1000)
    chg  = float(rng.normal(0, 1.2))
    fv   = spot * (1 + rng.uniform(-0.08, 0.08))
    fv_dev = (spot - fv) / fv * 100

    direction = "up" if chg > 0 else "down" if chg < 0 else "flat"
    fv_flag   = "rich" if fv_dev > 12 else "cheap" if fv_dev < -12 else "fairly priced"
    st.info(
        f"**{commodity}** is {direction} **{chg:+.2f}%** at `{spot:,.2f} {c['unit']}`. "
        f"Fair-value model: `{fv:,.2f}` → **{fv_flag}** ({fv_dev:+.1f}% vs spot). "
        f"Implied vol **{c['vol']*100:.0f}%**.",
        icon="🎯",
    )

    # KPIs
    cols = st.columns(5)
    kpi_data = [
        ("Spot Price",  f"{spot:,.2f} {c['unit']}", f"{chg:+.2f}% 1D",   AMBER),
        ("Fair Value",  f"{fv:,.2f}",               f"{fv_dev:+.1f}% vs spot", BLUE),
        ("Impl. Vol",   f"{c['vol']*100:.1f}%",     "annualised σ",       PURPLE),
        ("Storage",     f"{c['storage']*100:.1f}%", "annual cost",        GRAY),
        ("Conv. Yield", f"{c['conv']*100:.1f}%",    "annual yield",       TEAL),
    ]
    for col, (lbl, val, sub, acc) in zip(cols, kpi_data):
        col.markdown(kpi(lbl, val, sub, acc), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # S&D chart
    st.subheader("Supply, Demand & Stocks")
    st.caption("📦 How much is being produced (supply) vs consumed (demand), and how much is sitting in storage (stocks). When supply > demand, stocks build up and prices tend to fall. When demand > supply, stocks draw down and prices tend to rise.")
    df  = sd_dataset(commodity, 36)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["supply"], name="Supply",
                             line=dict(color=GREEN, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["demand"], name="Demand",
                             line=dict(color=RED,   width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["stocks"], name="Stocks",
                             line=dict(color=TEAL, width=1.5),
                             fill="tozeroy", fillcolor="rgba(57,208,216,0.08)",
                             yaxis="y2"))
    fc_start = df[df["is_forecast"]].index.min() if df["is_forecast"].any() else None
    if fc_start:
        fig.add_vline(x=str(fc_start), line=dict(color=GRAY, dash="dash"),
                      annotation_text="Forecast ▶", annotation_position="top right")
    fig.update_layout(yaxis2=dict(title="Stocks", overlaying="y", side="right",
                                  showgrid=False, color=TEAL))
    st.plotly_chart(_styled(fig, 380), use_container_width=True)

    # Market heatmap
    st.subheader("Market Heatmap")
    st.caption("🟢🔴 A snapshot of all commodities at once. Green = price up today, red = price down. Bigger boxes = bigger sectors. Click any cell to dig into that commodity.")
    rng2 = np.random.default_rng(int(datetime.now().minute))
    rows = []
    for name, info in COMMODITIES.items():
        chg_ = float(rng2.normal(0, 1.8))
        rows.append(dict(name=name, sector=info["sector"],
                         spot=info["spot"], change=chg_))
    sp_df = pd.DataFrame(rows)
    fig_hm = px.treemap(
        sp_df, path=[px.Constant("Markets"), "sector", "name"],
        values=[1]*len(sp_df), color="change",
        color_continuous_scale=[(0, RED), (0.5, PANEL), (1, GREEN)],
        color_continuous_midpoint=0,
        custom_data=["spot", "change"],
    )
    fig_hm.update_traces(
        # ── FIXED: both tile label and hover now show exactly 2 decimal places ──
        texttemplate="<b>%{label}</b><br>%{customdata[0]:.2f}<br>%{customdata[1]:+.2f}%",
        hovertemplate="<b>%{label}</b><br>Spot: %{customdata[0]:.2f}<br>"
                      "Chg: %{customdata[1]:+.2f}%<extra></extra>",
    )
    st.plotly_chart(_styled(fig_hm, 440), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: SUPPLY & DEMAND
# ══════════════════════════════════════════════════════════════════════════════
def page_balance(commodity: str) -> None:
    st.title(f"Supply & Demand — {commodity}")
    st.caption("⚖️ **What this page does:** Models the market balance — how much of this commodity is produced, consumed, and stored each month. Use the sliders to stress-test scenarios (e.g. what if supply drops 10%?). The fair value is the price the model thinks is justified given current stock levels.")
    c1, c2, c3, c4 = st.columns(4)
    sup_adj  = c1.slider("Supply adj %",   -20, 20,  0, 1, key="bal_sup")
    dem_adj  = c2.slider("Demand adj %",   -20, 20,  0, 1, key="bal_dem")
    gdp      = c3.slider("GDP growth %", -2.0, 6.0, 2.5, 0.1, key="bal_gdp")
    horizon  = c4.slider("Forecast months", 6, 36, 18, 3, key="bal_h")

    df = sd_dataset(commodity, horizon + 12)
    df["supply"]  *= 1 + sup_adj / 100
    df["demand"]  *= 1 + dem_adj / 100
    df["surplus"]  = df["supply"] - df["demand"]

    last = df.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("End stocks",       f"{last['stocks']:,.1f}")
    cols[1].metric("Avg surplus/def",  f"{df['surplus'].mean():+.2f}")
    cols[2].metric("Fair value",       f"{last['fair_value']:,.2f} {COMMODITIES[commodity]['unit']}")
    cols[3].metric("GDP assumption",   f"{gdp:+.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["supply"], name="Supply",
                             line=dict(color=GREEN, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["demand"], name="Demand",
                             line=dict(color=RED, width=2)))
    fig.add_trace(go.Bar(x=df.index, y=df["surplus"], name="Surplus/Deficit",
                         marker_color=np.where(df["surplus"] >= 0, GREEN, RED),
                         opacity=0.55, yaxis="y2"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
    st.plotly_chart(_styled(fig, 380), use_container_width=True)

    st.subheader("Price vs Fair Value")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["price"], name="Spot",
                              line=dict(color=TEXT, width=2)))
    fig2.add_trace(go.Scatter(x=df.index, y=df["fair_value"], name="Fair Value",
                              line=dict(color=AMBER, width=2, dash="dot")))
    st.plotly_chart(_styled(fig2, 300), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: REGIONAL FLOWS
# ══════════════════════════════════════════════════════════════════════════════
def page_regional(commodity: str) -> None:
    st.title(f"Regional Flows — {commodity}")
    st.caption("🌍 **What this page does:** Shows who produces and who consumes this commodity around the world. Green bubbles = net exporters (they sell to the world). Red bubbles = net importers (they buy from the world). The size of the bubble shows how large the imbalance is.")
    key = commodity if commodity in REGIONAL_DATA else list(REGIONAL_DATA.keys())[0]
    reg = pd.DataFrame(REGIONAL_DATA[key])
    reg["net"] = reg["supply"] - reg["demand"]
    reg["status"] = np.where(reg["net"] > 0, "Exporter", "Importer")

    ws = float(reg["supply"].sum())
    wd = float(reg["demand"].sum())
    cols = st.columns(4)
    cols[0].metric("World Supply",  f"{ws:,.1f} {COMMODITIES[commodity]['unit']}")
    cols[1].metric("World Demand",  f"{wd:,.1f} {COMMODITIES[commodity]['unit']}")
    cols[2].metric("Balance",       f"{ws-wd:+,.2f}", "surplus" if ws > wd else "deficit")
    cols[3].metric("Regions",       str(len(reg)))

    fig_map = go.Figure()
    for _, r in reg.iterrows():
        color = GREEN if r["net"] >= 0 else RED
        fig_map.add_trace(go.Scattergeo(
            lat=[r["lat"]], lon=[r["lon"]],
            mode="markers+text",
            marker=dict(size=abs(r["net"])**0.5 * 4 + 8, color=color, opacity=0.75,
                        line=dict(color=BORDER, width=1)),
            text=r["region"],
            textposition="top center",
            textfont=dict(size=10, color=TEXT, family="JetBrains Mono"),
            name=r["region"],
            hovertemplate=(
                f"<b>{r['region']}</b><br>"
                f"Supply: {r['supply']:.1f}<br>"
                f"Demand: {r['demand']:.1f}<br>"
                f"Net: {r['net']:+.1f}<br>"
                f"Status: {r['status']}<extra></extra>"
            ),
        ))
    fig_map.update_layout(
        geo=dict(
            bgcolor=BG, showframe=False, showcoastlines=True,
            coastlinecolor=BORDER, landcolor=PANEL, oceancolor=BG,
            showocean=True, showland=True, showlakes=False,
            projection_type="natural earth",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=420, margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    fig2 = go.Figure(go.Bar(
        x=reg["region"], y=reg["net"],
        marker_color=np.where(reg["net"] >= 0, GREEN, RED),
        text=[f"{v:+.1f}" for v in reg["net"]], textposition="outside",
    ))
    fig2.update_layout(title="Net Trade (Supply − Demand)")
    st.plotly_chart(_styled(fig2, 300), use_container_width=True)

    st.dataframe(reg.style.format({
        "supply": "{:,.2f}", "demand": "{:,.2f}", "net": "{:+,.2f}"}),
        use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FUTURES CURVE
# ══════════════════════════════════════════════════════════════════════════════
def page_curve(commodity: str) -> None:
    st.title(f"Futures Curve — {commodity}")
    st.caption("📈 **What this page does:** Shows what the market expects this commodity to cost in the future. *Contango* = futures prices rise with maturity (market expects higher prices later, common when storage is cheap). *Backwardation* = futures prices fall with maturity (market expects lower prices later, common when supply is tight right now).")
    c   = COMMODITIES[commodity]
    mnths = st.session_state.get("curve_months", 18)
    curve = forward_curve(c["spot"], 0.05, c["storage"], c["conv"], c["vol"], mnths)

    front = float(curve["price"].iloc[0])
    back  = float(curve["price"].iloc[-1])
    structure = ("CONTANGO" if back > front * 1.005
                 else "BACKWARDATION" if back < front * 0.995 else "FLAT")
    s_color = RED if structure == "CONTANGO" else GREEN if structure == "BACKWARDATION" else AMBER

    cols = st.columns(4)
    cols[0].metric("Spot",        f"{c['spot']:,.2f} {c['unit']}")
    cols[1].metric("Front Month", f"{front:,.2f}")
    cols[2].metric("12M Forward", f"{float(curve[curve['month']<=12]['price'].iloc[-1]):,.2f}")
    cols[3].metric("Structure",   structure, f"{(back-front)/front*100:+.2f}%")
    st.markdown(
        f'<span class="badge" style="border-color:{s_color};color:{s_color};">'
        f'⚡ {structure}</span>', unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["label"], y=curve["price"],
                             mode="lines+markers",
                             line=dict(color=AMBER, width=2.5),
                             marker=dict(size=7, color=AMBER),
                             name="Forward curve"))
    fig.add_trace(go.Scatter(x=curve["label"], y=[c["spot"]]*len(curve),
                             mode="lines", name="Spot",
                             line=dict(color=TEXT, dash="dot", width=1.5)))
    fig.update_layout(title=f"{commodity} Forward Curve")
    st.plotly_chart(_styled(fig, 380), use_container_width=True)
    st.dataframe(curve[["label","T","price"]].rename(
        columns={"label":"Contract","T":"Maturity (yr)","price":"Price"}),
        use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: OPTIONS & GREEKS
# ══════════════════════════════════════════════════════════════════════════════
def page_options(commodity: str) -> None:
    st.title(f"Options & Greeks — {commodity}")
    st.caption("🎯 **What this page does:** Prices a call or put option on this commodity using the Black-76 model (the industry standard for commodity options). A *call* profits if price rises above your strike. A *put* profits if price falls below. The Greeks measure sensitivities: Delta = how much the option moves per $1 move in the commodity; Vega = sensitivity to volatility; Theta = time decay per day.")
    c = COMMODITIES[commodity]
    F_def    = c["spot"]
    T_months = st.session_state.get("opt_T_months", 6)
    K_pct    = st.session_state.get("opt_K_pct", 100)
    vol_pct  = st.session_state.get("opt_vol_pct", int(c["vol"]*100))
    r_pct    = st.session_state.get("opt_r_pct", 5)

    col1, col2 = st.columns(2)
    with col1:
        F = st.number_input("Forward F", value=F_def, step=F_def*0.005)
    with col2:
        K = st.number_input("Strike K", value=F_def * K_pct/100, step=F_def*0.005)

    T     = T_months / 12
    sigma = vol_pct / 100
    r     = r_pct / 100

    call = black76(F, K, T, r, sigma, "call")
    put  = black76(F, K, T, r, sigma, "put")
    moneyness = "ITM" if F > K else "OTM" if F < K else "ATM"
    pcp_check = call["price"] - put["price"] - math.exp(-r*T)*(F-K)

    st.markdown(
        f'<span class="badge badge-amber">{moneyness}</span>'
        f'<span class="badge">T={T:.3f}yr</span>'
        f'<span class="badge">σ={sigma*100:.1f}%</span>'
        f'<span class="badge" style="color:{GREEN};">PCP err={pcp_check:.2e}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    greek_labels = ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"]
    greek_keys   = ["price", "delta", "gamma", "vega", "theta", "rho"]
    st.subheader("Greeks")
    tab_c, tab_p = st.tabs(["📞 Call", "📤 Put"])
    for tab, g, acc in [(tab_c, call, AMBER), (tab_p, put, BLUE)]:
        with tab:
            cols = st.columns(6)
            for col, lbl, k in zip(cols, greek_labels, greek_keys):
                col.markdown(kpi(lbl, f"{g[k]:.5f}", "", acc), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    strikes  = np.linspace(F * 0.55, F * 1.45, 80)
    call_pnl = np.maximum(strikes - K, 0) - call["price"]
    put_pnl  = np.maximum(K - strikes, 0) - put["price"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=call_pnl, name="Long Call",
                             line=dict(color=GREEN, width=2.5)))
    fig.add_trace(go.Scatter(x=strikes, y=put_pnl, name="Long Put",
                             line=dict(color=RED, width=2.5)))
    fig.add_hline(y=0, line=dict(color=GRAY, dash="dash", width=1))
    fig.add_vline(x=K, line=dict(color=AMBER, dash="dot", width=1.5),
                  annotation_text="Strike", annotation_position="top right")
    fig.add_vline(x=F, line=dict(color=BLUE, dash="dot", width=1.5),
                  annotation_text="Forward", annotation_position="top left")
    fig.update_layout(title="Payoff at Expiry (net of premium)")
    st.plotly_chart(_styled(fig, 360), use_container_width=True)

    st.subheader("Greeks vs Strike")
    ks = np.linspace(F*0.7, F*1.3, 60)
    tab1, tab2, tab3 = st.tabs(["Delta", "Gamma", "Vega"])
    for tab, gk, col_ in [(tab1,"delta",AMBER),(tab2,"gamma",PURPLE),(tab3,"vega",TEAL)]:
        with tab:
            vals_c = [black76(F, k, T, r, sigma, "call")[gk] for k in ks]
            vals_p = [black76(F, k, T, r, sigma, "put")[gk] for k in ks]
            fig_ = go.Figure()
            fig_.add_trace(go.Scatter(x=ks, y=vals_c, name=f"Call {gk}",
                                      line=dict(color=col_, width=2)))
            fig_.add_trace(go.Scatter(x=ks, y=vals_p, name=f"Put {gk}",
                                      line=dict(color=BLUE, width=2)))
            fig_.add_vline(x=K, line=dict(color=AMBER, dash="dot"))
            st.plotly_chart(_styled(fig_, 280), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: VOL SURFACE
# ══════════════════════════════════════════════════════════════════════════════
def page_vol_surface(commodity: str) -> None:
    st.title(f"Implied Vol Surface — {commodity}")
    st.caption("📉 **What this page does:** Shows how volatility varies by strike price and expiry date. In a perfect world vol would be flat (a horizontal plane). In reality it forms a 'smile' — out-of-the-money options are more expensive because traders pay a premium for tail protection. *Skew* tilts the smile (puts are usually pricier than calls in commodities). *Curvature* controls how curved the smile is. *Vol-of-vol* makes the surface steeper for longer maturities.")
    c = COMMODITIES[commodity]
    col1, col2, col3, col4 = st.columns(4)
    atm  = col1.slider("ATM vol %",     5, 120, int(c["vol"]*100), key="vs_atm")  / 100
    skew = col2.slider("Skew ×100",   -20,  20, -5,               key="vs_skew") / 100
    curv = col3.slider("Curvature×100", 0,  10,  2,               key="vs_curv") / 100
    vov  = col4.slider("Vol-of-vol",    0, 100, 15,               key="vs_vov")  / 100

    mats, Kgrid, Z = vol_surface(c["spot"], atm, skew, curv, vov)
    mat_labels = ["1M","2M","3M","6M","9M","12M","18M","24M"]

    fig = go.Figure(data=go.Surface(
        z=Z, x=np.log(Kgrid/c["spot"]), y=[m*12 for m in mats],
        colorscale=[[0, BLUE],[0.5, PURPLE],[1, AMBER]],
        showscale=True,
        colorbar=dict(title="σ", tickfont=dict(color=TEXT)),
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="ln(K/F)", color=GRAY, gridcolor=BORDER, backgroundcolor=BG),
            yaxis=dict(title="Maturity (months)", color=GRAY, gridcolor=BORDER, backgroundcolor=BG),
            zaxis=dict(title="Implied Vol", color=GRAY, gridcolor=BORDER, backgroundcolor=BG),
            bgcolor=BG,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=520, margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color=TEXT),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Vol Smile by Maturity")
    fig2 = go.Figure()
    pal = [AMBER, BLUE, GREEN, RED, PURPLE, TEAL, GRAY, TEXT]
    for i, (row, label, col_) in enumerate(zip(Z, mat_labels, pal)):
        fig2.add_trace(go.Scatter(
            x=np.log(Kgrid/c["spot"]), y=row*100,
            name=label, line=dict(color=col_, width=2),
        ))
    fig2.update_layout(xaxis_title="ln(K/F)", yaxis_title="Impl. Vol %")
    st.plotly_chart(_styled(fig2, 360), use_container_width=True)

    atm_ts = Z[:, Z.shape[1]//2] * 100
    fig3 = go.Figure(go.Bar(x=mat_labels, y=atm_ts, marker_color=AMBER))
    fig3.update_layout(title="ATM Vol Term Structure (%)", yaxis_title="σ %")
    st.plotly_chart(_styled(fig3, 260), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
def page_positions() -> None:
    st.title("Positions & P&L")
    st.caption("💼 **What this page does:** Your trade blotter. Add a position (Long = you profit if price rises, Short = you profit if price falls), enter your entry price, and the app calculates your live P&L based on current market prices. All positions are stored in your browser session — they reset on refresh.")

    if "positions" not in st.session_state:
        st.session_state["positions"] = []

    with st.expander("➕ Add Position", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        name  = c1.selectbox("Commodity", list(COMMODITIES.keys()), key="pos_name")
        side  = c2.selectbox("Side", ["Long", "Short"], key="pos_side")
        qty   = c3.number_input("Quantity", value=100, step=10, key="pos_qty")
        entry = c4.number_input("Entry Price",
                                value=COMMODITIES[name]["spot"], step=1.0, key="pos_entry")
        if c5.button("Add Trade", use_container_width=True, type="primary"):
            st.session_state["positions"].append(
                dict(commodity=name, side=side, quantity=float(qty), entry_price=float(entry))
            )
            st.rerun()

    positions = st.session_state["positions"]
    if not positions:
        st.info("No positions yet. Add one above.")
        return

    rows, total_pnl, total_long, total_short = [], 0.0, 0.0, 0.0
    for p in positions:
        c    = COMMODITIES[p["commodity"]]
        mark = c["spot"]
        sign = 1 if p["side"] == "Long" else -1
        pnl  = sign * (mark - p["entry_price"]) * p["quantity"]
        total_pnl += pnl
        if p["side"] == "Long": total_long  += mark * p["quantity"]
        else:                   total_short += mark * p["quantity"]
        rows.append(dict(
            Commodity=p["commodity"], Side=p["side"],
            Qty=p["quantity"], Entry=p["entry_price"], Mark=mark,
            **{"P&L/unit": sign*(mark-p["entry_price"]),
               "P&L Total": pnl,
               "Return %":  sign*(mark-p["entry_price"])/p["entry_price"]*100},
        ))

    cols = st.columns(4)
    cols[0].metric("Gross Long",   f"${total_long:,.0f}")
    cols[1].metric("Gross Short",  f"${total_short:,.0f}")
    cols[2].metric("Net Exposure", f"${total_long-total_short:+,.0f}")
    cols[3].metric("Total P&L",    f"${total_pnl:+,.0f}", f"{len(positions)} trades")

    df = pd.DataFrame(rows)

    def _color_pnl(val):
        if isinstance(val, (int, float)):
            return f"color: {GREEN}" if val >= 0 else f"color: {RED}"
        return ""

    st.dataframe(
        df.style
          .format({"Entry": "{:.2f}", "Mark": "{:.2f}",
                   "P&L/unit": "{:+.3f}", "P&L Total": "{:+,.0f}",
                   "Return %": "{:+.1f}%"})
          .map(_color_pnl, subset=["P&L/unit", "P&L Total", "Return %"]),
        use_container_width=True,
    )

    if st.button("🗑️ Clear All Positions"):
        st.session_state["positions"] = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RISK
# ══════════════════════════════════════════════════════════════════════════════
def page_risk() -> None:
    st.title("Risk Dashboard")
    st.caption("🛡️ **What this page does:** Measures how much money you could lose. *VaR (Value at Risk)* = the maximum loss you'd expect on X% of trading days (e.g. 95% VaR = you'd lose more than this amount only 5% of the time). *CVaR* = the average loss in the worst 5% of cases. Stress scenarios show the impact of extreme price shocks (+/-10%, +/-20%, +/-30%).")
    positions = st.session_state.get("positions", [])
    if not positions:
        st.warning("Add positions on the Positions page first.")
        return

    col1, col2 = st.columns(2)
    conf    = col1.selectbox("Confidence", [0.90, 0.95, 0.99], index=1)
    horizon = col2.slider("Horizon (days)", 1, 30, 1, key="risk_h")

    risk = portfolio_var(positions, conf=conf, horizon=horizon)
    cols = st.columns(3)
    cols[0].metric(f"VaR {int(conf*100)}%",  f"${risk['total_var']:,.0f}",  f"{horizon}d horizon")
    cols[1].metric(f"CVaR {int(conf*100)}%", f"${risk['total_cvar']:,.0f}", "Expected shortfall")
    cols[2].metric("Positions", str(len(positions)))

    st.subheader("Per-Position Decomposition")
    rdf = pd.DataFrame(risk["rows"])
    st.dataframe(rdf.style.format({
        "vol_pct": "{:.1f}%", "var": "${:,.0f}", "cvar": "${:,.0f}",
        "spot": "{:.2f}", "quantity": "{:.0f}"}),
        use_container_width=True)

    p    = positions[0]
    c_   = COMMODITIES.get(p["commodity"], {})
    base = c_.get("spot", p["entry_price"])
    shocks = [-30, -20, -10, -5, 5, 10, 20, 30]
    sign = 1 if p["side"] == "Long" else -1
    sc_rows = []
    for sh in shocks:
        new_price = base * (1 + sh/100)
        pnl = sign * (new_price - p["entry_price"]) * p["quantity"]
        sc_rows.append(dict(shock_pct=sh, new_price=new_price, pnl_impact=pnl))
    sdf = pd.DataFrame(sc_rows)

    st.subheader(f"Stress Scenarios — {p['commodity']}")
    fig = go.Figure(go.Bar(
        x=[f"{r['shock_pct']:+.0f}%" for _, r in sdf.iterrows()],
        y=sdf["pnl_impact"],
        marker_color=np.where(sdf["pnl_impact"] >= 0, GREEN, RED),
        text=[f"${v:+,.0f}" for v in sdf["pnl_impact"]],
        textposition="outside",
    ))
    st.plotly_chart(_styled(fig, 300), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════
def page_mc(commodity: str) -> None:
    st.title(f"Monte Carlo — {commodity}")
    st.caption("🎲 **What this page does:** Simulates thousands of possible price paths into the future using random shocks (Geometric Brownian Motion). The fan chart shows the range of outcomes: the thick amber line is the median (50% of paths end here), the green/red bands show the bull and bear extremes. Wider bands = more uncertain outlook. Useful for option pricing, hedging, and scenario planning.")

    col1, col2, col3, col4 = st.columns(4)
    n_paths = col1.slider("Paths",          100, 2000, 500, 50,  key="mc_n")
    sup_sig = col2.slider("Supply σ %",     0.5,  6.0, 1.5, 0.1, key="mc_ss")
    dem_sig = col3.slider("Demand σ %",     0.5,  6.0, 1.2, 0.1, key="mc_ds")
    horizon = col4.slider("Horizon months", 3,    36,  18,  3,   key="mc_h")

    with st.spinner(f"Running {n_paths:,} simulations…"):
        res = run_mc(commodity, n_paths, horizon, sup_sig, dem_sig)

    cols = st.columns(4)
    cols[0].metric("Median end price", f"{res['median']:,.2f} {res['unit']}")
    cols[1].metric("P5  (bear case)",  f"{res['p5']:,.2f}")
    cols[2].metric("P95 (bull case)",  f"{res['p95']:,.2f}")
    cols[3].metric("P95/P5 ratio",     f"{res['p95']/res['p5']:.2f}×")

    fan = res["fan"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fan["date"], y=fan["p95"], name="P95",
                             line=dict(color=GREEN, width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=fan["date"], y=fan["p75"], name="P75",
                             line=dict(color=GREEN, width=0.8),
                             fill="tonexty", fillcolor="rgba(63,185,80,0.08)"))
    fig.add_trace(go.Scatter(x=fan["date"], y=fan["p50"], name="P50 (Median)",
                             line=dict(color=AMBER, width=2.5),
                             fill="tonexty", fillcolor="rgba(240,165,0,0.06)"))
    fig.add_trace(go.Scatter(x=fan["date"], y=fan["p25"], name="P25",
                             line=dict(color=RED, width=0.8),
                             fill="tonexty", fillcolor="rgba(255,123,114,0.06)"))
    fig.add_trace(go.Scatter(x=fan["date"], y=fan["p5"], name="P5",
                             line=dict(color=RED, width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(255,123,114,0.08)"))
    fig.update_layout(title="Price Fan Chart")
    st.plotly_chart(_styled(fig, 420), use_container_width=True)

    st.subheader("Distribution of Simulated End Prices")
    fig_h = go.Figure(go.Bar(x=res["hist_x"], y=res["hist_y"],
                             marker_color=AMBER, opacity=0.75, name="Frequency"))
    fig_h.add_vline(x=res["median"], line=dict(color=TEXT,  dash="dash"), annotation_text="Median")
    fig_h.add_vline(x=res["p5"],    line=dict(color=RED,   dash="dot"),  annotation_text="P5")
    fig_h.add_vline(x=res["p95"],   line=dict(color=GREEN, dash="dot"),  annotation_text="P95")
    st.plotly_chart(_styled(fig_h, 280), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MACRO OVERLAY
# ══════════════════════════════════════════════════════════════════════════════
def page_macro() -> None:
    st.title("Macro Overlay")
    st.caption("🌐 **What this page does:** Overlays macroeconomic data on top of commodity analysis. Commodity prices are heavily influenced by the global economy — rising GDP drives energy and metals demand; high inflation erodes real returns; rate hikes strengthen the dollar (which tends to push commodity prices down). Compare countries side by side to spot divergences.")
    col1, col2 = st.columns([2, 5])
    primary = col1.selectbox("Country", COUNTRIES, key="macro_primary")
    compare = col2.multiselect("Compare with",
                               [c for c in COUNTRIES if c != primary],
                               default=[c for c in COUNTRIES if c != primary][:2],
                               key="macro_cmp")

    series = [(primary, macro_data(primary))]
    series.extend((c, macro_data(c)) for c in compare)

    snap = series[0][1].iloc[-1]
    cols = st.columns(4)
    for col, m, lab in zip(cols,
                            ["gdp_index","cpi_yoy","policy_rate","pmi"],
                            ["GDP Index","CPI YoY %","Policy Rate %","PMI"]):
        cols[cols.index(col)].metric(lab, f"{snap[m]:.2f}")

    metrics = dict(gdp_index="GDP Index", cpi_yoy="CPI YoY %",
                   policy_rate="Policy Rate %", pmi="PMI")
    tabs = st.tabs(list(metrics.values()))
    for tab, (mk, ml) in zip(tabs, metrics.items()):
        with tab:
            fig = go.Figure()
            pal = [AMBER, BLUE, GREEN, RED]
            for i, (name, df) in enumerate(series):
                if mk in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df[mk], name=name,
                                             line=dict(color=pal[i % len(pal)], width=2)))
            fig.update_layout(title=ml, yaxis_title=ml)
            st.plotly_chart(_styled(fig, 360), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: EVENTS
# ══════════════════════════════════════════════════════════════════════════════
def page_events() -> None:
    st.title("Market Events Calendar")
    st.caption("📅 **What this page does:** Lists upcoming scheduled events that typically move commodity prices. EIA reports (oil inventory data), WASDE (crop supply/demand), OPEC meetings (production decisions), and central bank decisions all cause volatility. Knowing when they're coming helps traders avoid being caught offside.")
    today = date.today()
    df = pd.DataFrame([
        dict(Date=str(e["date"]),
             Days=str((e["date"]-today).days)+"d",
             Event=e["event"],
             Tags=", ".join(e["tags"]),
             Today=(e["date"]==today))
        for e in EVENTS
    ])

    today_mask = df["Today"].tolist()
    display_df = df.drop(columns=["Today"])

    def _style(r):
        idx = int(r.name)
        if today_mask[idx]:
            return ["background-color:#1a2744"] * len(r)
        return [""] * len(r)

    st.dataframe(
        display_df.style.apply(_style, axis=1),
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
def page_about() -> None:
    st.title("About")
    st.markdown(f"""
**Commodity Trading Desk** — standalone Streamlit app, no external backend.

**Analytics (built-in):**
- Black-76 European options pricer (calls, puts, full Greeks)
- Parametric implied vol surface (ATM + skew + curvature + vol-of-vol)
- Cost-of-carry forward curve (Nelson-Siegel inspired with noise)
- Synthetic S&D balance with fair-value model
- GBM Monte Carlo with P5/P25/P50/P75/P95 fan chart
- Parametric VaR / CVaR with stress scenarios

**{len(COMMODITIES)} commodities** across Energy, Metals, Agriculture, Freight sectors.

---
### 🔗 My Other Projects

You may also find these two companion platforms useful:

**⚗️ Commodity Options & Derivatives Analytics Platform (CODAP)**  
A dedicated platform for commodity options pricing, Greeks, vol surfaces, exotic derivatives (Asian, barrier, crack spreads), and swap analytics.  
👉 [aeg-codap.streamlit.app](https://aeg-codap.streamlit.app)

**〽️ Commodity Forward Curve Analytics Platform (CFCAP)**  
A dedicated platform for forward curve construction, Nelson-Siegel fitting, calendar spreads, curve interpolation, and term structure analysis across all major commodity markets.  
👉 [aeg-cfcap.streamlit.app](https://aeg-cfcap.streamlit.app)

---
**Author:** Adam EL GBOURI  
GitHub · [github.com/adamelgbouri](https://github.com/adamelgbouri)
""")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    page, commodity = render_sidebar()
    render_header(page, commodity)

    dispatch = {
        "📊 Dashboard":        lambda: page_dashboard(commodity),
        "⚖️ Supply & Demand":  lambda: page_balance(commodity),
        "🌍 Regional Flows":   lambda: page_regional(commodity),
        "📈 Futures Curve":    lambda: page_curve(commodity),
        "🎯 Options & Greeks": lambda: page_options(commodity),
        "📉 Vol Surface":      lambda: page_vol_surface(commodity),
        "💼 Positions & P&L":  page_positions,
        "🛡️ Risk":             page_risk,
        "🎲 Monte Carlo":      lambda: page_mc(commodity),
        "🌐 Macro Overlay":    page_macro,
        "📅 Events":           page_events,
        "ℹ️ About":            page_about,
    }
    dispatch.get(page, lambda: st.error("Page not found"))()


if __name__ == "__main__":
    main()
