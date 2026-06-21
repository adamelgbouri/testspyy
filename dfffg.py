"""
Commodity Trading Desk — standalone single-file Streamlit app
by Adam EL GBOURI

Real market data via yfinance for all Yahoo-sourced commodities.
Heatmap compares prices between two user-selected dates/times.

Run:
    pip install streamlit plotly numpy pandas scipy yfinance
    streamlit run commodity_trading_desk.py
"""
from __future__ import annotations

import math
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
from scipy.optimize import brentq

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  COLOR PALETTE
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
#  PAGE CONFIG & CSS
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
#  COMMODITY REGISTRY  — yf_ticker = Yahoo Finance continuous contract symbol
# ══════════════════════════════════════════════════════════════════════════════
COMMODITIES: Dict[str, dict] = {
    # ── Energy ────────────────────────────────────────────────────────────────
    "WTI Crude Oil":         dict(sector="Energy",      unit="$/bbl",      yf_ticker="CL=F",  vol=0.32, storage=0.096, conv=0.08,  ticker="CL",  fallback=67.50),
    "Brent Crude Oil":       dict(sector="Energy",      unit="$/bbl",      yf_ticker="BZ=F",  vol=0.30, storage=0.096, conv=0.07,  ticker="BZ",  fallback=71.20),
    "Natural Gas":           dict(sector="Energy",      unit="$/MMBtu",    yf_ticker="NG=F",  vol=0.55, storage=0.120, conv=0.10,  ticker="NG",  fallback=3.15),
    "RBOB Gasoline":         dict(sector="Energy",      unit="$/gallon",   yf_ticker="RB=F",  vol=0.36, storage=0.084, conv=0.07,  ticker="RB",  fallback=2.48),
    "Heating Oil (ULSD)":    dict(sector="Energy",      unit="$/gallon",   yf_ticker="HO=F",  vol=0.34, storage=0.084, conv=0.07,  ticker="HO",  fallback=2.62),
    "Gasoil ICE":            dict(sector="Energy",      unit="$/mt",       yf_ticker="LGO=F", vol=0.32, storage=0.072, conv=0.07,  ticker="GO",  fallback=770.0),
    "European Carbon (EUA)": dict(sector="Energy",      unit="EUR/tCO2",   yf_ticker=None,    vol=0.35, storage=0.024, conv=0.02,  ticker="EUA", fallback=63.0),
    "Coal API2":             dict(sector="Energy",      unit="$/mt",       yf_ticker=None,    vol=0.30, storage=0.048, conv=0.04,  ticker="MTF", fallback=108.0),
    # ── Metals ────────────────────────────────────────────────────────────────
    "Gold":                  dict(sector="Metals",      unit="$/troy oz",  yf_ticker="GC=F",  vol=0.15, storage=0.024, conv=0.005, ticker="GC",  fallback=3310.0),
    "Silver":                dict(sector="Metals",      unit="$/troy oz",  yf_ticker="SI=F",  vol=0.28, storage=0.036, conv=0.010, ticker="SI",  fallback=32.8),
    "Copper (COMEX)":        dict(sector="Metals",      unit="$/lb",       yf_ticker="HG=F",  vol=0.22, storage=0.048, conv=0.030, ticker="HG",  fallback=4.55),
    "Platinum":              dict(sector="Metals",      unit="$/troy oz",  yf_ticker="PL=F",  vol=0.20, storage=0.030, conv=0.015, ticker="PL",  fallback=1010.0),
    "Palladium":             dict(sector="Metals",      unit="$/troy oz",  yf_ticker="PA=F",  vol=0.30, storage=0.030, conv=0.020, ticker="PA",  fallback=1090.0),
    "LME Copper":            dict(sector="Metals",      unit="$/mt",       yf_ticker="HG=F",  vol=0.22, storage=0.048, conv=0.030, ticker="LP",  fallback=9750.0),
    "LME Aluminum":          dict(sector="Metals",      unit="$/mt",       yf_ticker=None,    vol=0.20, storage=0.048, conv=0.025, ticker="LA",  fallback=2390.0),
    "LME Nickel":            dict(sector="Metals",      unit="$/mt",       yf_ticker=None,    vol=0.30, storage=0.048, conv=0.035, ticker="LN",  fallback=15800.0),
    # ── Agriculture ───────────────────────────────────────────────────────────
    "Corn":                  dict(sector="Agriculture", unit="c/bushel",   yf_ticker="ZC=F",  vol=0.25, storage=0.060, conv=0.04,  ticker="ZC",  fallback=468.0),
    "Wheat (CBOT)":          dict(sector="Agriculture", unit="c/bushel",   yf_ticker="ZW=F",  vol=0.28, storage=0.060, conv=0.04,  ticker="ZW",  fallback=558.0),
    "Soybeans":              dict(sector="Agriculture", unit="c/bushel",   yf_ticker="ZS=F",  vol=0.23, storage=0.060, conv=0.05,  ticker="ZS",  fallback=1002.0),
    "Sugar #11":             dict(sector="Agriculture", unit="c/lb",       yf_ticker="SB=F",  vol=0.30, storage=0.048, conv=0.04,  ticker="SB",  fallback=18.9),
    "Coffee (Arabica)":      dict(sector="Agriculture", unit="c/lb",       yf_ticker="KC=F",  vol=0.35, storage=0.048, conv=0.05,  ticker="KC",  fallback=345.0),
    "Cocoa":                 dict(sector="Agriculture", unit="$/mt",       yf_ticker="CC=F",  vol=0.32, storage=0.048, conv=0.04,  ticker="CC",  fallback=7850.0),
    "Live Cattle":           dict(sector="Agriculture", unit="c/lb",       yf_ticker="LE=F",  vol=0.18, storage=0.036, conv=0.03,  ticker="LE",  fallback=183.0),
    "Lean Hogs":             dict(sector="Agriculture", unit="c/lb",       yf_ticker="HE=F",  vol=0.25, storage=0.036, conv=0.03,  ticker="HE",  fallback=91.5),
    # ── Freight ───────────────────────────────────────────────────────────────
    "Capesize (BCI 5TC)":    dict(sector="Freight",     unit="$/day",      yf_ticker=None,    vol=0.55, storage=0.0,   conv=0.00,  ticker="BCI", fallback=17500.0),
    "Panamax (BPI 4TC)":     dict(sector="Freight",     unit="$/day",      yf_ticker=None,    vol=0.50, storage=0.0,   conv=0.00,  ticker="BPI", fallback=11800.0),
}

ALL_SECTORS = sorted({v["sector"] for v in COMMODITIES.values()})

# ── Yahoo tickers that actually have data ─────────────────────────────────────
YF_TICKERS = {
    name: info["yf_ticker"]
    for name, info in COMMODITIES.items()
    if info.get("yf_ticker")
}

# ══════════════════════════════════════════════════════════════════════════════
#  LIVE DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)   # cache 5 min — refreshes automatically
def fetch_live_prices() -> Dict[str, float]:
    """Download latest close for all Yahoo-sourced commodities. Falls back to hardcoded values."""
    result = {name: info["fallback"] for name, info in COMMODITIES.items()}
    if not YF_AVAILABLE:
        return result
    tickers = list(YF_TICKERS.values())
    try:
        raw = yf.download(tickers, period="5d", auto_adjust=True,
                          progress=False, threads=True)
        closes = raw["Close"].iloc[-1] if isinstance(raw.columns, pd.MultiIndex) else raw.iloc[-1]
        for name, yf_t in YF_TICKERS.items():
            if yf_t in closes.index and pd.notna(closes[yf_t]):
                result[name] = float(closes[yf_t])
    except Exception:
        pass
    return result


@st.cache_data(ttl=3600)   # cache 1 h
def fetch_history(yf_ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download daily OHLCV for a single ticker. Returns empty df on failure."""
    if not YF_AVAILABLE or not yf_ticker:
        return pd.DataFrame()
    try:
        df = yf.download(yf_ticker, period=period, auto_adjust=True,
                         progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_price_at_date(yf_ticker: str, target_date: date) -> Optional[float]:
    """Return the closing price on or just before target_date."""
    if not YF_AVAILABLE or not yf_ticker:
        return None
    try:
        start = (datetime.combine(target_date, datetime.min.time()) - timedelta(days=7)).strftime("%Y-%m-%d")
        end   = (datetime.combine(target_date, datetime.min.time()) + timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(yf_ticker, start=start, end=end,
                         auto_adjust=True, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None
        # pick the row closest to (and not after) target_date
        df = df[df.index.date <= target_date]
        return float(df["Close"].iloc[-1]) if not df.empty else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  REGIONAL DATA  (fundamental, not price-based — kept static/realistic)
# ══════════════════════════════════════════════════════════════════════════════
REGIONAL_DATA = {
    "WTI Crude Oil": [
        dict(region="North America", supply=15.2, demand=10.8, lat=45,  lon=-100),
        dict(region="Middle East",   supply=28.1, demand=8.2,  lat=25,  lon=50),
        dict(region="Russia/FSU",    supply=13.5, demand=5.1,  lat=60,  lon=70),
        dict(region="Europe",        supply=3.4,  demand=12.6, lat=52,  lon=10),
        dict(region="Asia Pacific",  supply=8.1,  demand=34.5, lat=30,  lon=115),
        dict(region="Africa",        supply=7.9,  demand=4.3,  lat=5,   lon=20),
        dict(region="Latin America", supply=5.8,  demand=6.2,  lat=-15, lon=-60),
    ],
    "Gold": [
        dict(region="China",      supply=370, demand=950,  lat=35,  lon=105),
        dict(region="Australia",  supply=330, demand=30,   lat=-25, lon=133),
        dict(region="Russia",     supply=295, demand=90,   lat=60,  lon=70),
        dict(region="Canada",     supply=190, demand=50,   lat=60,  lon=-95),
        dict(region="USA",        supply=170, demand=230,  lat=38,  lon=-97),
        dict(region="S. Africa",  supply=120, demand=45,   lat=-30, lon=25),
        dict(region="India",      supply=35,  demand=800,  lat=20,  lon=80),
        dict(region="Europe",     supply=30,  demand=280,  lat=50,  lon=15),
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
def black76(F, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return dict(price=0, delta=0, gamma=0, vega=0, theta=0, rho=0)
    d1   = (math.log(F/K) + 0.5*sigma**2*T) / (sigma*math.sqrt(T))
    d2   = d1 - sigma*math.sqrt(T)
    disc = math.exp(-r*T)
    if option_type == "call":
        price, delta = disc*(F*norm.cdf(d1)-K*norm.cdf(d2)), disc*norm.cdf(d1)
    else:
        price, delta = disc*(K*norm.cdf(-d2)-F*norm.cdf(-d1)), -disc*norm.cdf(-d1)
    gamma = disc*norm.pdf(d1)/(F*sigma*math.sqrt(T))
    vega  = disc*F*norm.pdf(d1)*math.sqrt(T)/100
    theta = (-(disc*F*norm.pdf(d1)*sigma)/(2*math.sqrt(T)) - r*price)/365
    rho   = -T*price/100
    return dict(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def vol_surface_fn(F, atm_vol, skew=-0.05, curv=0.02, vov=0.15):
    mats = np.array([1/12,2/12,3/12,6/12,9/12,1.0,1.5,2.0])
    Kgrid = F * np.exp(np.linspace(-0.40, 0.40, 25))
    Z = np.zeros((len(mats), len(Kgrid)))
    for i, T in enumerate(mats):
        for j, K in enumerate(Kgrid):
            x = math.log(K/F)
            Z[i,j] = max(atm_vol*(1+vov*math.sqrt(T)) + skew*x + curv*x**2, 0.01)
    return mats, Kgrid, Z


def sd_dataset_real(commodity: str, hist_df: pd.DataFrame, months: int = 36) -> pd.DataFrame:
    """
    Build S&D balance dataset.
    If real price history is available, use it for the price column.
    Supply/demand/stocks are model-driven (no public API for these).
    """
    c    = COMMODITIES[commodity]
    spot = c["fallback"]
    rng  = np.random.default_rng(hash(commodity) % 2**32)
    today = date.today().replace(day=1)
    dates = [today - timedelta(days=30*(months-i)) for i in range(months)]
    hist_n = int(months * 0.65)
    sup  = spot * (1 + rng.normal(0, 0.02, months)).cumprod()
    dem  = spot * (1 + rng.normal(0, 0.018, months)).cumprod()
    sto  = np.zeros(months)
    sto[0] = spot * 0.15
    for i in range(1, months):
        sto[i] = max(sto[i-1] + (sup[i]-dem[i])*0.05, spot*0.03)
    fv = [spot*(sto[j]/sto[0])**(-0.3)*(dem[j]/dem[0])**0.5 for j in range(months)]
    is_fc = [j >= hist_n for j in range(months)]

    # Use real prices where available
    prices = []
    for d_ in dates:
        if not hist_df.empty:
            mask = hist_df.index.date <= d_
            if mask.any():
                prices.append(float(hist_df["Close"][mask].iloc[-1]))
            else:
                prices.append(spot)
        else:
            prices.append(spot * math.exp(rng.normal(-0.002, c["vol"]/12)))

    return pd.DataFrame(dict(date=dates, supply=sup, demand=dem,
                             stocks=sto, price=prices, fair_value=fv,
                             surplus=sup-dem, is_forecast=is_fc)).set_index("date")


def run_mc(spot, vol, n_paths=500, horizon=18):
    rng   = np.random.default_rng(0)
    dt    = 1/12
    paths = np.zeros((n_paths, horizon+1))
    paths[:,0] = spot
    for t in range(1, horizon+1):
        z = rng.standard_normal(n_paths)
        paths[:,t] = paths[:,t-1] * np.exp((-0.5*vol**2)*dt + vol*math.sqrt(dt)*z)
    fan_dates = [date.today() + timedelta(days=30*i) for i in range(horizon+1)]
    pcts = np.percentile(paths, [5,25,50,75,95], axis=0)
    fan  = pd.DataFrame(dict(date=fan_dates, p5=pcts[0], p25=pcts[1],
                             p50=pcts[2], p75=pcts[3], p95=pcts[4]))
    hist_bins = np.histogram(paths[:,-1], bins=40)
    return dict(fan=fan,
                median=float(np.median(paths[:,-1])),
                p5=float(np.percentile(paths[:,-1],5)),
                p95=float(np.percentile(paths[:,-1],95)),
                hist_x=hist_bins[1][:-1].tolist(),
                hist_y=hist_bins[0].tolist())


def portfolio_var(positions, live_prices, conf=0.95, horizon=1):
    z = norm.ppf(conf)
    rows, total_var, total_cvar = [], 0.0, 0.0
    for p in positions:
        c    = COMMODITIES.get(p["commodity"], {})
        vol  = c.get("vol", 0.30)
        spot = live_prices.get(p["commodity"], c.get("fallback", 100))
        sign = 1 if p.get("side") == "Long" else -1
        notional  = spot * p.get("quantity", 0) * sign
        daily_vol = vol / math.sqrt(252)
        var  = abs(notional)*daily_vol*z*math.sqrt(horizon)
        cvar = abs(notional)*daily_vol*norm.pdf(z)/(1-conf)*math.sqrt(horizon)
        total_var  += var
        total_cvar += cvar
        rows.append(dict(commodity=p["commodity"], side=p.get("side"),
                         quantity=p.get("quantity"), spot=round(spot,4),
                         vol_pct=vol*100, var=var, cvar=cvar))
    return dict(total_var=total_var, total_cvar=total_cvar, rows=rows)


def macro_data(country, months=48):
    rng   = np.random.default_rng(hash(country) % 2**32)
    today = date.today().replace(day=1)
    dates = [today - timedelta(days=30*(months-i)) for i in range(months)]
    base  = dict(USA=dict(gdp=100,cpi=3.2,rate=5.25,pmi=52.1),
                 China=dict(gdp=100,cpi=0.8,rate=3.45,pmi=50.4),
                 Germany=dict(gdp=100,cpi=2.9,rate=3.50,pmi=47.2),
                 Japan=dict(gdp=100,cpi=2.5,rate=0.50,pmi=49.8),
                 UK=dict(gdp=100,cpi=3.4,rate=5.25,pmi=51.3),
                 Brazil=dict(gdp=100,cpi=4.8,rate=10.75,pmi=50.9),
                 India=dict(gdp=100,cpi=4.5,rate=6.50,pmi=57.5),
                 France=dict(gdp=100,cpi=2.7,rate=3.50,pmi=48.6))
    b = base.get(country, dict(gdp=100,cpi=2.5,rate=4.0,pmi=50.0))
    gdp  = b["gdp"] + np.cumsum(rng.normal(0.5,0.3,months))
    cpi  = b["cpi"] + np.cumsum(rng.normal(0.0,0.08,months))
    rate = b["rate"] + np.cumsum(rng.normal(0.0,0.05,months))
    pmi  = b["pmi"] + rng.normal(0,1.2,months)
    return pd.DataFrame(dict(date=dates, gdp_index=gdp,
                             cpi_yoy=np.clip(cpi,-2,20),
                             policy_rate=np.clip(rate,0,20),
                             pmi=np.clip(pmi,30,70))).set_index("date")

# ══════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _styled(fig, h=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=10,t=30,b=10), height=h,
        font=dict(family="Inter,system-ui", size=12, color=TEXT),
        legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor=BORDER, borderwidth=1),
    )
    fig.update_xaxes(gridcolor=PANEL, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=PANEL, zerolinecolor=BORDER)
    return fig


def kpi(label, value, sub="", accent=AMBER):
    return (f'<div class="kpi-card" style="border-left-color:{accent}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(live_prices):
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
        "📊 Dashboard","⚖️ Supply & Demand","🌍 Regional Flows",
        "📈 Futures Curve","🎯 Options & Greeks","📉 Vol Surface",
        "💼 Positions & P&L","🛡️ Risk","🎲 Monte Carlo",
        "🌐 Macro Overlay","📅 Events","ℹ️ About",
    ]
    page = st.sidebar.radio("Pages", pages, label_visibility="collapsed")
    st.sidebar.markdown("---")

    st.sidebar.markdown(
        f'<div style="font-size:10px;color:{GRAY};font-family:JetBrains Mono,monospace;'
        f'letter-spacing:0.15em;text-transform:uppercase;margin-bottom:6px;">Select Commodity</div>',
        unsafe_allow_html=True,
    )
    sector    = st.sidebar.selectbox("Sector", ALL_SECTORS, key="sidebar_sector")
    names_in  = [k for k,v in COMMODITIES.items() if v["sector"]==sector]
    commodity = st.sidebar.selectbox("Commodity", names_in, key="sidebar_commodity")

    c = COMMODITIES[commodity]
    spot = live_prices.get(commodity, c["fallback"])

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div style="font-size:10px;color:{GRAY};font-family:JetBrains Mono,monospace;'
        f'letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;">Option Parameters</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.slider("Maturity T (months)", 1, 36, 6,  key="opt_T_months")
    st.sidebar.slider("Strike %F",          70,130,100, key="opt_K_pct")
    st.sidebar.slider("Volatility σ %",  5, 120, int(c["vol"]*100), key="opt_vol_pct")
    st.sidebar.slider("Rate r %",           0,  10,  5, key="opt_r_pct")
    st.sidebar.slider("Curve months",       3,  36, 18, key="curve_months")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    src = "yfinance" if (YF_AVAILABLE and c.get("yf_ticker")) else "fallback"
    st.sidebar.markdown(
        f'<div style="font-size:9px;color:{GRAY};font-family:JetBrains Mono,monospace;'
        f'letter-spacing:0.08em;text-transform:uppercase;margin-top:20px;">'
        f'{now}<br>by Adam EL GBOURI · {date.today().year}<br>'
        f'aeg-snd.streamlit.app<br>'
        f'<span style="color:{GREEN if src=="yfinance" else AMBER};">data: {src}</span></div>',
        unsafe_allow_html=True,
    )
    return page, commodity, spot

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
def render_header(page, commodity, spot):
    c = COMMODITIES[commodity]
    col1, col2 = st.columns([5,2])
    with col1:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">'
            f'<div style="width:36px;height:36px;border-radius:9px;'
            f'background:linear-gradient(135deg,{AMBER},{TEAL});display:flex;'
            f'align-items:center;justify-content:center;color:{BG};font-weight:800;font-size:17px;">C</div>'
            f'<div><div style="font-size:19px;font-weight:700;color:{TEXT};">Commodity Trading Desk</div>'
            f'<div style="font-size:9px;color:{GRAY};font-family:JetBrains Mono,monospace;'
            f'letter-spacing:0.18em;text-transform:uppercase;">by Adam EL GBOURI · {date.today().year}</div>'
            f'</div></div>', unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div style="text-align:right;padding-top:10px;color:{GRAY};'
            f'font-size:11px;font-family:JetBrains Mono,monospace;">'
            f'{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>',
            unsafe_allow_html=True,
        )
    src_badge = f'<span class="badge" style="color:{GREEN};border-color:rgba(63,185,80,0.4);">LIVE yfinance</span>' \
                if (YF_AVAILABLE and COMMODITIES[commodity].get("yf_ticker")) \
                else f'<span class="badge" style="color:{AMBER};border-color:rgba(240,165,0,0.4);">FALLBACK</span>'
    st.markdown(
        f'{src_badge}'
        f'<span class="badge">{c["sector"]}</span>'
        f'<span class="badge">{c["ticker"]}</span>'
        f'<span class="badge">{commodity}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard(commodity, live_prices):
    c    = COMMODITIES[commodity]
    spot = live_prices.get(commodity, c["fallback"])

    # Real 1-day change from history
    hist = fetch_history(c.get("yf_ticker",""), period="5d")
    if not hist.empty and len(hist) >= 2:
        prev  = float(hist["Close"].iloc[-2])
        chg   = (spot - prev) / prev * 100
        chg_label = "vs prev close (real)"
    else:
        chg, chg_label = 0.0, "N/A"

    fv     = spot * 0.97   # simple fair value placeholder
    fv_dev = (spot - fv) / fv * 100
    direction = "up" if chg > 0 else "down" if chg < 0 else "flat"
    fv_flag   = "rich" if fv_dev > 12 else "cheap" if fv_dev < -12 else "fairly priced"

    st.info(
        f"**{commodity}** is {direction} **{chg:+.2f}%** at `{spot:,.2f} {c['unit']}` ({chg_label}). "
        f"Fair-value proxy: `{fv:,.2f}` → **{fv_flag}** ({fv_dev:+.1f}% vs spot). "
        f"Implied vol **{c['vol']*100:.0f}%**.",
        icon="🎯",
    )

    cols = st.columns(5)
    kpi_data = [
        ("Spot Price",  f"{spot:,.2f} {c['unit']}", f"{chg:+.2f}% 1D",    AMBER),
        ("Fair Value",  f"{fv:,.2f}",               f"{fv_dev:+.1f}% dev", BLUE),
        ("Impl. Vol",   f"{c['vol']*100:.1f}%",     "annualised σ",        PURPLE),
        ("Storage",     f"{c['storage']*100:.1f}%", "annual cost",         GRAY),
        ("Conv. Yield", f"{c['conv']*100:.1f}%",    "annual yield",        TEAL),
    ]
    for col, (lbl,val,sub,acc) in zip(cols, kpi_data):
        col.markdown(kpi(lbl,val,sub,acc), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Real price history chart ──────────────────────────────────────────────
    st.subheader("Price History (Real)")
    st.caption("📈 Real daily closing prices from Yahoo Finance (continuous front-month futures contract). Dashed line = simple fair-value proxy (spot × 0.97).")
    hist1y = fetch_history(c.get("yf_ticker",""), period="2y")
    if not hist1y.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist1y.index, y=hist1y["Close"],
                                 name="Close", line=dict(color=AMBER, width=2)))
        fig.add_trace(go.Scatter(x=hist1y.index, y=hist1y["Close"]*0.97,
                                 name="Fair Value proxy",
                                 line=dict(color=BLUE, width=1.5, dash="dot")))
        st.plotly_chart(_styled(fig, 360), use_container_width=True)
    else:
        st.info("No price history available for this commodity via Yahoo Finance.")

    # ── Market heatmap with real data + date comparison ───────────────────────
    st.subheader("Market Heatmap — Date Comparison")
    st.caption("🟢🔴 Compare real closing prices between two dates. Green = price rose, red = fell. Select dates below.")

    col_d1, col_t1, col_d2, col_t2 = st.columns(4)
    date_a = col_d1.date_input("Date A", value=date.today()-timedelta(days=30), key="hm_date_a")
    date_b = col_d2.date_input("Date B", value=date.today(),                    key="hm_date_b")

    if st.button("🔄 Load Heatmap Data", type="primary"):
        st.session_state["hm_loaded"] = True

    if st.session_state.get("hm_loaded"):
        with st.spinner("Fetching prices for both dates…"):
            rows_hm = []
            for name, info in COMMODITIES.items():
                yf_t = info.get("yf_ticker")
                p_a  = fetch_price_at_date(yf_t, date_a) if yf_t else None
                p_b  = fetch_price_at_date(yf_t, date_b) if yf_t else None
                if p_a and p_b and p_a > 0:
                    chg_ = (p_b - p_a) / p_a * 100
                    src  = "real"
                else:
                    p_b  = live_prices.get(name, info["fallback"])
                    chg_ = 0.0
                    src  = "fallback"
                chg_str = f"{chg_:+.2f}%"
                rows_hm.append(dict(name=name, sector=info["sector"],
                                    spot=round(p_b,2), change=chg_,
                                    change_str=chg_str, source=src))

        sp_df = pd.DataFrame(rows_hm)
        fig_hm = px.treemap(
            sp_df, path=[px.Constant("Markets"), "sector", "name"],
            values=[1]*len(sp_df), color="change",
            color_continuous_scale=[(0,RED),(0.5,PANEL),(1,GREEN)],
            color_continuous_midpoint=0,
            custom_data=["spot","change_str","source"],
        )
        fig_hm.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[0]:.2f}<br>%{customdata[1]}",
            hovertemplate="<b>%{label}</b><br>Price: %{customdata[0]:.2f}<br>"
                          "Chg: %{customdata[1]}<br>Source: %{customdata[2]}<extra></extra>",
        )
        st.plotly_chart(_styled(fig_hm, 480), use_container_width=True)
        n_real = sum(1 for r in rows_hm if r["source"]=="real")
        st.caption(f"✅ {n_real}/{len(rows_hm)} commodities have real data for both dates. "
                   f"Others show fallback price with 0.00% change.")
    else:
        st.info("Select two dates above and click **Load Heatmap Data** to compare real prices.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: SUPPLY & DEMAND
# ══════════════════════════════════════════════════════════════════════════════
def page_balance(commodity, live_prices):
    st.title(f"Supply & Demand — {commodity}")
    st.caption("⚖️ **What this page does:** Real price history anchors the price series. Supply, demand and stocks are model-driven (no public fundamental API). Use the sliders to stress-test scenarios.")
    c    = COMMODITIES[commodity]
    spot = live_prices.get(commodity, c["fallback"])

    c1,c2,c3,c4 = st.columns(4)
    sup_adj = c1.slider("Supply adj %",   -20, 20, 0, 1, key="bal_sup")
    dem_adj = c2.slider("Demand adj %",   -20, 20, 0, 1, key="bal_dem")
    gdp     = c3.slider("GDP growth %", -2.0,6.0,2.5,0.1, key="bal_gdp")
    horizon = c4.slider("Forecast months", 6, 36, 18, 3, key="bal_h")

    hist = fetch_history(c.get("yf_ticker",""), period="3y")
    df   = sd_dataset_real(commodity, hist, horizon+12)
    df["supply"]  *= 1 + sup_adj/100
    df["demand"]  *= 1 + dem_adj/100
    df["surplus"]  = df["supply"] - df["demand"]

    last = df.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("Live Spot",       f"{spot:,.2f} {c['unit']}")
    cols[1].metric("Avg surplus/def", f"{df['surplus'].mean():+.2f}")
    cols[2].metric("Fair value",      f"{last['fair_value']:,.2f}")
    cols[3].metric("GDP assumption",  f"{gdp:+.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["supply"], name="Supply",
                             line=dict(color=GREEN,width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["demand"], name="Demand",
                             line=dict(color=RED,width=2)))
    fig.add_trace(go.Bar(x=df.index, y=df["surplus"], name="Surplus/Deficit",
                         marker_color=np.where(df["surplus"]>=0, GREEN, RED),
                         opacity=0.55, yaxis="y2"))
    fig.update_layout(yaxis2=dict(overlaying="y",side="right",showgrid=False))
    st.plotly_chart(_styled(fig,380), use_container_width=True)

    st.subheader("Real Price vs Fair Value Model")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["price"], name="Real Price",
                              line=dict(color=TEXT,width=2)))
    fig2.add_trace(go.Scatter(x=df.index, y=df["fair_value"], name="Fair Value",
                              line=dict(color=AMBER,width=2,dash="dot")))
    st.plotly_chart(_styled(fig2,300), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: REGIONAL FLOWS
# ══════════════════════════════════════════════════════════════════════════════
def page_regional(commodity, live_prices):
    st.title(f"Regional Flows — {commodity}")
    st.caption("🌍 **What this page does:** Regional supply/demand fundamentals (IEA/USDA-style). Data is based on published estimates — not real-time. Green = net exporter, red = net importer.")
    key = commodity if commodity in REGIONAL_DATA else list(REGIONAL_DATA.keys())[0]
    reg = pd.DataFrame(REGIONAL_DATA[key])
    reg["net"]    = reg["supply"] - reg["demand"]
    reg["status"] = np.where(reg["net"]>0, "Exporter", "Importer")

    ws,wd = float(reg["supply"].sum()), float(reg["demand"].sum())
    cols = st.columns(4)
    cols[0].metric("World Supply", f"{ws:,.1f} {COMMODITIES[commodity]['unit']}")
    cols[1].metric("World Demand", f"{wd:,.1f} {COMMODITIES[commodity]['unit']}")
    cols[2].metric("Balance",      f"{ws-wd:+,.2f}", "surplus" if ws>wd else "deficit")
    cols[3].metric("Regions",      str(len(reg)))

    fig_map = go.Figure()
    for _, r in reg.iterrows():
        color = GREEN if r["net"]>=0 else RED
        fig_map.add_trace(go.Scattergeo(
            lat=[r["lat"]], lon=[r["lon"]], mode="markers+text",
            marker=dict(size=abs(r["net"])**0.5*4+8, color=color, opacity=0.75,
                        line=dict(color=BORDER,width=1)),
            text=r["region"], textposition="top center",
            textfont=dict(size=10,color=TEXT,family="JetBrains Mono"),
            name=r["region"],
            hovertemplate=f"<b>{r['region']}</b><br>Supply:{r['supply']:.1f}<br>"
                          f"Demand:{r['demand']:.1f}<br>Net:{r['net']:+.1f}<extra></extra>",
        ))
    fig_map.update_layout(
        geo=dict(bgcolor=BG,showframe=False,showcoastlines=True,
                 coastlinecolor=BORDER,landcolor=PANEL,oceancolor=BG,
                 showocean=True,showland=True,projection_type="natural earth"),
        paper_bgcolor="rgba(0,0,0,0)", height=420, margin=dict(l=0,r=0,t=0,b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    fig2 = go.Figure(go.Bar(
        x=reg["region"], y=reg["net"],
        marker_color=np.where(reg["net"]>=0, GREEN, RED),
        text=[f"{v:+.1f}" for v in reg["net"]], textposition="outside",
    ))
    fig2.update_layout(title="Net Trade (Supply − Demand)")
    st.plotly_chart(_styled(fig2,300), use_container_width=True)
    st.dataframe(reg.style.format({"supply":"{:,.2f}","demand":"{:,.2f}","net":"{:+,.2f}"}),
                 use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FUTURES CURVE
# ══════════════════════════════════════════════════════════════════════════════
def page_curve(commodity, live_prices):
    st.title(f"Futures Curve — {commodity}")
    st.caption("📈 **What this page does:** Cost-of-carry forward curve anchored to the real live spot price. *Contango* = upward-sloping (storage cost > convenience yield). *Backwardation* = downward-sloping (supply tight).")
    c     = COMMODITIES[commodity]
    spot  = live_prices.get(commodity, c["fallback"])
    mnths = st.session_state.get("curve_months",18)
    r     = st.session_state.get("opt_r_pct",5) / 100

    rng = np.random.default_rng(42)
    rows = []
    today = date.today()
    for i in range(1, mnths+1):
        T = i/12
        F = spot * math.exp((r + c["storage"] - c["conv"])*T)
        F += rng.normal(0, c["vol"]*math.sqrt(T/12)*0.3)*spot*0.01
        label = (today.replace(day=1)+timedelta(days=32*i)).strftime("%b %y")
        rows.append(dict(month=i, label=label, T=T, price=F))
    curve = pd.DataFrame(rows)

    front = float(curve["price"].iloc[0])
    back  = float(curve["price"].iloc[-1])
    structure = ("CONTANGO" if back>front*1.005
                 else "BACKWARDATION" if back<front*0.995 else "FLAT")
    s_color = RED if structure=="CONTANGO" else GREEN if structure=="BACKWARDATION" else AMBER

    cols = st.columns(4)
    cols[0].metric("Live Spot",    f"{spot:,.2f} {c['unit']}")
    cols[1].metric("Front Month",  f"{front:,.2f}")
    cols[2].metric("12M Forward",  f"{float(curve[curve['month']<=12]['price'].iloc[-1]):,.2f}")
    cols[3].metric("Structure",    structure, f"{(back-front)/front*100:+.2f}%")
    st.markdown(f'<span class="badge" style="border-color:{s_color};color:{s_color};">⚡ {structure}</span>',
                unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["label"], y=curve["price"],
                             mode="lines+markers", line=dict(color=AMBER,width=2.5),
                             marker=dict(size=7,color=AMBER), name="Forward curve"))
    fig.add_trace(go.Scatter(x=curve["label"], y=[spot]*len(curve),
                             mode="lines", name="Live Spot",
                             line=dict(color=TEXT,dash="dot",width=1.5)))
    fig.update_layout(title=f"{commodity} Forward Curve (anchored to live spot)")
    st.plotly_chart(_styled(fig,380), use_container_width=True)
    st.dataframe(curve[["label","T","price"]].rename(
        columns={"label":"Contract","T":"Maturity (yr)","price":"Price"}),
        use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: OPTIONS & GREEKS
# ══════════════════════════════════════════════════════════════════════════════
def page_options(commodity, live_prices):
    st.title(f"Options & Greeks — {commodity}")
    st.caption("🎯 **What this page does:** Black-76 pricer using the real live spot as the default forward. Greeks measure sensitivities: Delta per $1 move; Vega per 1% vol; Theta per day.")
    c    = COMMODITIES[commodity]
    spot = live_prices.get(commodity, c["fallback"])

    T_months = st.session_state.get("opt_T_months",6)
    K_pct    = st.session_state.get("opt_K_pct",100)
    vol_pct  = st.session_state.get("opt_vol_pct",int(c["vol"]*100))
    r_pct    = st.session_state.get("opt_r_pct",5)

    col1,col2 = st.columns(2)
    F = col1.number_input("Forward F (live spot default)", value=spot, step=spot*0.005)
    K = col2.number_input("Strike K", value=spot*K_pct/100, step=spot*0.005)

    T,sigma,r = T_months/12, vol_pct/100, r_pct/100
    call = black76(F,K,T,r,sigma,"call")
    put  = black76(F,K,T,r,sigma,"put")
    moneyness = "ITM" if F>K else "OTM" if F<K else "ATM"
    pcp = call["price"]-put["price"]-math.exp(-r*T)*(F-K)

    st.markdown(
        f'<span class="badge badge-amber">{moneyness}</span>'
        f'<span class="badge">T={T:.3f}yr</span>'
        f'<span class="badge">σ={sigma*100:.1f}%</span>'
        f'<span class="badge" style="color:{GREEN};">PCP err={pcp:.2e}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    tab_c,tab_p = st.tabs(["📞 Call","📤 Put"])
    greek_labels = ["Price","Delta","Gamma","Vega","Theta","Rho"]
    greek_keys   = ["price","delta","gamma","vega","theta","rho"]
    for tab,g,acc in [(tab_c,call,AMBER),(tab_p,put,BLUE)]:
        with tab:
            cols = st.columns(6)
            for col,lbl,k in zip(cols,greek_labels,greek_keys):
                col.markdown(kpi(lbl,f"{g[k]:.5f}","",acc), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    strikes  = np.linspace(F*0.55,F*1.45,80)
    call_pnl = np.maximum(strikes-K,0)-call["price"]
    put_pnl  = np.maximum(K-strikes,0)-put["price"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes,y=call_pnl,name="Long Call",line=dict(color=GREEN,width=2.5)))
    fig.add_trace(go.Scatter(x=strikes,y=put_pnl, name="Long Put", line=dict(color=RED,  width=2.5)))
    fig.add_hline(y=0, line=dict(color=GRAY,dash="dash",width=1))
    fig.add_vline(x=K, line=dict(color=AMBER,dash="dot",width=1.5), annotation_text="Strike")
    fig.add_vline(x=F, line=dict(color=BLUE, dash="dot",width=1.5), annotation_text="Forward")
    fig.update_layout(title="Payoff at Expiry (net of premium)")
    st.plotly_chart(_styled(fig,360), use_container_width=True)

    st.subheader("Greeks vs Strike")
    ks = np.linspace(F*0.7,F*1.3,60)
    tab1,tab2,tab3 = st.tabs(["Delta","Gamma","Vega"])
    for tab,gk,col_ in [(tab1,"delta",AMBER),(tab2,"gamma",PURPLE),(tab3,"vega",TEAL)]:
        with tab:
            vc = [black76(F,k,T,r,sigma,"call")[gk] for k in ks]
            vp = [black76(F,k,T,r,sigma,"put")[gk]  for k in ks]
            fig_ = go.Figure()
            fig_.add_trace(go.Scatter(x=ks,y=vc,name=f"Call {gk}",line=dict(color=col_,width=2)))
            fig_.add_trace(go.Scatter(x=ks,y=vp,name=f"Put {gk}", line=dict(color=BLUE, width=2)))
            fig_.add_vline(x=K, line=dict(color=AMBER,dash="dot"))
            st.plotly_chart(_styled(fig_,280), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: VOL SURFACE
# ══════════════════════════════════════════════════════════════════════════════
def page_vol_surface(commodity, live_prices):
    st.title(f"Implied Vol Surface — {commodity}")
    st.caption("📉 **What this page does:** Parametric vol surface anchored to the real live spot. ATM vol is seeded from the commodity's known historical vol. Skew/curvature/vol-of-vol are adjustable.")
    c    = COMMODITIES[commodity]
    spot = live_prices.get(commodity, c["fallback"])

    col1,col2,col3,col4 = st.columns(4)
    atm  = col1.slider("ATM vol %",    5,120,int(c["vol"]*100),key="vs_atm") /100
    skew = col2.slider("Skew ×100",  -20, 20,-5,               key="vs_skew")/100
    curv = col3.slider("Curv ×100",    0, 10, 2,               key="vs_curv")/100
    vov  = col4.slider("Vol-of-vol",   0,100,15,               key="vs_vov") /100

    mats,Kgrid,Z = vol_surface_fn(spot,atm,skew,curv,vov)
    mat_labels = ["1M","2M","3M","6M","9M","12M","18M","24M"]

    fig = go.Figure(data=go.Surface(
        z=Z, x=np.log(Kgrid/spot), y=[m*12 for m in mats],
        colorscale=[[0,BLUE],[0.5,PURPLE],[1,AMBER]],
        showscale=True, colorbar=dict(title="σ",tickfont=dict(color=TEXT)),
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="ln(K/F)",color=GRAY,gridcolor=BORDER,backgroundcolor=BG),
            yaxis=dict(title="Maturity (months)",color=GRAY,gridcolor=BORDER,backgroundcolor=BG),
            zaxis=dict(title="Implied Vol",color=GRAY,gridcolor=BORDER,backgroundcolor=BG),
            bgcolor=BG,
        ),
        paper_bgcolor="rgba(0,0,0,0)", height=520,
        margin=dict(l=10,r=10,t=10,b=10), font=dict(color=TEXT),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Vol Smile by Maturity")
    fig2 = go.Figure()
    pal = [AMBER,BLUE,GREEN,RED,PURPLE,TEAL,GRAY,TEXT]
    for row,label,col_ in zip(Z,mat_labels,pal):
        fig2.add_trace(go.Scatter(x=np.log(Kgrid/spot),y=row*100,
                                  name=label,line=dict(color=col_,width=2)))
    fig2.update_layout(xaxis_title="ln(K/F)",yaxis_title="Impl. Vol %")
    st.plotly_chart(_styled(fig2,360), use_container_width=True)

    atm_ts = Z[:,Z.shape[1]//2]*100
    fig3 = go.Figure(go.Bar(x=mat_labels,y=atm_ts,marker_color=AMBER))
    fig3.update_layout(title="ATM Vol Term Structure (%)",yaxis_title="σ %")
    st.plotly_chart(_styled(fig3,260), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
def page_positions(live_prices):
    st.title("Positions & P&L")
    st.caption("💼 **What this page does:** Trade blotter marked to real live prices. Long = profit if price rises. Short = profit if price falls.")

    if "positions" not in st.session_state:
        st.session_state["positions"] = []

    with st.expander("➕ Add Position", expanded=True):
        c1,c2,c3,c4,c5 = st.columns(5)
        name  = c1.selectbox("Commodity", list(COMMODITIES.keys()), key="pos_name")
        side  = c2.selectbox("Side", ["Long","Short"], key="pos_side")
        qty   = c3.number_input("Quantity", value=100, step=10, key="pos_qty")
        default_entry = live_prices.get(name, COMMODITIES[name]["fallback"])
        entry = c4.number_input("Entry Price", value=default_entry, step=1.0, key="pos_entry")
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
        mark = live_prices.get(p["commodity"], COMMODITIES[p["commodity"]]["fallback"])
        sign = 1 if p["side"]=="Long" else -1
        pnl  = sign*(mark-p["entry_price"])*p["quantity"]
        total_pnl += pnl
        if p["side"]=="Long": total_long  += mark*p["quantity"]
        else:                 total_short += mark*p["quantity"]
        rows.append(dict(
            Commodity=p["commodity"], Side=p["side"],
            Qty=p["quantity"], Entry=p["entry_price"], Mark=round(mark,4),
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
        if isinstance(val,(int,float)): return f"color:{GREEN}" if val>=0 else f"color:{RED}"
        return ""
    st.dataframe(
        df.style
          .format({"Entry":"{:.2f}","Mark":"{:.4f}",
                   "P&L/unit":"{:+.4f}","P&L Total":"{:+,.0f}","Return %":"{:+.2f}%"})
          .map(_color_pnl, subset=["P&L/unit","P&L Total","Return %"]),
        use_container_width=True,
    )
    if st.button("🗑️ Clear All Positions"):
        st.session_state["positions"] = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RISK
# ══════════════════════════════════════════════════════════════════════════════
def page_risk(live_prices):
    st.title("Risk Dashboard")
    st.caption("🛡️ **What this page does:** Parametric VaR/CVaR using real live spot prices and historical vols. Stress scenarios show P&L impact of ±5% to ±30% price shocks.")
    positions = st.session_state.get("positions",[])
    if not positions:
        st.warning("Add positions on the Positions page first.")
        return

    col1,col2 = st.columns(2)
    conf    = col1.selectbox("Confidence",[0.90,0.95,0.99],index=1)
    horizon = col2.slider("Horizon (days)",1,30,1,key="risk_h")

    risk = portfolio_var(positions, live_prices, conf=conf, horizon=horizon)
    cols = st.columns(3)
    cols[0].metric(f"VaR {int(conf*100)}%",  f"${risk['total_var']:,.0f}",  f"{horizon}d horizon")
    cols[1].metric(f"CVaR {int(conf*100)}%", f"${risk['total_cvar']:,.0f}", "Expected shortfall")
    cols[2].metric("Positions", str(len(positions)))

    st.subheader("Per-Position Decomposition")
    rdf = pd.DataFrame(risk["rows"])
    st.dataframe(rdf.style.format({"vol_pct":"{:.1f}%","var":"${:,.0f}",
                                   "cvar":"${:,.0f}","spot":"{:.4f}","quantity":"{:.0f}"}),
                 use_container_width=True)

    p   = positions[0]
    base = live_prices.get(p["commodity"], COMMODITIES[p["commodity"]]["fallback"])
    sign = 1 if p["side"]=="Long" else -1
    sc_rows = []
    for sh in [-30,-20,-10,-5,5,10,20,30]:
        new_p = base*(1+sh/100)
        sc_rows.append(dict(shock_pct=sh, new_price=new_p,
                            pnl_impact=sign*(new_p-p["entry_price"])*p["quantity"]))
    sdf = pd.DataFrame(sc_rows)
    st.subheader(f"Stress Scenarios — {p['commodity']} (live spot: {base:,.2f})")
    fig = go.Figure(go.Bar(
        x=[f"{r['shock_pct']:+.0f}%" for _,r in sdf.iterrows()],
        y=sdf["pnl_impact"],
        marker_color=np.where(sdf["pnl_impact"]>=0,GREEN,RED),
        text=[f"${v:+,.0f}" for v in sdf["pnl_impact"]], textposition="outside",
    ))
    st.plotly_chart(_styled(fig,300), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════
def page_mc(commodity, live_prices):
    st.title(f"Monte Carlo — {commodity}")
    st.caption("🎲 **What this page does:** GBM simulation starting from the real live spot price. The fan chart shows the distribution of future price outcomes. Wider = more uncertain.")
    c    = COMMODITIES[commodity]
    spot = live_prices.get(commodity, c["fallback"])

    col1,col2,col3,col4 = st.columns(4)
    n_paths = col1.slider("Paths",         100,2000,500,50, key="mc_n")
    vol_pct = col2.slider("Vol override %", 5, 150, int(c["vol"]*100), key="mc_vol")
    horizon = col3.slider("Horizon months", 3,  36, 18, 3, key="mc_h")
    st.markdown(f"**Starting from live spot: {spot:,.2f} {c['unit']}**")

    with st.spinner(f"Running {n_paths:,} GBM paths from live spot {spot:,.2f}…"):
        res = run_mc(spot, vol_pct/100, n_paths, horizon)

    cols = st.columns(4)
    cols[0].metric("Live Spot (t=0)",  f"{spot:,.2f} {c['unit']}")
    cols[1].metric("Median at horizon",f"{res['median']:,.2f}")
    cols[2].metric("P5  (bear)",       f"{res['p5']:,.2f}")
    cols[3].metric("P95 (bull)",       f"{res['p95']:,.2f}")

    fan = res["fan"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fan["date"],y=fan["p95"],name="P95",
                             line=dict(color=GREEN,width=1,dash="dot")))
    fig.add_trace(go.Scatter(x=fan["date"],y=fan["p75"],name="P75",
                             line=dict(color=GREEN,width=0.8),
                             fill="tonexty",fillcolor="rgba(63,185,80,0.08)"))
    fig.add_trace(go.Scatter(x=fan["date"],y=fan["p50"],name="P50 Median",
                             line=dict(color=AMBER,width=2.5),
                             fill="tonexty",fillcolor="rgba(240,165,0,0.06)"))
    fig.add_trace(go.Scatter(x=fan["date"],y=fan["p25"],name="P25",
                             line=dict(color=RED,width=0.8),
                             fill="tonexty",fillcolor="rgba(255,123,114,0.06)"))
    fig.add_trace(go.Scatter(x=fan["date"],y=fan["p5"],name="P5",
                             line=dict(color=RED,width=1,dash="dot"),
                             fill="tonexty",fillcolor="rgba(255,123,114,0.08)"))
    fig.update_layout(title=f"Price Fan Chart — starting from live spot {spot:,.2f}")
    st.plotly_chart(_styled(fig,420), use_container_width=True)

    fig_h = go.Figure(go.Bar(x=res["hist_x"],y=res["hist_y"],
                             marker_color=AMBER,opacity=0.75,name="Frequency"))
    fig_h.add_vline(x=res["median"],line=dict(color=TEXT, dash="dash"),annotation_text="Median")
    fig_h.add_vline(x=res["p5"],    line=dict(color=RED,  dash="dot"), annotation_text="P5")
    fig_h.add_vline(x=res["p95"],   line=dict(color=GREEN,dash="dot"), annotation_text="P95")
    fig_h.add_vline(x=spot,         line=dict(color=AMBER,dash="dot"), annotation_text="Live Spot")
    st.plotly_chart(_styled(fig_h,280), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MACRO OVERLAY
# ══════════════════════════════════════════════════════════════════════════════
def page_macro():
    st.title("Macro Overlay")
    st.caption("🌐 **What this page does:** Macro indicators (GDP, CPI, policy rate, PMI) by country. Data is model-estimated — no live macro API. Use to contextualise commodity price moves.")
    col1,col2 = st.columns([2,5])
    primary = col1.selectbox("Country", COUNTRIES, key="macro_primary")
    compare = col2.multiselect("Compare with",
                               [c for c in COUNTRIES if c!=primary],
                               default=[c for c in COUNTRIES if c!=primary][:2],
                               key="macro_cmp")
    series = [(primary, macro_data(primary))]
    series.extend((c,macro_data(c)) for c in compare)

    snap = series[0][1].iloc[-1]
    cols = st.columns(4)
    for col,m,lab in zip(cols,
                          ["gdp_index","cpi_yoy","policy_rate","pmi"],
                          ["GDP Index","CPI YoY %","Policy Rate %","PMI"]):
        cols[cols.index(col)].metric(lab,f"{snap[m]:.2f}")

    metrics = dict(gdp_index="GDP Index",cpi_yoy="CPI YoY %",
                   policy_rate="Policy Rate %",pmi="PMI")
    tabs = st.tabs(list(metrics.values()))
    for tab,(mk,ml) in zip(tabs,metrics.items()):
        with tab:
            fig = go.Figure()
            pal = [AMBER,BLUE,GREEN,RED]
            for i,(name,df) in enumerate(series):
                if mk in df.columns:
                    fig.add_trace(go.Scatter(x=df.index,y=df[mk],name=name,
                                            line=dict(color=pal[i%len(pal)],width=2)))
            fig.update_layout(title=ml,yaxis_title=ml)
            st.plotly_chart(_styled(fig,360), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: EVENTS
# ══════════════════════════════════════════════════════════════════════════════
def page_events():
    st.title("Market Events Calendar")
    st.caption("📅 **What this page does:** Upcoming market-moving events — EIA, WASDE, OPEC, FOMC, IEA. These drive commodity volatility. Knowing when they're coming helps avoid being caught offside.")
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
        return (["background-color:#1a2744"]*len(r) if today_mask[int(r.name)]
                else [""]*len(r))
    st.dataframe(display_df.style.apply(_style,axis=1),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
def page_about():
    st.title("About")
    yf_status = ("✅ yfinance " + yf.__version__ + " installed — real data active") if YF_AVAILABLE                 else "⚠️ yfinance not installed — run `pip install yfinance` to enable real data"
    st.markdown(
        "**Commodity Trading Desk** — standalone Streamlit app with real market data via yfinance.\n\n"
        "**Data layer:**\n"
        "- Spot prices: Yahoo Finance continuous futures (`CL=F`, `GC=F`, `ZC=F`, etc.) — cached 5 min\n"
        "- Price history: Yahoo Finance daily OHLCV — cached 1 h\n"
        "- Heatmap: real daily closing prices for any two selected dates\n"
        "- Supply/demand/stocks: model-driven (no public fundamental API)\n"
        "- Macro: model-estimated\n\n"
        f"**Status:** {yf_status}\n\n"
        "**Analytics:**\n"
        "- Black-76 options pricer (calls, puts, full Greeks)\n"
        "- Parametric vol surface (ATM + skew + curvature + vol-of-vol)\n"
        "- Cost-of-carry forward curve anchored to live spot\n"
        "- GBM Monte Carlo starting from live spot\n"
        "- Parametric VaR/CVaR marked to real prices\n"
        "- Real price history chart (2y daily)\n\n"
        f"**{len(COMMODITIES)} commodities** across Energy, Metals, Agriculture, Freight.\n\n"
        "---\n"
        "### 🔗 My Other Projects\n\n"
        "**⚗️ Commodity Options & Derivatives Analytics Platform (CODAP)**  \n"
        "👉 [aeg-codap.streamlit.app](https://aeg-codap.streamlit.app)\n\n"
        "**〽️ Commodity Forward Curve Analytics Platform (CFCAP)**  \n"
        "👉 [aeg-cfcap.streamlit.app](https://aeg-cfcap.streamlit.app)\n\n"
        "---\n"
        "**Author:** Adam EL GBOURI  \n"
        "GitHub · [github.com/adamelgbouri](https://github.com/adamelgbouri)"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    with st.spinner("Loading live market data…"):
        live_prices = fetch_live_prices()

    page, commodity, spot = render_sidebar(live_prices)
    render_header(page, commodity, spot)

    dispatch = {
        "📊 Dashboard":        lambda: page_dashboard(commodity, live_prices),
        "⚖️ Supply & Demand":  lambda: page_balance(commodity, live_prices),
        "🌍 Regional Flows":   lambda: page_regional(commodity, live_prices),
        "📈 Futures Curve":    lambda: page_curve(commodity, live_prices),
        "🎯 Options & Greeks": lambda: page_options(commodity, live_prices),
        "📉 Vol Surface":      lambda: page_vol_surface(commodity, live_prices),
        "💼 Positions & P&L":  lambda: page_positions(live_prices),
        "🛡️ Risk":             lambda: page_risk(live_prices),
        "🎲 Monte Carlo":      lambda: page_mc(commodity, live_prices),
        "🌐 Macro Overlay":    page_macro,
        "📅 Events":           page_events,
        "ℹ️ About":            page_about,
    }
    dispatch.get(page, lambda: st.error("Page not found"))()


if __name__ == "__main__":
    main()
