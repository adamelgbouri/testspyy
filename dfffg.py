"""
Commodity Trading Desk — standalone single-file Streamlit app
by Adam EL GBOURI

LIVE MARK-TO-MARKET BUILD.
Every contract carries BOTH a live front-month settle AND a live dated forward strip.
No hardcoded marks. No cost-of-carry curves standing in for a market. No cross-exchange proxies.
If the feed dies, the screen says so rather than showing a fabricated price.

Run:
    pip install streamlit plotly numpy pandas scipy yfinance requests
    streamlit run desk.py

EIA fundamentals (optional): set an API key in the sidebar. Free at eia.gov/opendata.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE
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

st.set_page_config(page_title="S&D — Commodity Trading Desk", page_icon="🌐",
                   layout="wide", initial_sidebar_state="expanded")
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
#  CONTRACT REGISTRY — LIVE MARK-TO-MARKET ONLY
# ══════════════════════════════════════════════════════════════════════════════
#  Inclusion rule (enforced by assertion below):
#    yf_ticker — continuous front month, drives the mark
#    yf_fmt    — dated contract template, drives the forward strip
#  Missing either => the contract cannot be marked honestly => it is not on this desk.
#
#  contract_size / price_divisor convert a price move into cash.
#  Grains, softs and livestock quote in CENTS: price_divisor=100.
#  bbl_conv: gallons->barrels factor for the crack stack (42 gal = 1 bbl).
#
#  mr_halflife: mean-reversion half-life in YEARS, used by the OU simulator.
#    Storables with tight inventory linkage revert fast (gas, power-adjacent).
#    Precious metals barely revert at all — they behave closer to a financial asset.
#    None => simulate as GBM (no reversion).
#
#  EXCLUDED and why (all need a paid feed to return):
#    LME Copper       — was proxied off HG=F (COMEX, $/lb) but labelled $/mt. ~2200x error.
#    ICE Gasoil       — live front (LGO=F) but no dated strip. Curve was a model.
#    EUA / Coal API2  — no feed. Marks were constants.
#    LME Alu / Nickel — no feed. Marks were constants.
#    Capesize/Panamax — no feed, and cost-of-carry on a non-storable TC route is meaningless.
# ══════════════════════════════════════════════════════════════════════════════
COMMODITIES: Dict[str, dict] = {
    # ── Energy ────────────────────────────────────────────────────────────────
    "WTI Crude (CL)": dict(
        sector="Energy", exchange="NYMEX", unit="$/bbl",
        yf_ticker="CL=F", yf_fmt="CL{M}{YY}.NYM",
        active_months="FGHJKMNQUVXZ", liquid_months=18,
        contract_size=1_000, size_unit="bbl", bbl_conv=1.0,
        vol=0.32, mr_halflife=2.0, ticker="CL",
        reg_unit="mb/d", reg_label="Million barrels per day",
    ),
    "Brent Crude (BZ)": dict(
        sector="Energy", exchange="ICE", unit="$/bbl",
        yf_ticker="BZ=F", yf_fmt="BZ{M}{YY}.NYM",
        active_months="FGHJKMNQUVXZ", liquid_months=18,
        contract_size=1_000, size_unit="bbl", bbl_conv=1.0,
        vol=0.30, mr_halflife=2.0, ticker="BZ",
        reg_unit="mb/d", reg_label="Million barrels per day",
    ),
    "Henry Hub Nat Gas (NG)": dict(
        sector="Energy", exchange="NYMEX", unit="$/MMBtu",
        yf_ticker="NG=F", yf_fmt="NG{M}{YY}.NYM",
        active_months="FGHJKMNQUVXZ", liquid_months=12,
        contract_size=10_000, size_unit="MMBtu",
        vol=0.55, mr_halflife=0.75, ticker="NG", seasonal=True,
        reg_unit="bcf/d", reg_label="Billion cubic feet per day",
    ),
    "RBOB Gasoline (RB)": dict(
        sector="Energy", exchange="NYMEX", unit="$/gal",
        yf_ticker="RB=F", yf_fmt="RB{M}{YY}.NYM",
        active_months="FGHJKMNQUVXZ", liquid_months=12,
        contract_size=42_000, size_unit="gal", bbl_conv=42.0,
        vol=0.36, mr_halflife=1.5, ticker="RB", seasonal=True,
        reg_unit="mb/d", reg_label="Million barrels per day",
    ),
    "ULSD Heating Oil (HO)": dict(
        sector="Energy", exchange="NYMEX", unit="$/gal",
        yf_ticker="HO=F", yf_fmt="HO{M}{YY}.NYM",
        active_months="FGHJKMNQUVXZ", liquid_months=12,
        contract_size=42_000, size_unit="gal", bbl_conv=42.0,
        vol=0.34, mr_halflife=1.5, ticker="HO", seasonal=True,
        reg_unit="mb/d", reg_label="Million barrels per day",
    ),
    # ── Metals ────────────────────────────────────────────────────────────────
    "Gold (GC)": dict(
        sector="Metals", exchange="COMEX", unit="$/troy oz",
        yf_ticker="GC=F", yf_fmt="GC{M}{YY}.CMX",
        active_months="GJMQVZ", liquid_months=8,
        contract_size=100, size_unit="troy oz",
        vol=0.15, mr_halflife=None, ticker="GC",
        reg_unit="t/y", reg_label="Tonnes per year",
    ),
    "Silver (SI)": dict(
        sector="Metals", exchange="COMEX", unit="$/troy oz",
        yf_ticker="SI=F", yf_fmt="SI{M}{YY}.CMX",
        active_months="HKNUZ", liquid_months=6,
        contract_size=5_000, size_unit="troy oz",
        vol=0.28, mr_halflife=None, ticker="SI",
        reg_unit="Moz/y", reg_label="Million troy oz per year",
    ),
    "Copper (HG)": dict(
        sector="Metals", exchange="COMEX", unit="$/lb",
        yf_ticker="HG=F", yf_fmt="HG{M}{YY}.CMX",
        active_months="HKNUZ", liquid_months=8,
        contract_size=25_000, size_unit="lb",
        vol=0.22, mr_halflife=3.0, ticker="HG",
        reg_unit="kt/y", reg_label="Thousand tonnes per year",
    ),
    "Platinum (PL)": dict(
        sector="Metals", exchange="NYMEX", unit="$/troy oz",
        yf_ticker="PL=F", yf_fmt="PL{M}{YY}.NYM",
        active_months="FJNV", liquid_months=6,
        contract_size=50, size_unit="troy oz",
        vol=0.20, mr_halflife=None, ticker="PL",
        reg_unit="Moz/y", reg_label="Million troy oz per year",
    ),
    "Palladium (PA)": dict(
        sector="Metals", exchange="NYMEX", unit="$/troy oz",
        yf_ticker="PA=F", yf_fmt="PA{M}{YY}.NYM",
        active_months="HMUZ", liquid_months=6,
        contract_size=100, size_unit="troy oz",
        vol=0.30, mr_halflife=None, ticker="PA",
        reg_unit="Moz/y", reg_label="Million troy oz per year",
    ),
    # ── Grains & Oilseeds ─────────────────────────────────────────────────────
    "Corn (ZC)": dict(
        sector="Grains", exchange="CBOT", unit="c/bu",
        yf_ticker="ZC=F", yf_fmt="ZC{M}{YY}.CBT",
        active_months="HKNUZ", liquid_months=8,
        contract_size=5_000, size_unit="bu", price_divisor=100.0,
        vol=0.25, mr_halflife=1.5, ticker="ZC", seasonal=True,
        reg_unit="Mbu/y", reg_label="Million bushels per year",
    ),
    "Wheat CBOT SRW (ZW)": dict(
        sector="Grains", exchange="CBOT", unit="c/bu",
        yf_ticker="ZW=F", yf_fmt="ZW{M}{YY}.CBT",
        active_months="HKNUZ", liquid_months=8,
        contract_size=5_000, size_unit="bu", price_divisor=100.0,
        vol=0.28, mr_halflife=1.5, ticker="ZW", seasonal=True,
        reg_unit="Mbu/y", reg_label="Million bushels per year",
    ),
    "Soybeans (ZS)": dict(
        sector="Grains", exchange="CBOT", unit="c/bu",
        yf_ticker="ZS=F", yf_fmt="ZS{M}{YY}.CBT",
        active_months="FHKNQUX", liquid_months=8,
        contract_size=5_000, size_unit="bu", price_divisor=100.0,
        vol=0.23, mr_halflife=1.5, ticker="ZS", seasonal=True,
        reg_unit="Mbu/y", reg_label="Million bushels per year",
    ),
    "Soybean Meal (ZM)": dict(
        sector="Grains", exchange="CBOT", unit="$/short ton",
        yf_ticker="ZM=F", yf_fmt="ZM{M}{YY}.CBT",
        active_months="FHKNQUVZ", liquid_months=8,
        contract_size=100, size_unit="short ton",
        vol=0.26, mr_halflife=1.5, ticker="ZM", seasonal=True,
        reg_unit="kt/y", reg_label="Thousand tonnes per year",
    ),
    "Soybean Oil (ZL)": dict(
        sector="Grains", exchange="CBOT", unit="c/lb",
        yf_ticker="ZL=F", yf_fmt="ZL{M}{YY}.CBT",
        active_months="FHKNQUVZ", liquid_months=8,
        contract_size=60_000, size_unit="lb", price_divisor=100.0,
        vol=0.30, mr_halflife=1.5, ticker="ZL", seasonal=True,
        reg_unit="kt/y", reg_label="Thousand tonnes per year",
    ),
    # ── Softs ─────────────────────────────────────────────────────────────────
    "Sugar #11 (SB)": dict(
        sector="Softs", exchange="ICE US", unit="c/lb",
        yf_ticker="SB=F", yf_fmt="SB{M}{YY}.NYB",
        active_months="HKNV", liquid_months=6,
        contract_size=112_000, size_unit="lb", price_divisor=100.0,
        vol=0.30, mr_halflife=2.0, ticker="SB",
        reg_unit="Mt/y", reg_label="Million tonnes per year",
    ),
    "Arabica Coffee (KC)": dict(
        sector="Softs", exchange="ICE US", unit="c/lb",
        yf_ticker="KC=F", yf_fmt="KC{M}{YY}.NYB",
        active_months="HKNUZ", liquid_months=6,
        contract_size=37_500, size_unit="lb", price_divisor=100.0,
        vol=0.35, mr_halflife=2.0, ticker="KC",
        reg_unit="M bags/y", reg_label="Million 60-kg bags per year",
    ),
    "Cocoa (CC)": dict(
        sector="Softs", exchange="ICE US", unit="$/mt",
        yf_ticker="CC=F", yf_fmt="CC{M}{YY}.NYB",
        active_months="HKNUZ", liquid_months=6,
        contract_size=10, size_unit="mt",
        vol=0.32, mr_halflife=2.0, ticker="CC",
        reg_unit="kt/y", reg_label="Thousand tonnes per year",
    ),
    # ── Livestock ─────────────────────────────────────────────────────────────
    "Live Cattle (LE)": dict(
        sector="Livestock", exchange="CME", unit="c/lb",
        yf_ticker="LE=F", yf_fmt="LE{M}{YY}.CME",
        active_months="GJMQVZ", liquid_months=8,
        contract_size=40_000, size_unit="lb", price_divisor=100.0,
        vol=0.18, mr_halflife=1.0, ticker="LE", seasonal=True,
        reg_unit="Mlb/y", reg_label="Million pounds per year",
    ),
    "Lean Hogs (HE)": dict(
        sector="Livestock", exchange="CME", unit="c/lb",
        yf_ticker="HE=F", yf_fmt="HE{M}{YY}.CME",
        active_months="GJKMNQVZ", liquid_months=6,
        contract_size=40_000, size_unit="lb", price_divisor=100.0,
        vol=0.25, mr_halflife=0.75, ticker="HE", seasonal=True,
        reg_unit="Mlb/y", reg_label="Million pounds per year",
    ),
}

# Fail loudly rather than ship a bad mark.
for _n, _c in COMMODITIES.items():
    assert _c.get("yf_ticker"), f"{_n}: no live front-month ticker"
    assert _c.get("yf_fmt"),    f"{_n}: no dated forward strip template"

ALL_SECTORS = sorted({v["sector"] for v in COMMODITIES.values()})
YF_TICKERS  = {n: c["yf_ticker"] for n, c in COMMODITIES.items()}

MONTH_CODES = list("FGHJKMNQUVXZ")
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ── Structural spread definitions ────────────────────────────────────────────
# Legs are (contract, ratio). Ratio sign = long(+) / short(-) in the structure.
STRUCTURES = {
    "3-2-1 Crack Spread": dict(
        kind="crack",
        legs=[("RBOB Gasoline (RB)", 2), ("ULSD Heating Oil (HO)", 1), ("WTI Crude (CL)", -3)],
        divisor=3, unit="$/bbl",
        desc=("Refining margin. Three barrels of crude in, two of gasoline and one of "
              "distillate out. RB and HO quote in $/gal and are converted at 42 gal/bbl. "
              "Long the crack = long the refiner."),
        typical=(5, 35),
    ),
    "Gasoline Crack (RB-CL)": dict(
        kind="crack",
        legs=[("RBOB Gasoline (RB)", 1), ("WTI Crude (CL)", -1)],
        divisor=1, unit="$/bbl",
        desc="Single-product gasoline crack. Peaks into US driving season, collapses in Q4.",
        typical=(2, 40),
    ),
    "Distillate Crack (HO-CL)": dict(
        kind="crack",
        legs=[("ULSD Heating Oil (HO)", 1), ("WTI Crude (CL)", -1)],
        divisor=1, unit="$/bbl",
        desc="Diesel/heating oil crack. Winter-weighted; driven by distillate stocks.",
        typical=(5, 50),
    ),
    "Board Crush (ZS/ZM/ZL)": dict(
        kind="crush",
        legs=[("Soybean Meal (ZM)", 1), ("Soybean Oil (ZL)", 1), ("Soybeans (ZS)", -1)],
        divisor=1, unit="$/bu",
        desc=("Processor margin. One bushel of beans yields ~44 lb meal and ~11 lb oil. "
              "Long the crush = long the crusher."),
        typical=(0.4, 2.5),
    ),
    "WTI-Brent Arb": dict(
        kind="simple",
        legs=[("WTI Crude (CL)", 1), ("Brent Crude (BZ)", -1)],
        divisor=1, unit="$/bbl",
        desc="Transatlantic arb. Drives US export economics. Historically -1 to -8.",
        typical=(-8, 0),
    ),
    "Gold-Silver Ratio": dict(
        kind="ratio",
        legs=[("Gold (GC)", 1), ("Silver (SI)", 1)],
        divisor=1, unit="ratio",
        desc="Classic precious relative-value gauge. Ratio, not a spread.",
        typical=(60, 95),
    ),
}

# Soybean crush yields: 1 bu (60 lb) -> ~44 lb meal + ~11 lb oil
CRUSH_MEAL_LB = 44.0
CRUSH_OIL_LB  = 11.0
LB_PER_SHORT_TON = 2000.0


def notional_per_lot(commodity: str, price: float) -> float:
    """Cash value of one lot. Handles cent-quoted contracts."""
    c = COMMODITIES[commodity]
    return price / c.get("price_divisor", 1.0) * c["contract_size"]


def price_multiplier(commodity: str) -> float:
    """Cash P&L per 1.0 move in the quoted price, per lot."""
    c = COMMODITIES[commodity]
    return c["contract_size"] / c.get("price_divisor", 1.0)


def to_bbl(commodity: str, price: float) -> float:
    """Normalise a refined-product price to $/bbl for the crack stack."""
    return price * COMMODITIES[commodity].get("bbl_conv", 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE DATA LAYER — no fallbacks, no fabrication
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def fetch_live_marks() -> Dict[str, Optional[float]]:
    """Front-month settle for every contract. None where the feed returned nothing."""
    result: Dict[str, Optional[float]] = {n: None for n in COMMODITIES}
    if not YF_AVAILABLE:
        return result
    try:
        raw = yf.download(list(YF_TICKERS.values()), period="5d",
                          auto_adjust=True, progress=False, threads=True)
        closes = raw["Close"].iloc[-1] if isinstance(raw.columns, pd.MultiIndex) else raw.iloc[-1]
        for n, t in YF_TICKERS.items():
            if t in closes.index and pd.notna(closes[t]):
                result[n] = float(closes[t])
    except Exception:
        pass
    return result


@st.cache_data(ttl=3600)
def fetch_history(yf_ticker: str, period: str = "1y") -> pd.DataFrame:
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
def fetch_panel(period: str = "3y") -> pd.DataFrame:
    """
    Aligned close panel for the whole board. One grouped download.
    Backbone of the correlation matrix, the structure history and seasonality.
    """
    if not YF_AVAILABLE:
        return pd.DataFrame()
    try:
        raw = yf.download(list(YF_TICKERS.values()), period=period,
                          auto_adjust=True, progress=False, threads=True)
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        inv = {v: k for k, v in YF_TICKERS.items()}
        closes = closes.rename(columns=inv)
        keep = [c for c in closes.columns if c in COMMODITIES]
        return closes[keep].dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_close_at_date(yf_ticker: str, target: date) -> Optional[float]:
    if not YF_AVAILABLE or not yf_ticker:
        return None
    try:
        start = (datetime.combine(target, datetime.min.time()) - timedelta(days=10)).strftime("%Y-%m-%d")
        end   = (datetime.combine(target, datetime.min.time()) + timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(yf_ticker, start=start, end=end,
                         auto_adjust=True, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        df = df[df.index.date <= target]
        return float(df["Close"].iloc[-1]) if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def realised_vol(yf_ticker: str, window: int = 60) -> Optional[float]:
    """Annualised close-to-close realised vol."""
    df = fetch_history(yf_ticker, period="1y")
    if df.empty or len(df) < window + 1:
        return None
    lr = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if len(lr) < window:
        return None
    return float(lr.tail(window).std() * math.sqrt(252))


@st.cache_data(ttl=3600)
def correlation_matrix(period: str = "2y", window: int = 252) -> pd.DataFrame:
    """
    Return correlation of daily log returns across the board.
    This is what turns the VaR from an undiversified sum into a real portfolio number.
    """
    panel = fetch_panel(period)
    if panel.empty or len(panel) < 30:
        return pd.DataFrame()
    lr = np.log(panel / panel.shift(1)).dropna(how="all")
    lr = lr.tail(window)
    return lr.corr(min_periods=30)


@st.cache_data(ttl=3600)
def fetch_forward_strip(commodity: str) -> pd.DataFrame:
    """
    Live dated forward strip. Builds exchange codes month by month, drops expired
    months, respects the delivery cycle, batches the download.
    Empty frame => no curve is drawn. This desk will not fit a model and call it a market.
    """
    c      = COMMODITIES[commodity]
    yf_fmt = c["yf_fmt"]
    now    = datetime.now()

    contracts, offset = [], 0
    while len(contracts) < c["liquid_months"] and offset < c["liquid_months"] * 4:
        m    = (now.month - 1 + offset) % 12
        year = now.year + (now.month - 1 + offset) // 12
        offset += 1
        if MONTH_CODES[m] not in c["active_months"]:
            continue
        # Expiry proxy: ~20th of the month preceding delivery.
        exp_m = m - 1 if m > 0 else 11
        exp_y = year if m > 0 else year - 1
        if now > datetime(exp_y, exp_m + 1, 20):
            continue
        contracts.append(dict(
            label=f"{MONTH_NAMES[m]}-{year}",
            month=len(contracts) + 1,
            T=round(len(contracts) / 12 + 1 / 12, 4),
            ticker=yf_fmt.replace("{M}", MONTH_CODES[m]).replace("{YY}", str(year)[-2:]),
        ))

    if not contracts or not YF_AVAILABLE:
        return pd.DataFrame()

    rows = []
    try:
        raw = yf.download([k["ticker"] for k in contracts], period="5d",
                          auto_adjust=True, progress=False, threads=True)
        closes = raw["Close"].iloc[-1] if isinstance(raw.columns, pd.MultiIndex) else raw.iloc[-1]
        for k in contracts:
            t = k["ticker"]
            if t in closes.index and pd.notna(closes[t]):
                rows.append(dict(label=k["label"], month=k["month"], T=k["T"],
                                 price=round(float(closes[t]), 4), ticker=t))
    except Exception:
        return pd.DataFrame()

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def fetch_spread_history(commodity: str, m1_offset: int = 0, m2_offset: int = 1,
                         period: str = "2y") -> pd.DataFrame:
    """
    History of a calendar spread by tracking two specific dated contracts through time.
    A single M1-M2 print is a point; its 2y percentile is what says whether it is cheap.
    """
    strip = fetch_forward_strip(commodity)
    if strip.empty or len(strip) <= max(m1_offset, m2_offset):
        return pd.DataFrame()
    t1 = strip["ticker"].iloc[m1_offset]
    t2 = strip["ticker"].iloc[m2_offset]
    h1 = fetch_history(t1, period=period)
    h2 = fetch_history(t2, period=period)
    if h1.empty or h2.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"near": h1["Close"], "far": h2["Close"]}).dropna()
    if df.empty:
        return pd.DataFrame()
    df["spread"] = df["near"] - df["far"]
    df.attrs["near_label"] = strip["label"].iloc[m1_offset]
    df.attrs["far_label"]  = strip["label"].iloc[m2_offset]
    return df


@st.cache_data(ttl=3600)
def fetch_structure_history(structure: str, period: str = "3y") -> pd.DataFrame:
    """
    History of a crack / crush / arb, computed off the continuous front months.
    Everything is normalised to a common unit before the legs are combined.
    """
    spec  = STRUCTURES[structure]
    panel = fetch_panel(period)
    if panel.empty:
        return pd.DataFrame()

    legs = [n for n, _ in spec["legs"]]
    if any(l not in panel.columns for l in legs):
        return pd.DataFrame()
    df = panel[legs].dropna()
    if df.empty:
        return pd.DataFrame()

    kind = spec["kind"]
    if kind == "crack":
        # Normalise every leg to $/bbl, then apply the ratios.
        val = 0.0
        for name, ratio in spec["legs"]:
            val = val + ratio * df[name] * COMMODITIES[name].get("bbl_conv", 1.0)
        out = val / spec["divisor"]
    elif kind == "crush":
        # Meal $/short ton -> $/bu ; Oil c/lb -> $/bu ; Beans c/bu -> $/bu
        meal = df["Soybean Meal (ZM)"] * CRUSH_MEAL_LB / LB_PER_SHORT_TON
        oil  = df["Soybean Oil (ZL)"] / 100.0 * CRUSH_OIL_LB
        bean = df["Soybeans (ZS)"] / 100.0
        out  = meal + oil - bean
    elif kind == "ratio":
        a, b = spec["legs"][0][0], spec["legs"][1][0]
        out = df[a] / df[b]
    else:  # simple
        val = 0.0
        for name, ratio in spec["legs"]:
            val = val + ratio * df[name]
        out = val / spec["divisor"]

    return pd.DataFrame({"value": out}).dropna()


# ══════════════════════════════════════════════════════════════════════════════
#  EIA FUNDAMENTALS  (free API key at eia.gov/opendata)
# ══════════════════════════════════════════════════════════════════════════════
EIA_SERIES = {
    "US Crude Stocks (ex-SPR)":  dict(sid="PET.WCESTUS1.W",  unit="kbbl",   sector="Energy"),
    "Cushing Crude Stocks":      dict(sid="PET.W_EPC0_SAX_YCUOK_MBBL.W", unit="kbbl", sector="Energy"),
    "US Gasoline Stocks":        dict(sid="PET.WGTSTUS1.W",  unit="kbbl",   sector="Energy"),
    "US Distillate Stocks":      dict(sid="PET.WDISTUS1.W",  unit="kbbl",   sector="Energy"),
    "US Crude Production":       dict(sid="PET.WCRFPUS2.W",  unit="kb/d",   sector="Energy"),
    "US Nat Gas Storage (L48)":  dict(sid="NG.NW2_EPG0_SWO_R48_BCF.W", unit="bcf", sector="Energy"),
}

# Which EIA series matter for which contract.
EIA_MAP = {
    "WTI Crude (CL)":         ["US Crude Stocks (ex-SPR)", "Cushing Crude Stocks", "US Crude Production"],
    "Brent Crude (BZ)":       ["US Crude Stocks (ex-SPR)", "US Crude Production"],
    "RBOB Gasoline (RB)":     ["US Gasoline Stocks"],
    "ULSD Heating Oil (HO)":  ["US Distillate Stocks"],
    "Henry Hub Nat Gas (NG)": ["US Nat Gas Storage (L48)"],
}


@st.cache_data(ttl=3600)
def fetch_eia(series_name: str, api_key: str, n: int = 260) -> pd.DataFrame:
    """
    Pull one weekly EIA series. Returns empty frame on any failure — the page then
    says the feed is unavailable rather than inventing a stock level.
    """
    if not REQUESTS_AVAILABLE or not api_key:
        return pd.DataFrame()
    sid = EIA_SERIES[series_name]["sid"]
    url = "https://api.eia.gov/v2/seriesid/" + sid
    try:
        r = requests.get(url, params={"api_key": api_key, "length": n}, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        rows = r.json().get("response", {}).get("data", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"]  = pd.to_datetime(df["period"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "value"]].dropna().sort_values("date").set_index("date")
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
#  REGIONAL BALANCES  (static IEA/USDA-style estimates — not a live feed)
# ══════════════════════════════════════════════════════════════════════════════
REGIONAL_DATA = {
    "WTI Crude (CL)": [
        dict(region="North America", supply=15.2, demand=10.8, lat=45,  lon=-100),
        dict(region="Middle East",   supply=28.1, demand=8.2,  lat=25,  lon=50),
        dict(region="Russia/FSU",    supply=13.5, demand=5.1,  lat=60,  lon=70),
        dict(region="Europe",        supply=3.4,  demand=12.6, lat=52,  lon=10),
        dict(region="Asia Pacific",  supply=8.1,  demand=34.5, lat=30,  lon=115),
        dict(region="Africa",        supply=7.9,  demand=4.3,  lat=5,   lon=20),
        dict(region="Latin America", supply=5.8,  demand=6.2,  lat=-15, lon=-60),
    ],
    "Gold (GC)": [
        dict(region="China",     supply=370, demand=950, lat=35,  lon=105),
        dict(region="Australia", supply=330, demand=30,  lat=-25, lon=133),
        dict(region="Russia",    supply=295, demand=90,  lat=60,  lon=70),
        dict(region="Canada",    supply=190, demand=50,  lat=60,  lon=-95),
        dict(region="USA",       supply=170, demand=230, lat=38,  lon=-97),
        dict(region="S. Africa", supply=120, demand=45,  lat=-30, lon=25),
        dict(region="India",     supply=35,  demand=800, lat=20,  lon=80),
        dict(region="Europe",    supply=30,  demand=280, lat=50,  lon=15),
    ],
    "Corn (ZC)": [
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
    dict(date=date.today()+timedelta(days=3),  event="EIA Weekly Petroleum Status Report", tags=["Energy", "Crude"]),
    dict(date=date.today()+timedelta(days=5),  event="USDA WASDE",                          tags=["Grains", "Softs"]),
    dict(date=date.today()+timedelta(days=7),  event="OPEC+ Ministerial (JMMC)",            tags=["Energy", "OPEC"]),
    dict(date=date.today()+timedelta(days=10), event="FOMC Rate Decision",                  tags=["Macro", "Rates"]),
    dict(date=date.today()+timedelta(days=12), event="IEA Oil Market Report (OMR)",         tags=["Energy"]),
    dict(date=date.today()+timedelta(days=14), event="US CPI",                              tags=["Macro", "Inflation"]),
    dict(date=date.today()+timedelta(days=16), event="EIA Natural Gas Storage Report",      tags=["Energy", "Gas"]),
    dict(date=date.today()+timedelta(days=18), event="USDA Crop Progress",                  tags=["Grains"]),
    dict(date=date.today()+timedelta(days=21), event="LME Week",                            tags=["Metals"]),
    dict(date=date.today()+timedelta(days=25), event="ECB Governing Council",               tags=["Macro", "Rates"]),
    dict(date=date.today()+timedelta(days=28), event="Baker Hughes Rig Count",              tags=["Energy"]),
    dict(date=date.today()+timedelta(days=32), event="OPEC MOMR",                           tags=["Energy", "OPEC"]),
    dict(date=date.today()+timedelta(days=35), event="USDA Cattle on Feed",                 tags=["Livestock"]),
    dict(date=date.today()+timedelta(days=38), event="USDA Grain Stocks",                   tags=["Grains"]),
]

# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def black76(F, K, T, r, sigma, option_type="call"):
    """European option on a futures contract. Greeks per unit of underlying."""
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return dict(price=0, delta=0, gamma=0, vega=0, theta=0, rho=0)
    d1   = (math.log(F/K) + 0.5*sigma**2*T) / (sigma*math.sqrt(T))
    d2   = d1 - sigma*math.sqrt(T)
    disc = math.exp(-r*T)
    if option_type == "call":
        price, delta = disc*(F*norm.cdf(d1) - K*norm.cdf(d2)), disc*norm.cdf(d1)
    else:
        price, delta = disc*(K*norm.cdf(-d2) - F*norm.cdf(-d1)), -disc*norm.cdf(-d1)
    gamma = disc*norm.pdf(d1)/(F*sigma*math.sqrt(T))
    vega  = disc*F*norm.pdf(d1)*math.sqrt(T)/100        # per vol point
    theta = (-(disc*F*norm.pdf(d1)*sigma)/(2*math.sqrt(T)) - r*price)/365
    rho   = -T*price/100
    return dict(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def vol_surface_fn(F, atm_vol, skew=-0.05, curv=0.02, vov=0.15):
    """Parametric surface in log-moneyness. NOT calibrated to listed quotes."""
    mats  = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])
    Kgrid = F * np.exp(np.linspace(-0.40, 0.40, 25))
    Z = np.zeros((len(mats), len(Kgrid)))
    for i, T in enumerate(mats):
        for j, K in enumerate(Kgrid):
            x = math.log(K/F)
            Z[i, j] = max(atm_vol*(1 + vov*math.sqrt(T)) + skew*x + curv*x**2, 0.01)
    return mats, Kgrid, Z


def implied_carry(strip: pd.DataFrame) -> pd.DataFrame:
    """
    Carry read straight off the strip. No storage cost or convenience yield assumed —
    the market has already priced them, and this just reads what it priced.
    """
    if strip.empty or len(strip) < 2:
        return pd.DataFrame()
    f1  = float(strip["price"].iloc[0])
    out = strip.copy()
    out["spread_vs_M1"] = out["price"] - f1
    out["spread_pct"]   = (out["price"] / f1 - 1) * 100
    out["roll_yield"]   = np.where(out["T"] > 0,
                                   (f1 - out["price"]) / out["price"] / out["T"] * 100, 0.0)
    return out


def simulate(spot: float, vol: float, n_paths: int = 1000, horizon: int = 18,
             halflife: Optional[float] = None, seed: int = 0) -> dict:
    """
    Price simulator.

    halflife=None  -> GBM. Driftless, no reversion. Appropriate for gold/silver, which
                      behave like financial assets rather than consumables.

    halflife=h     -> One-factor Schwartz (OU on log price), mean-reverting to the
                      current forward level with speed kappa = ln(2)/h.

                          d(lnS) = kappa*(mu - lnS)*dt - 0.5*sigma^2*dt + sigma*dW

                      This matters. GBM at nat gas vol (55%) over 3y puts the P95 at
                      ~3x spot and the P5 near zero — physically meaningless for a
                      storable with an inventory-driven price. Reversion fixes the tails.
    """
    rng = np.random.default_rng(seed)
    dt  = 1/12
    paths = np.zeros((n_paths, horizon+1))
    paths[:, 0] = spot

    if halflife is None:
        for t in range(1, horizon+1):
            z = rng.standard_normal(n_paths)
            paths[:, t] = paths[:, t-1] * np.exp(-0.5*vol**2*dt + vol*math.sqrt(dt)*z)
        model = "GBM (no reversion)"
    else:
        kappa = math.log(2) / halflife
        mu    = math.log(spot)
        x     = np.full(n_paths, math.log(spot))
        for t in range(1, horizon+1):
            z = rng.standard_normal(n_paths)
            x = x + kappa*(mu - x)*dt - 0.5*vol**2*dt + vol*math.sqrt(dt)*z
            paths[:, t] = np.exp(x)
        model = f"Schwartz 1-factor (half-life {halflife:.2f}y, κ={kappa:.2f})"

    fan_dates = [date.today() + timedelta(days=30*i) for i in range(horizon+1)]
    pcts = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fan  = pd.DataFrame(dict(date=fan_dates, p5=pcts[0], p25=pcts[1],
                             p50=pcts[2], p75=pcts[3], p95=pcts[4]))
    hb = np.histogram(paths[:, -1], bins=40)
    return dict(fan=fan, model=model,
                median=float(np.median(paths[:, -1])),
                p5=float(np.percentile(paths[:, -1], 5)),
                p95=float(np.percentile(paths[:, -1], 95)),
                hist_x=hb[1][:-1].tolist(), hist_y=hb[0].tolist())


def portfolio_var(positions: List[dict], marks: Dict[str, Optional[float]],
                  corr: pd.DataFrame, conf: float = 0.95,
                  horizon: int = 1, diversified: bool = True) -> dict:
    """
    Parametric VaR / expected shortfall on the delta-equivalent book.

    diversified=True  -> sigma_p = sqrt(w' Sigma w) using the live correlation matrix.
                         A long WTI / short Brent book nets down to near-nothing, which
                         is the economic truth. This is the number to look at.

    diversified=False -> position VaRs summed. Undiversified upper bound. Kept as a
                         comparison so the diversification benefit is explicit.
    """
    z = norm.ppf(conf)
    rows, signed_notional, names = [], [], []
    gross = 0.0

    for p in positions:
        name = p["commodity"]
        mark = marks.get(name)
        if mark is None:
            continue
        c    = COMMODITIES[name]
        vol  = p.get("vol", c["vol"])
        sign = 1 if p["side"] == "Long" else -1
        notl = notional_per_lot(name, mark) * p["lots"] * sign
        dvol = vol / math.sqrt(252)
        sd_var = abs(notl) * dvol * z * math.sqrt(horizon)
        sd_es  = abs(notl) * dvol * norm.pdf(z) / (1 - conf) * math.sqrt(horizon)
        gross += abs(notl)
        signed_notional.append(notl)
        names.append(name)
        rows.append(dict(Contract=name, Side=p["side"], Lots=p["lots"],
                         Mark=round(mark, 4), Notional=notl, Vol=vol*100,
                         StandaloneVaR=sd_var, StandaloneES=sd_es,
                         _dvol=dvol))

    if not rows:
        return dict(rows=[], var=0.0, es=0.0, undiversified=0.0,
                    gross=0.0, benefit=0.0, corr_used=False)

    undiversified = sum(r["StandaloneVaR"] for r in rows)

    # Position vol in cash terms: |notional| * daily vol
    sigma_vec = np.array([abs(r["Notional"]) * r["_dvol"] for r in rows])
    sgn       = np.array([1 if r["Notional"] >= 0 else -1 for r in rows])
    w         = sigma_vec * sgn                       # signed cash-vol vector

    corr_used = False
    if diversified and not corr.empty and all(n in corr.index for n in names):
        R = corr.loc[names, names].values
        R = np.nan_to_num(R, nan=0.0)
        np.fill_diagonal(R, 1.0)
        port_sigma = math.sqrt(max(float(w @ R @ w), 0.0))
        corr_used = True
    else:
        # No usable correlation => fall back to the conservative sum. Never silently
        # assume independence, which would understate risk.
        port_sigma = float(np.abs(w).sum())

    var = port_sigma * z * math.sqrt(horizon)
    es  = port_sigma * norm.pdf(z) / (1 - conf) * math.sqrt(horizon)

    # Marginal / component VaR — who is actually carrying the risk.
    if corr_used and port_sigma > 0:
        R = corr.loc[names, names].values
        R = np.nan_to_num(R, nan=0.0)
        np.fill_diagonal(R, 1.0)
        mcv = (R @ w) / port_sigma                     # d(sigma_p)/d(w_i)
        for i, r in enumerate(rows):
            comp = w[i] * mcv[i] / port_sigma * var
            r["ComponentVaR"] = float(comp)
            r["PctOfVaR"] = float(comp / var * 100) if var else 0.0
    else:
        for r in rows:
            r["ComponentVaR"] = r["StandaloneVaR"]
            r["PctOfVaR"] = r["StandaloneVaR"] / undiversified * 100 if undiversified else 0.0

    for r in rows:
        r.pop("_dvol", None)

    return dict(rows=rows, var=var, es=es, undiversified=undiversified,
                gross=gross,
                benefit=(undiversified - var) / undiversified * 100 if undiversified else 0.0,
                corr_used=corr_used)


def seasonality(yf_ticker: str, years: int = 10) -> pd.DataFrame:
    """
    Monthly seasonal distribution of returns. For gas, hogs, RB and the grains this
    is a first-order driver and it is trivially available from the history we already pull.
    """
    df = fetch_history(yf_ticker, period=f"{years}y")
    if df.empty or len(df) < 250:
        return pd.DataFrame()
    m = df["Close"].resample("ME").last()
    r = (m / m.shift(1) - 1).dropna() * 100
    out = pd.DataFrame({"ret": r})
    out["month"] = out.index.month
    out["year"]  = out.index.year
    return out


@st.cache_data(ttl=3600)
def fetch_fred(series_id: str, api_key: str, start: str = "2015-01-01") -> pd.DataFrame:
    """
    One FRED series. Real data or nothing — same contract as the price feed.
    Free key: fred.stlouisfed.org/docs/api/api_key.html
    """
    if not REQUESTS_AVAILABLE or not api_key or not series_id:
        return pd.DataFrame()
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": api_key, "file_type": "json",
                    "observation_start": start},
            timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        obs = r.json().get("observations", [])
        if not obs:
            return pd.DataFrame()
        df = pd.DataFrame(obs)
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["date", "value"]].dropna().set_index("date")
    except Exception:
        return pd.DataFrame()


def macro_series(country: str, metric: str, api_key: str) -> pd.DataFrame:
    """Resolve a (country, metric) pair to a FRED series and pull it."""
    sid = FRED_SERIES.get(country, {}).get(metric)
    if not sid:
        return pd.DataFrame()
    return fetch_fred(sid, api_key)


# ══════════════════════════════════════════════════════════════════════════════
#  FRED SERIES MAP  (free key at fred.stlouisfed.org)
# ══════════════════════════════════════════════════════════════════════════════
#  Macro was the last model-generated page on this desk. It is now live or empty,
#  same rule as everything else. FRED carries non-US series too, so the country
#  list is real rather than a set of invented base levels.
FRED_SERIES = {
    "USA":     dict(cpi_yoy="CPIAUCSL", policy_rate="DFF",       gdp="GDPC1",     pmi="MANEMP"),
    "Euro Area": dict(cpi_yoy="CP0000EZ19M086NEST", policy_rate="ECBDFR", gdp="CLVMNACSCAB1GQEA19", pmi=None),
    "Germany": dict(cpi_yoy="DEUCPIALLMINMEI", policy_rate="ECBDFR", gdp="CLVMNACSCAB1GQDE", pmi=None),
    "France":  dict(cpi_yoy="FRACPIALLMINMEI", policy_rate="ECBDFR", gdp="CLVMNACSCAB1GQFR", pmi=None),
    "UK":      dict(cpi_yoy="GBRCPIALLMINMEI", policy_rate="IUDSOIA", gdp="NGDPRSAXDCGBQ", pmi=None),
    "Japan":   dict(cpi_yoy="JPNCPIALLMINMEI", policy_rate="IRSTCI01JPM156N", gdp="JPNRGDPEXP", pmi=None),
    "China":   dict(cpi_yoy="CHNCPIALLMINMEI", policy_rate=None,  gdp="NGDPRSAXDCCNQ", pmi=None),
    "Brazil":  dict(cpi_yoy="BRACPIALLMINMEI", policy_rate="INTDSRBRM193N", gdp="NGDPRSAXDCBRQ", pmi=None),
    "India":   dict(cpi_yoy="INDCPIALLMINMEI", policy_rate="INTDSRINM193N", gdp="NGDPRSAXDCINQ", pmi=None),
}

MACRO_METRICS = {
    "cpi_yoy":     dict(label="CPI (index)",      note="Index level. YoY % is derived below."),
    "policy_rate": dict(label="Policy rate (%)",  note="Central bank target / overnight rate."),
    "gdp":         dict(label="Real GDP",         note="Real GDP, local units. Quarterly."),
}

# Commodity-relevant FRED series — the ones a commodity desk actually watches.
FRED_COMMODITY_CONTEXT = {
    "US Dollar Index (DXY proxy)": "DTWEXBGS",
    "US 10Y Treasury Yield":       "DGS10",
    "US 10Y Breakeven Inflation":  "T10YIE",
    "US Industrial Production":    "INDPRO",
}


# ══════════════════════════════════════════════════════════════════════════════
#  BLOTTER PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════
#  st.session_state dies on refresh. A book that vanishes when you close the tab
#  is not a book. Positions are serialised to JSON so they survive a reload, and
#  can be exported / re-imported to move a book between machines.
BLOTTER_FILE = "blotter.json"


def blotter_save(positions: List[dict]) -> bool:
    try:
        with open(BLOTTER_FILE, "w") as f:
            json.dump(positions, f, indent=2)
        return True
    except Exception:
        return False


def blotter_load() -> List[dict]:
    try:
        if os.path.exists(BLOTTER_FILE):
            with open(BLOTTER_FILE) as f:
                data = json.load(f)
            # Drop any line referencing a contract no longer on the desk.
            return [p for p in data if p.get("commodity") in COMMODITIES]
    except Exception:
        pass
    return []


def blotter_serialise(positions: List[dict]) -> str:
    return json.dumps(positions, indent=2)


def blotter_deserialise(raw: str) -> Optional[List[dict]]:
    """Parse an uploaded book. Returns None if it is not a valid blotter."""
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return None
        out = []
        for p in data:
            if p.get("commodity") not in COMMODITIES:
                continue
            if p.get("kind", "future") == "option":
                out.append(dict(
                    kind="option", commodity=p["commodity"], side=p["side"],
                    lots=int(p["lots"]), entry=float(p["entry"]),
                    opt_type=p.get("opt_type", "call"), strike=float(p["strike"]),
                    tenor=float(p["tenor"]), vol=float(p.get("vol", 0.3))))
            else:
                out.append(dict(
                    kind="future", commodity=p["commodity"], side=p["side"],
                    lots=int(p["lots"]), entry=float(p["entry"]),
                    vol=float(p.get("vol", COMMODITIES[p["commodity"]]["vol"]))))
        return out
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  BOOK ANALYTICS — Greeks, roll P&L, historical VaR
# ══════════════════════════════════════════════════════════════════════════════
def book_greeks(positions: List[dict], marks: Dict[str, Optional[float]],
                r: float = 0.05) -> dict:
    """
    Net Greeks across the whole book, in cash.

    Futures carry delta 1.0 per unit and nothing else. Options carry the full set.
    This answers the question a real options desk lives by — "how much vega am I
    actually short?" — which a per-option pricer cannot.
    """
    tot = dict(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
    rows = []
    for p in positions:
        n = p["commodity"]
        mark = marks.get(n)
        if mark is None:
            continue
        mult = price_multiplier(n)
        sign = 1 if p["side"] == "Long" else -1
        lots = p["lots"]

        if p.get("kind", "future") == "option":
            g = black76(mark, p["strike"], p["tenor"], r, p["vol"], p["opt_type"])
            d = g["delta"] * mult * lots * sign
            gm = g["gamma"] * mult * lots * sign
            v  = g["vega"]  * mult * lots * sign
            th = g["theta"] * mult * lots * sign
            label = f"{n} {p['opt_type'][:1].upper()}{p['strike']:g}"
        else:
            d, gm, v, th = 1.0 * mult * lots * sign, 0.0, 0.0, 0.0
            label = f"{n} fut"

        tot["delta"] += d
        tot["gamma"] += gm
        tot["vega"]  += v
        tot["theta"] += th
        rows.append(dict(Position=label, Side=p["side"], Lots=lots,
                         Delta=d, Gamma=gm, Vega=v, Theta=th))
    return dict(total=tot, rows=rows)


def roll_pnl(positions: List[dict], marks: Dict[str, Optional[float]]) -> List[dict]:
    """
    Split P&L into price and carry.

    Price P&L  = (mark - entry) x multiplier x lots
    Roll P&L   = (M1 - M2) x multiplier x lots, sign-adjusted

    A long position in backwardation earns the roll; in contango it bleeds. A book
    that only reports a single P&L number cannot tell a trader which of the two is
    actually happening, and they have opposite implications for whether to stay in.
    """
    out = []
    for p in positions:
        if p.get("kind", "future") == "option":
            continue
        n = p["commodity"]
        mark = marks.get(n)
        if mark is None:
            continue
        mult = price_multiplier(n)
        sign = 1 if p["side"] == "Long" else -1
        price_p = sign * (mark - p["entry"]) * mult * p["lots"]

        strip = fetch_forward_strip(n)
        if strip.empty or len(strip) < 2:
            roll_p, m1m2, ann = 0.0, None, None
        else:
            m1 = float(strip["price"].iloc[0])
            m2 = float(strip["price"].iloc[1])
            m1m2 = m1 - m2
            roll_p = sign * m1m2 * mult * p["lots"]
            ann = (m1m2 / m2 * 12 * 100) if m2 else None

        out.append(dict(Contract=n, Side=p["side"], Lots=p["lots"],
                        PricePnL=price_p, MonthlyRoll=roll_p,
                        M1M2=m1m2, RollAnnPct=ann))
    return out


def historical_var(positions: List[dict], marks: Dict[str, Optional[float]],
                   conf: float = 0.95, horizon: int = 1,
                   lookback: int = 500) -> dict:
    """
    Historical simulation VaR. Replays actual daily return vectors on today's book.

    Makes NO distributional assumption — no normality, no correlation matrix. The
    joint behaviour, including the fat tails and the way correlations snapped to 1
    on the worst days, is already in the data.

    Parametric VaR understates the tail precisely when it matters. Showing both, and
    the gap between them, is the honest presentation.
    """
    panel = fetch_panel("3y")
    if panel.empty:
        return dict(available=False)

    names, w = [], []
    for p in positions:
        if p.get("kind", "future") == "option":
            continue                      # delta-equivalent only; options need a full reval
        n = p["commodity"]
        mark = marks.get(n)
        if mark is None or n not in panel.columns:
            continue
        sign = 1 if p["side"] == "Long" else -1
        names.append(n)
        w.append(notional_per_lot(n, mark) * p["lots"] * sign)

    if not names:
        return dict(available=False)

    rets = panel[names].pct_change().dropna().tail(lookback)
    if len(rets) < 100:
        return dict(available=False)

    pnl = (rets.values @ np.array(w)) * math.sqrt(horizon)
    var = float(-np.percentile(pnl, (1 - conf) * 100))
    tail = pnl[pnl <= -var]
    es = float(-tail.mean()) if len(tail) else var

    worst_i = int(np.argmin(pnl))
    return dict(available=True, var=var, es=es, pnl=pnl, n_days=len(pnl),
                worst_date=rets.index[worst_i].date(),
                worst_pnl=float(pnl[worst_i]))


# ── Historical stress episodes ───────────────────────────────────────────────
#  A parallel +/-30% shock is a weak test: real dislocations are not parallel.
#  These replay dated windows against the current book. Only episodes within the
#  Yahoo history window are usable — the 2022 LME nickel squeeze is deliberately
#  absent because nickel is not on this desk and never will be on a free feed.
STRESS_EPISODES = {
    "COVID crash (Feb–Mar 2020)":        ("2020-02-19", "2020-03-23"),
    "WTI negative print (Apr 2020)":     ("2020-04-01", "2020-04-30"),
    "Ukraine invasion (Feb–Mar 2022)":   ("2022-02-21", "2022-03-09"),
    "2022 energy peak → bust (Jun–Sep)": ("2022-06-08", "2022-09-26"),
    "Banking wobble (Mar 2023)":         ("2023-03-08", "2023-03-24"),
}


def stress_replay(positions: List[dict], marks: Dict[str, Optional[float]],
                  start: str, end: str) -> dict:
    """Apply the actual move of a dated episode to the current book, per contract."""
    panel = fetch_panel("5y")
    if panel.empty:
        return dict(available=False)

    try:
        window = panel.loc[start:end]
    except Exception:
        return dict(available=False)
    if window.empty or len(window) < 2:
        return dict(available=False)

    rows, total = [], 0.0
    for p in positions:
        if p.get("kind", "future") == "option":
            continue
        n = p["commodity"]
        mark = marks.get(n)
        if mark is None or n not in window.columns:
            continue
        s = window[n].dropna()
        if len(s) < 2:
            continue
        move = float(s.iloc[-1] / s.iloc[0] - 1)
        sign = 1 if p["side"] == "Long" else -1
        pnl  = sign * mark * move * price_multiplier(n) * p["lots"]
        total += pnl
        rows.append(dict(Contract=n, Side=p["side"], Lots=p["lots"],
                         Move=move*100, PnL=pnl))
    if not rows:
        return dict(available=False)
    return dict(available=True, rows=rows, total=total,
                start=str(window.index[0].date()), end=str(window.index[-1].date()))


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL SCANNER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=900)
def build_signals() -> pd.DataFrame:
    """
    One row per contract, one column per signal. Pure aggregation of what the other
    pages already compute — the desk had sixteen screens and no single place that
    said "where should I be looking today". This is that place.

    Every column is live-derived. Nothing here is modelled.
    """
    panel = fetch_panel("3y")
    rows = []
    this_month = date.today().month

    for name, c in COMMODITIES.items():
        row = dict(Contract=name, Sector=c["sector"])

        # ── Curve structure & carry ──
        strip = fetch_forward_strip(name)
        if not strip.empty and len(strip) >= 2:
            f1 = float(strip["price"].iloc[0])
            f2 = float(strip["price"].iloc[1])
            fn = float(strip["price"].iloc[-1])
            row["Mark"]      = f1
            row["Carry%"]    = (fn - f1) / f1 * 100
            row["Structure"] = ("CONTANGO" if row["Carry%"] > 0.5
                                else "BACKWARD" if row["Carry%"] < -0.5 else "FLAT")
            row["M1M2"] = f1 - f2
        else:
            row.update(Mark=np.nan, **{"Carry%": np.nan}, Structure="n/a", M1M2=np.nan)

        # ── Vol regime: is realised vol above or below its own 1y norm? ──
        h = fetch_history(c["yf_ticker"], "1y")
        if not h.empty and len(h) > 130:
            lr = np.log(h["Close"] / h["Close"].shift(1)).dropna()
            rv60  = float(lr.tail(60).std() * math.sqrt(252))
            rv252 = float(lr.std() * math.sqrt(252))
            row["RV60"]      = rv60 * 100
            row["VolRegime"] = rv60 / rv252 if rv252 > 0 else np.nan
        else:
            row["RV60"], row["VolRegime"] = np.nan, np.nan

        # ── Momentum ──
        if not h.empty and len(h) > 60:
            px = h["Close"]
            row["Chg1M"] = float(px.iloc[-1] / px.iloc[-21] - 1) * 100 if len(px) > 21 else np.nan
            row["Chg3M"] = float(px.iloc[-1] / px.iloc[-63] - 1) * 100 if len(px) > 63 else np.nan
            row["Px%ile1y"] = float((px < px.iloc[-1]).mean() * 100)
        else:
            row["Chg1M"] = row["Chg3M"] = row["Px%ile1y"] = np.nan

        # ── Seasonal bias for the current calendar month ──
        if c.get("seasonal"):
            s = seasonality(c["yf_ticker"], 10)
            if not s.empty:
                d = s[s["month"] == this_month]["ret"]
                row["SeasonMed"] = float(d.median()) if len(d) else np.nan
                row["SeasonHit"] = float((d > 0).mean() * 100) if len(d) else np.nan
            else:
                row["SeasonMed"] = row["SeasonHit"] = np.nan
        else:
            row["SeasonMed"] = row["SeasonHit"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _styled(fig, h=380):
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


def kpi(label, value, sub="", accent=AMBER):
    return (f'<div class="kpi-card" style="border-left-color:{accent}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def pctile_badge(series: pd.Series, current: float) -> Tuple[float, str, str]:
    """Where does the current print sit in its own history?"""
    if series.empty:
        return 0.0, "n/a", GRAY
    pct = float((series < current).mean() * 100)
    if pct >= 80:
        return pct, "RICH", RED
    if pct <= 20:
        return pct, "CHEAP", GREEN
    return pct, "MID", AMBER


def require_mark(commodity: str, marks: Dict[str, Optional[float]]) -> Optional[float]:
    """Gate every page behind a live mark. No mark, no screen."""
    mark = marks.get(commodity)
    if mark is None:
        st.error(
            f"**No live mark for {commodity}.** The feed returned nothing for "
            f"`{COMMODITIES[commodity]['yf_ticker']}`. This desk does not substitute "
            f"a modelled or stale price. Retry, or check feed status."
        )
    return mark


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(marks):
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
        # ── Market ──
        "📡 Signals", "📊 Dashboard", "📈 Forward Curve",
        # ── Relative value ──
        "🔀 Spreads & Roll", "⚗️ Crack & Crush", "🔗 Correlation",
        # ── Fundamentals ──
        "🛢️ EIA Fundamentals", "🗓️ Seasonality", "🌍 Regional Balances", "📅 Calendar",
        # ── Derivatives ──
        "🎯 Options & Greeks", "📉 Vol Surface",
        # ── Book ──
        "💼 Blotter", "🛡️ Risk", "🎲 Monte Carlo",
        # ── System ──
        "🌐 Macro", "ℹ️ About",
    ]
    page = st.sidebar.radio("Pages", pages, label_visibility="collapsed")
    st.sidebar.markdown("---")

    st.sidebar.markdown(
        f'<div style="font-size:10px;color:{GRAY};font-family:JetBrains Mono,monospace;'
        f'letter-spacing:0.15em;text-transform:uppercase;margin-bottom:6px;">Contract</div>',
        unsafe_allow_html=True,
    )
    sector    = st.sidebar.selectbox("Sector", ALL_SECTORS, key="sb_sector")
    names_in  = [k for k, v in COMMODITIES.items() if v["sector"] == sector]
    commodity = st.sidebar.selectbox("Contract", names_in, key="sb_contract")
    c = COMMODITIES[commodity]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div style="font-size:10px;color:{GRAY};font-family:JetBrains Mono,monospace;'
        f'letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;">Option Params</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.slider("Tenor (months)", 1, 36, 6, key="opt_T_months")
    st.sidebar.slider("Strike (% of F)", 70, 130, 100, key="opt_K_pct")
    st.sidebar.slider("Discount rate r (%)", 0, 10, 5, key="opt_r_pct")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div style="font-size:10px;color:{GRAY};font-family:JetBrains Mono,monospace;'
        f'letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;">Data Keys</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.text_input("EIA key", type="password", key="eia_key",
                          help="Free at eia.gov/opendata. Enables real US crude, "
                               "product and gas inventories on the Fundamentals page.")
    st.sidebar.text_input("FRED key", type="password", key="fred_key",
                          help="Free at fred.stlouisfed.org. Enables real CPI, policy rates, "
                               "GDP, the dollar index and real yields on the Macro page.")

    n_live = sum(1 for v in marks.values() if v is not None)
    ok = n_live == len(COMMODITIES)
    st.sidebar.markdown(
        f'<div style="font-size:9px;color:{GRAY};font-family:JetBrains Mono,monospace;'
        f'letter-spacing:0.08em;text-transform:uppercase;margin-top:20px;">'
        f'{datetime.now().strftime("%Y-%m-%d %H:%M")}<br>'
        f'by Adam EL GBOURI · {date.today().year}<br>aeg-snd.streamlit.app<br>'
        f'<span style="color:{GREEN if ok else RED};">'
        f'feed: {n_live}/{len(COMMODITIES)} marked</span></div>',
        unsafe_allow_html=True,
    )
    return page, commodity


def render_header(commodity, mark):
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
            f'</div></div>', unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div style="text-align:right;padding-top:10px;color:{GRAY};'
            f'font-size:11px;font-family:JetBrains Mono,monospace;">'
            f'{datetime.now().strftime("%Y-%m-%d %H:%M")}</div>',
            unsafe_allow_html=True,
        )
    badge = (f'<span class="badge" style="color:{GREEN};border-color:rgba(63,185,80,0.4);">● LIVE</span>'
             if mark is not None else
             f'<span class="badge" style="color:{RED};border-color:rgba(255,123,114,0.4);">● NO MARK</span>')
    st.markdown(
        f'{badge}<span class="badge">{c["exchange"]}</span>'
        f'<span class="badge">{c["ticker"]}</span>'
        f'<span class="badge">{c["contract_size"]:,} {c["size_unit"]}/lot</span>'
        f'<span class="badge">{c["sector"]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard(commodity, marks):
    c = COMMODITIES[commodity]
    mark = require_mark(commodity, marks)
    if mark is None:
        return

    hist = fetch_history(c["yf_ticker"], period="5d")
    chg = ((mark - float(hist["Close"].iloc[-2])) / float(hist["Close"].iloc[-2]) * 100
           if not hist.empty and len(hist) >= 2 else 0.0)

    rv = realised_vol(c["yf_ticker"], 60)
    rv_txt = f"{rv*100:.1f}%" if rv else "n/a"

    strip = fetch_forward_strip(commodity)
    if not strip.empty and len(strip) >= 2:
        f1, fn = float(strip["price"].iloc[0]), float(strip["price"].iloc[-1])
        carry  = (fn - f1) / f1 * 100
        struct = "CONTANGO" if carry > 0.5 else "BACKWARDATION" if carry < -0.5 else "FLAT"
    else:
        carry, struct = 0.0, "n/a"

    s_col = GREEN if struct == "BACKWARDATION" else RED if struct == "CONTANGO" else GRAY
    st.markdown(
        f'<span class="badge badge-amber">M1 {mark:,.4f} {c["unit"]}</span>'
        f'<span class="badge" style="color:{GREEN if chg>=0 else RED};">{chg:+.2f}% D/D</span>'
        f'<span class="badge">RV60 {rv_txt}</span>'
        f'<span class="badge" style="color:{s_col};">{struct}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(5)
    for col, (l, v, s, a) in zip(cols, [
        ("Front Month",  f"{mark:,.4f}", c["unit"], AMBER),
        ("Change D/D",   f"{chg:+.2f}%", "vs prior settle", GREEN if chg >= 0 else RED),
        ("Realised Vol", rv_txt, "60d annualised", PURPLE),
        ("Curve Carry",  f"{carry:+.2f}%", "M1 → back", TEAL),
        ("Lot Notional", f"${notional_per_lot(commodity, mark):,.0f}",
         f"{c['contract_size']:,} {c['size_unit']}", BLUE),
    ]):
        col.markdown(kpi(l, v, s, a), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Front-Month History")
    h2 = fetch_history(c["yf_ticker"], period="2y")
    if not h2.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h2.index, y=h2["Close"], name="Settle",
                                 line=dict(color=AMBER, width=2)))
        fig.add_trace(go.Scatter(x=h2.index, y=h2["Close"].rolling(50).mean(),
                                 name="50d MA", line=dict(color=BLUE, width=1.2, dash="dot")))
        fig.update_layout(yaxis_title=c["unit"])
        st.plotly_chart(_styled(fig, 360), use_container_width=True)

    st.subheader("Board — Performance Between Two Settles")
    cd1, cd2 = st.columns(2)
    d_a = cd1.date_input("From", value=date.today()-timedelta(days=30), key="hm_a")
    d_b = cd2.date_input("To",   value=date.today(), key="hm_b")
    if st.button("Load Board", type="primary"):
        st.session_state["hm_loaded"] = True

    if st.session_state.get("hm_loaded"):
        with st.spinner("Pulling settlements…"):
            rows = []
            for n, info in COMMODITIES.items():
                pa = fetch_close_at_date(info["yf_ticker"], d_a)
                pb = fetch_close_at_date(info["yf_ticker"], d_b)
                if pa and pb and pa > 0:
                    rows.append(dict(name=n, sector=info["sector"], px=round(pb, 2),
                                     chg=(pb-pa)/pa*100, chg_str=f"{(pb-pa)/pa*100:+.2f}%"))
        if not rows:
            st.warning("No settlements returned for those dates.")
            return
        bdf = pd.DataFrame(rows)
        fig = px.treemap(bdf, path=[px.Constant("Board"), "sector", "name"],
                         values=[1]*len(bdf), color="chg",
                         color_continuous_scale=[(0, RED), (0.5, PANEL), (1, GREEN)],
                         color_continuous_midpoint=0, custom_data=["px", "chg_str"])
        fig.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[0]:.2f}<br>%{customdata[1]}",
            hovertemplate="<b>%{label}</b><br>Settle: %{customdata[0]:.2f}"
                          "<br>Chg: %{customdata[1]}<extra></extra>")
        st.plotly_chart(_styled(fig, 480), use_container_width=True)
        st.caption(f"{len(rows)}/{len(COMMODITIES)} contracts settled on both dates.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FORWARD CURVE
# ══════════════════════════════════════════════════════════════════════════════
def page_curve(commodity, marks):
    st.title(f"Forward Curve — {commodity}")
    st.caption(
        "Live dated settlements pulled month by month off the exchange strip. Expired months "
        "dropped, delivery cycle respected. **Contango** = deferred over prompt (carry market). "
        "**Backwardation** = prompt over deferred (tight prompt, physical squeeze)."
    )
    c = COMMODITIES[commodity]
    mark = require_mark(commodity, marks)
    if mark is None:
        return

    with st.spinner("Pulling forward strip…"):
        strip = fetch_forward_strip(commodity)

    if strip.empty or len(strip) < 2:
        st.error(
            f"**Forward strip unavailable for {commodity}.** Fewer than two dated contracts "
            f"settled. No curve is drawn — this desk will not fit a cost-of-carry model and "
            f"present it as a market."
        )
        return

    f1, fn = float(strip["price"].iloc[0]), float(strip["price"].iloc[-1])
    carry  = (fn - f1) / f1 * 100
    struct = "CONTANGO" if carry > 0.5 else "BACKWARDATION" if carry < -0.5 else "FLAT"
    s_col  = RED if struct == "CONTANGO" else GREEN if struct == "BACKWARDATION" else AMBER
    m1m2   = float(strip["price"].iloc[1]) - f1

    cols = st.columns(5)
    cols[0].metric("M1", f"{f1:,.4f}", strip["label"].iloc[0])
    cols[1].metric("Back", f"{fn:,.4f}", strip["label"].iloc[-1])
    cols[2].metric("M1–M2", f"{m1m2:+,.4f}", "prompt spread")
    cols[3].metric("Carry", f"{carry:+.2f}%", "M1 → back")
    cols[4].metric("Contracts", f"{len(strip)}", "settled")

    st.markdown(
        f'<span class="badge" style="border-color:{s_col};color:{s_col};">⚡ {struct}</span>'
        f'<span class="badge badge-green">● {len(strip)} LIVE CONTRACTS</span>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strip["label"], y=strip["price"], mode="lines+markers", name="Forward strip",
        line=dict(color=AMBER, width=2.5), marker=dict(size=8, color=AMBER),
        customdata=strip["ticker"],
        hovertemplate="<b>%{x}</b><br>%{y:,.4f}<br>%{customdata}<extra></extra>"))
    fig.add_hline(y=mark, line=dict(color=TEXT, dash="dot", width=1.2),
                  annotation_text="Front month", annotation_position="right")
    fig.update_layout(yaxis_title=c["unit"], title=f"{commodity} — exchange strip")
    st.plotly_chart(_styled(fig, 400), use_container_width=True)

    disp = strip[["label", "ticker", "T", "price"]].rename(columns={
        "label": "Contract", "ticker": "Exchange Code",
        "T": "Tenor (yr)", "price": f"Settle ({c['unit']})"})
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: SPREADS & ROLL  (with history + percentile)
# ══════════════════════════════════════════════════════════════════════════════
def page_spreads(commodity, marks):
    st.title(f"Spreads & Roll — {commodity}")
    st.caption(
        "Calendar spreads and roll yield read straight off the live strip — no storage cost or "
        "convenience yield assumed. Positive roll yield = the curve pays you to be long and roll."
    )
    c = COMMODITIES[commodity]
    if require_mark(commodity, marks) is None:
        return

    strip = fetch_forward_strip(commodity)
    if strip.empty or len(strip) < 2:
        st.error("Forward strip unavailable — cannot compute spreads.")
        return

    carry = implied_carry(strip)

    cols = st.columns(4)
    cols[0].metric("M1–M2", f"{carry['spread_vs_M1'].iloc[1]:+,.4f}", c["unit"])
    if len(carry) > 2:
        cols[1].metric("M1–M3", f"{carry['spread_vs_M1'].iloc[2]:+,.4f}", c["unit"])
    if len(carry) > 5:
        cols[2].metric("M1–M6", f"{carry['spread_vs_M1'].iloc[5]:+,.4f}", c["unit"])
    cols[3].metric("Front roll yield", f"{carry['roll_yield'].iloc[1]:+.2f}%", "annualised")

    # ── Spread history + percentile ──────────────────────────────────────────
    st.subheader("Spread History — is this print actually cheap?")
    st.caption(
        "A single M1–M2 number is a point. Its distribution over two years is what tells you "
        "whether it is dislocated. Tracks the two specific dated contracts through time."
    )
    sc1, sc2 = st.columns(2)
    max_leg = min(len(strip) - 1, 11)
    near_i = sc1.selectbox("Near leg", list(range(max_leg)),
                           format_func=lambda i: strip["label"].iloc[i], key="sp_near")
    far_i  = sc2.selectbox("Far leg", list(range(1, max_leg + 1)),
                           index=0, format_func=lambda i: strip["label"].iloc[i], key="sp_far")

    if near_i >= far_i:
        st.warning("Near leg must be before the far leg.")
    else:
        with st.spinner("Pulling spread history…"):
            sh = fetch_spread_history(commodity, near_i, far_i, period="2y")
        if sh.empty:
            st.info(
                "No history for this contract pair. Deferred contracts often have thin or "
                "absent history on a free feed — try a nearer pair."
            )
        else:
            cur = float(sh["spread"].iloc[-1])
            pct, tag, tcol = pctile_badge(sh["spread"], cur)
            m1, m2 = st.columns([1, 3])
            m1.markdown(kpi("Current", f"{cur:+,.4f}", c["unit"], tcol), unsafe_allow_html=True)
            m2.markdown(
                kpi(f"{pct:.0f}th percentile (2y)", tag,
                    f"min {sh['spread'].min():+,.3f} · med {sh['spread'].median():+,.3f} "
                    f"· max {sh['spread'].max():+,.3f}", tcol),
                unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sh.index, y=sh["spread"], name="Spread",
                                     line=dict(color=AMBER, width=2)))
            for q, col_, lab in [(0.8, RED, "80th"), (0.5, GRAY, "median"), (0.2, GREEN, "20th")]:
                fig.add_hline(y=float(sh["spread"].quantile(q)),
                              line=dict(color=col_, dash="dot", width=1),
                              annotation_text=lab, annotation_position="right")
            fig.update_layout(
                title=f"{sh.attrs.get('near_label','near')} − {sh.attrs.get('far_label','far')}",
                yaxis_title=c["unit"])
            st.plotly_chart(_styled(fig, 340), use_container_width=True)

            fig_h = go.Figure(go.Histogram(x=sh["spread"], nbinsx=50, marker_color=BLUE,
                                           opacity=0.75))
            fig_h.add_vline(x=cur, line=dict(color=AMBER, width=2),
                            annotation_text="now")
            fig_h.update_layout(title="2y distribution", xaxis_title=c["unit"])
            st.plotly_chart(_styled(fig_h, 260), use_container_width=True)

    st.subheader("Calendar Spread Ladder (Mn − M1)")
    fig2 = go.Figure(go.Bar(
        x=carry["label"], y=carry["spread_vs_M1"],
        marker_color=np.where(carry["spread_vs_M1"] >= 0, RED, GREEN),
        text=[f"{v:+.3f}" for v in carry["spread_vs_M1"]], textposition="outside"))
    fig2.update_layout(yaxis_title=f"Spread ({c['unit']})")
    st.plotly_chart(_styled(fig2, 320), use_container_width=True)

    st.subheader("Roll Yield Term Structure")
    fig3 = go.Figure(go.Scatter(x=carry["label"], y=carry["roll_yield"], mode="lines+markers",
                                line=dict(color=TEAL, width=2.5), marker=dict(size=7)))
    fig3.add_hline(y=0, line=dict(color=GRAY, dash="dash"))
    fig3.update_layout(yaxis_title="Roll yield (% p.a.)")
    st.plotly_chart(_styled(fig3, 300), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CRACK & CRUSH
# ══════════════════════════════════════════════════════════════════════════════
def page_structures(marks):
    st.title("Crack & Crush")
    st.caption(
        "Processing margins and relative-value structures, built from contracts that are all "
        "live on this desk. Units are normalised before the legs are combined — RB and HO are "
        "converted from $/gal to $/bbl at 42 gal/bbl; meal and oil are converted to $/bu."
    )

    name = st.selectbox("Structure", list(STRUCTURES.keys()), key="struct_sel")
    spec = STRUCTURES[name]
    st.info(spec["desc"], icon="⚗️")

    legs = [n for n, _ in spec["legs"]]
    missing = [l for l in legs if marks.get(l) is None]
    if missing:
        st.error(f"**Cannot price {name}.** No live mark for: {', '.join(missing)}.")
        return

    # ── Live value ───────────────────────────────────────────────────────────
    kind = spec["kind"]
    if kind == "crack":
        val = sum(ratio * to_bbl(n, marks[n]) for n, ratio in spec["legs"]) / spec["divisor"]
    elif kind == "crush":
        meal = marks["Soybean Meal (ZM)"] * CRUSH_MEAL_LB / LB_PER_SHORT_TON
        oil  = marks["Soybean Oil (ZL)"] / 100.0 * CRUSH_OIL_LB
        bean = marks["Soybeans (ZS)"] / 100.0
        val  = meal + oil - bean
    elif kind == "ratio":
        val = marks[legs[0]] / marks[legs[1]]
    else:
        val = sum(ratio * marks[n] for n, ratio in spec["legs"]) / spec["divisor"]

    hist = fetch_structure_history(name, period="3y")
    if hist.empty:
        st.warning("No history available — showing the live print only.")
        pct, tag, tcol = 0.0, "n/a", GRAY
    else:
        pct, tag, tcol = pctile_badge(hist["value"], val)

    lo, hi = spec["typical"]
    in_range = lo <= val <= hi

    cols = st.columns(4)
    cols[0].markdown(kpi("Live", f"{val:,.2f}", spec["unit"], tcol), unsafe_allow_html=True)
    cols[1].markdown(kpi("Percentile (3y)", f"{pct:.0f}th", tag, tcol), unsafe_allow_html=True)
    cols[2].markdown(kpi("Typical range", f"{lo:g} – {hi:g}",
                         "in range" if in_range else "OUTSIDE range",
                         GREEN if in_range else RED), unsafe_allow_html=True)
    cols[3].markdown(kpi("Legs", str(len(spec["legs"])),
                         " / ".join(COMMODITIES[n]["ticker"] for n, _ in spec["legs"]),
                         BLUE), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Leg breakdown ────────────────────────────────────────────────────────
    st.subheader("Leg Breakdown")
    rows = []
    for n, ratio in spec["legs"]:
        c = COMMODITIES[n]
        raw = marks[n]
        if kind == "crack":
            norm_px = to_bbl(n, raw)
            contrib = ratio * norm_px / spec["divisor"]
            note = f"×{c.get('bbl_conv',1.0):g} → $/bbl"
        elif kind == "crush":
            if n == "Soybean Meal (ZM)":
                norm_px = raw * CRUSH_MEAL_LB / LB_PER_SHORT_TON
                note = f"×{CRUSH_MEAL_LB:g} lb / 2000 → $/bu"
                contrib = norm_px
            elif n == "Soybean Oil (ZL)":
                norm_px = raw / 100.0 * CRUSH_OIL_LB
                note = f"c/lb ÷100 ×{CRUSH_OIL_LB:g} lb → $/bu"
                contrib = norm_px
            else:
                norm_px = raw / 100.0
                note = "c/bu ÷100 → $/bu"
                contrib = -norm_px
        elif kind == "ratio":
            norm_px, contrib, note = raw, raw, "—"
        else:
            norm_px = raw
            contrib = ratio * raw / spec["divisor"]
            note = "—"
        rows.append({
            "Leg": n, "Ratio": f"{ratio:+d}" if isinstance(ratio, int) else str(ratio),
            "Raw": raw, "Unit": c["unit"],
            "Normalised": norm_px, "Conversion": note,
            "Contribution": contrib,
        })
    ldf = pd.DataFrame(rows)
    st.dataframe(
        ldf.style.format({"Raw": "{:,.4f}", "Normalised": "{:,.4f}",
                          "Contribution": "{:+,.4f}"}),
        use_container_width=True, hide_index=True)

    # ── History ──────────────────────────────────────────────────────────────
    if not hist.empty:
        st.subheader("3-Year History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["value"], name=name,
                                 line=dict(color=AMBER, width=2)))
        for q, col_, lab in [(0.8, RED, "80th"), (0.5, GRAY, "median"), (0.2, GREEN, "20th")]:
            fig.add_hline(y=float(hist["value"].quantile(q)),
                          line=dict(color=col_, dash="dot", width=1),
                          annotation_text=lab, annotation_position="right")
        fig.add_hline(y=val, line=dict(color=TEAL, width=2),
                      annotation_text="live", annotation_position="left")
        fig.update_layout(yaxis_title=spec["unit"])
        st.plotly_chart(_styled(fig, 380), use_container_width=True)

        cA, cB = st.columns(2)
        with cA:
            fh = go.Figure(go.Histogram(x=hist["value"], nbinsx=50,
                                        marker_color=BLUE, opacity=0.75))
            fh.add_vline(x=val, line=dict(color=AMBER, width=2), annotation_text="now")
            fh.update_layout(title="Distribution", xaxis_title=spec["unit"])
            st.plotly_chart(_styled(fh, 300), use_container_width=True)
        with cB:
            # Seasonal profile of the structure — cracks are violently seasonal.
            h = hist.copy()
            h["month"] = h.index.month
            mm = h.groupby("month")["value"].median()
            fs = go.Figure(go.Bar(x=[MONTH_NAMES[m-1] for m in mm.index], y=mm.values,
                                  marker_color=TEAL))
            fs.update_layout(title="Median by calendar month", yaxis_title=spec["unit"])
            st.plotly_chart(_styled(fs, 300), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
def page_correlation():
    st.title("Correlation")
    st.caption(
        "Daily log-return correlation across the board. This is what the Risk page uses to turn "
        "an undiversified sum of position VaRs into a real portfolio number — a long WTI / short "
        "Brent book should net down to almost nothing, and only the correlation matrix knows that."
    )

    col1, col2 = st.columns(2)
    period = col1.selectbox("Sample", ["1y", "2y", "3y", "5y"], index=1, key="corr_period")
    window = col2.slider("Window (trading days)", 60, 756, 252, 21, key="corr_window")

    with st.spinner("Building correlation matrix…"):
        corr = correlation_matrix(period, window)

    if corr.empty:
        st.error("Insufficient overlapping history to build a correlation matrix.")
        return

    st.caption(f"{len(corr)} contracts · {window} trading days · {period} sample")

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, RED], [0.5, PANEL], [1, GREEN]], zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=9), colorbar=dict(title="ρ")))
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT))
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Most and least correlated pairs — where the diversification actually lives.
    pairs = []
    cols_ = list(corr.columns)
    for i in range(len(cols_)):
        for j in range(i+1, len(cols_)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append(dict(A=cols_[i], B=cols_[j], rho=float(v)))
    pdf = pd.DataFrame(pairs).sort_values("rho", ascending=False)

    cA, cB = st.columns(2)
    with cA:
        st.subheader("Most correlated")
        st.caption("Hedge each other. Spread trades live here.")
        st.dataframe(pdf.head(10).style.format({"rho": "{:+.3f}"}),
                     use_container_width=True, hide_index=True)
    with cB:
        st.subheader("Least correlated")
        st.caption("Genuine diversification. VaR nets down across these.")
        st.dataframe(pdf.tail(10).sort_values("rho").style.format({"rho": "{:+.3f}"}),
                     use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: SEASONALITY
# ══════════════════════════════════════════════════════════════════════════════
def page_seasonality(commodity, marks):
    st.title(f"Seasonality — {commodity}")
    c = COMMODITIES[commodity]

    if not c.get("seasonal"):
        st.info(
            f"**{commodity} is not flagged seasonal.** Precious metals and the crude complex "
            f"have no reliable calendar pattern — showing one would invite reading noise as signal. "
            f"Seasonal contracts on this desk: "
            f"{', '.join(n for n, x in COMMODITIES.items() if x.get('seasonal'))}."
        )
        return

    st.caption(
        "Monthly return distribution over 10 years. Gas, gasoline, distillate, the grains and "
        "hogs all have genuine calendar structure — injection/withdrawal, driving season, "
        "harvest pressure, herd cycles. This is descriptive, not predictive: a strong median "
        "with a wide box is not a trade."
    )

    years = st.slider("Lookback (years)", 5, 15, 10, key="seas_years")
    with st.spinner("Pulling history…"):
        s = seasonality(c["yf_ticker"], years)

    if s.empty:
        st.error("Insufficient history for a seasonal profile.")
        return

    st.caption(f"{len(s)} monthly observations · {s['year'].min()}–{s['year'].max()}")

    fig = go.Figure()
    for m in range(1, 13):
        d = s[s["month"] == m]["ret"]
        if d.empty:
            continue
        med = float(d.median())
        fig.add_trace(go.Box(y=d, name=MONTH_NAMES[m-1],
                             marker_color=GREEN if med >= 0 else RED,
                             line=dict(width=1.5), boxmean=True))
    fig.add_hline(y=0, line=dict(color=GRAY, dash="dash"))
    fig.update_layout(title="Monthly return distribution (%)", yaxis_title="Return (%)",
                      showlegend=False)
    st.plotly_chart(_styled(fig, 420), use_container_width=True)

    stats = s.groupby("month")["ret"].agg(
        Median="median", Mean="mean", StdDev="std",
        HitRate=lambda x: (x > 0).mean() * 100, N="count").reset_index()
    stats["Month"] = [MONTH_NAMES[m-1] for m in stats["month"]]
    stats = stats[["Month", "Median", "Mean", "StdDev", "HitRate", "N"]]

    cA, cB = st.columns([3, 2])
    with cA:
        fig2 = go.Figure(go.Bar(
            x=stats["Month"], y=stats["Median"],
            marker_color=np.where(stats["Median"] >= 0, GREEN, RED),
            text=[f"{v:+.1f}%" for v in stats["Median"]], textposition="outside"))
        fig2.update_layout(title="Median return by month", yaxis_title="%")
        st.plotly_chart(_styled(fig2, 320), use_container_width=True)
    with cB:
        st.dataframe(
            stats.style.format({"Median": "{:+.2f}%", "Mean": "{:+.2f}%",
                                "StdDev": "{:.2f}%", "HitRate": "{:.0f}%"}),
            use_container_width=True, hide_index=True, height=320)

    # Cumulative seasonal path — the shape traders actually carry in their heads.
    st.subheader("Average Seasonal Path")
    st.caption("Cumulative median return through the calendar year, rebased to Jan = 100.")
    path = [100.0]
    for m in range(1, 13):
        med = float(stats[stats["Month"] == MONTH_NAMES[m-1]]["Median"].iloc[0])
        path.append(path[-1] * (1 + med/100))
    fig3 = go.Figure(go.Scatter(x=["Start"] + MONTH_NAMES, y=path, mode="lines+markers",
                                line=dict(color=AMBER, width=2.5), marker=dict(size=7)))
    fig3.add_hline(y=100, line=dict(color=GRAY, dash="dash"))
    fig3.update_layout(yaxis_title="Index (Jan = 100)")
    st.plotly_chart(_styled(fig3, 300), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: EIA FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════════════════
def page_eia(commodity, marks):
    st.title("EIA Fundamentals")

    key = st.session_state.get("eia_key", "")
    if not key:
        st.warning(
            "**No EIA API key set.** Enter one in the sidebar to pull real US crude, product "
            "and natural gas inventories. The key is free at "
            "[eia.gov/opendata](https://www.eia.gov/opendata/). Without it this page stays "
            "empty — no modelled stock levels are shown.",
            icon="🔑")
        return

    if not REQUESTS_AVAILABLE:
        st.error("`requests` is not installed. Run `pip install requests`.")
        return

    relevant = EIA_MAP.get(commodity, [])
    if relevant:
        st.caption(f"Series relevant to **{commodity}** are pre-selected. "
                   f"EIA covers US energy only — there is no free fundamental feed for "
                   f"metals, softs or livestock.")
    else:
        st.info(
            f"**EIA does not cover {commodity}.** Its dataset is US energy only. "
            f"Select an energy contract in the sidebar, or pick series manually below."
        )

    chosen = st.multiselect("Series", list(EIA_SERIES.keys()),
                            default=relevant if relevant else ["US Crude Stocks (ex-SPR)"],
                            key="eia_series")
    if not chosen:
        return

    for name in chosen:
        meta = EIA_SERIES[name]
        with st.spinner(f"Pulling {name}…"):
            df = fetch_eia(name, key)

        if df.empty:
            st.error(f"**{name}** — feed returned nothing. Check the API key, or the series "
                     f"may have been retired by EIA.")
            continue

        st.subheader(name)
        last  = float(df["value"].iloc[-1])
        prev  = float(df["value"].iloc[-2]) if len(df) > 1 else last
        wow   = last - prev
        yr    = df[df.index >= df.index.max() - pd.Timedelta(days=365)]
        yr_avg = float(yr["value"].mean())
        z = ((last - yr_avg) / yr["value"].std()) if yr["value"].std() > 0 else 0.0

        cols = st.columns(4)
        cols[0].metric("Latest", f"{last:,.0f}", meta["unit"])
        cols[1].metric("W/W", f"{wow:+,.0f}",
                       "build" if wow > 0 else "draw" if wow < 0 else "flat")
        cols[2].metric("1y average", f"{yr_avg:,.0f}", meta["unit"])
        cols[3].metric("Z-score vs 1y", f"{z:+.2f}σ",
                       "tight" if z < -1 else "loose" if z > 1 else "normal")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["value"], name=name,
                                 line=dict(color=AMBER, width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df["value"].rolling(52).mean(),
                                 name="52w MA", line=dict(color=BLUE, width=1.2, dash="dot")))
        fig.update_layout(yaxis_title=meta["unit"])
        st.plotly_chart(_styled(fig, 320), use_container_width=True)

        # 5-year band — the standard way a desk reads an inventory print.
        h = df.copy()
        h["year"], h["week"] = h.index.year, h.index.isocalendar().week
        recent = h[h["year"] >= h["year"].max() - 5]
        band = recent.groupby("week")["value"].agg(["min", "mean", "max"])
        cur_yr = h[h["year"] == h["year"].max()]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=band.index, y=band["max"], name="5y max",
                                  line=dict(color=GRAY, width=0.5)))
        fig2.add_trace(go.Scatter(x=band.index, y=band["min"], name="5y range",
                                  line=dict(color=GRAY, width=0.5), fill="tonexty",
                                  fillcolor="rgba(139,148,158,0.15)"))
        fig2.add_trace(go.Scatter(x=band.index, y=band["mean"], name="5y avg",
                                  line=dict(color=BLUE, width=1.5, dash="dot")))
        fig2.add_trace(go.Scatter(x=cur_yr["week"], y=cur_yr["value"], name="Current year",
                                  line=dict(color=AMBER, width=2.5)))
        fig2.update_layout(title="Current year vs 5-year range",
                           xaxis_title="ISO week", yaxis_title=meta["unit"])
        st.plotly_chart(_styled(fig2, 340), use_container_width=True)

        # Inventory vs price — the actual reason a trader opens this page.
        c = COMMODITIES[commodity]
        px_h = fetch_history(c["yf_ticker"], period="3y")
        if not px_h.empty:
            merged = pd.DataFrame({"stock": df["value"]}).join(
                pd.DataFrame({"px": px_h["Close"]}), how="inner").dropna()
            if len(merged) > 30:
                rho = float(np.corrcoef(merged["stock"], merged["px"])[0, 1])
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=merged.index, y=merged["stock"], name=name,
                                          line=dict(color=TEAL, width=2)))
                fig3.add_trace(go.Scatter(x=merged.index, y=merged["px"], name=f"{c['ticker']} settle",
                                          line=dict(color=AMBER, width=2), yaxis="y2"))
                fig3.update_layout(
                    title=f"Inventory vs price — ρ = {rho:+.2f}",
                    yaxis=dict(title=meta["unit"]),
                    yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                title=c["unit"]))
                st.plotly_chart(_styled(fig3, 340), use_container_width=True)
                st.caption(
                    "Negative ρ is the textbook relationship — stocks build, price falls. "
                    "A positive reading usually means the sample is dominated by a demand "
                    "shock rather than a supply one."
                )
        st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: REGIONAL BALANCES
# ══════════════════════════════════════════════════════════════════════════════
def page_regional(commodity, marks):
    c  = COMMODITIES[commodity]
    ru, rl = c["reg_unit"], c["reg_label"]
    st.title(f"Regional Balances — {commodity}")

    if commodity not in REGIONAL_DATA:
        st.info(
            f"**No regional balance is maintained for {commodity}.** Coverage exists for: "
            f"{', '.join(REGIONAL_DATA)}. Rather than show another contract's flows under this "
            f"heading, this page stays empty. Wire in an IEA / USDA / WBMS feed to populate it."
        )
        return

    st.caption(
        f"Annual supply/demand by region in **{rl} ({ru})**. Green = net exporter, "
        f"red = net importer, bubble = size of the imbalance. Static estimates, not a live feed."
    )

    reg = pd.DataFrame(REGIONAL_DATA[commodity])
    reg["net"] = reg["supply"] - reg["demand"]
    reg["status"] = np.where(reg["net"] > 0, "Exporter", "Importer")
    ws, wd = float(reg["supply"].sum()), float(reg["demand"].sum())

    cols = st.columns(4)
    cols[0].metric(f"World supply ({ru})", f"{ws:,.1f}")
    cols[1].metric(f"World demand ({ru})", f"{wd:,.1f}")
    cols[2].metric(f"Balance ({ru})", f"{ws-wd:+,.2f}",
                   "surplus" if ws > wd else "deficit")
    cols[3].metric("Regions", str(len(reg)))

    fig = go.Figure()
    for _, r in reg.iterrows():
        fig.add_trace(go.Scattergeo(
            lat=[r["lat"]], lon=[r["lon"]], mode="markers+text",
            marker=dict(size=abs(r["net"])**0.5 * 4 + 8,
                        color=GREEN if r["net"] >= 0 else RED,
                        opacity=0.75, line=dict(color=BORDER, width=1)),
            text=r["region"], textposition="top center",
            textfont=dict(size=10, color=TEXT, family="JetBrains Mono"),
            name=r["region"],
            hovertemplate=(f"<b>{r['region']}</b><br>Supply: {r['supply']:.1f} {ru}"
                           f"<br>Demand: {r['demand']:.1f} {ru}"
                           f"<br>Net: {r['net']:+.1f} {ru}<extra></extra>")))
    fig.update_layout(
        geo=dict(bgcolor=BG, showframe=False, showcoastlines=True, coastlinecolor=BORDER,
                 landcolor=PANEL, oceancolor=BG, showocean=True, showland=True,
                 projection_type="natural earth"),
        paper_bgcolor="rgba(0,0,0,0)", height=420,
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure(go.Bar(x=reg["region"], y=reg["net"],
                            marker_color=np.where(reg["net"] >= 0, GREEN, RED),
                            text=[f"{v:+.1f}" for v in reg["net"]], textposition="outside"))
    fig2.update_layout(title=f"Net trade position — {ru}", yaxis_title=rl)
    st.plotly_chart(_styled(fig2, 300), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: OPTIONS & GREEKS
# ══════════════════════════════════════════════════════════════════════════════
def page_options(commodity, marks):
    st.title(f"Options & Greeks — {commodity}")
    st.warning(
        "**Vol is an input, not a market quote.** No listed option chain is pulled — Yahoo does "
        "not serve chains for futures, only for equities and ETFs. σ defaults to 60-day realised, "
        "which is a starting point, not a broker mark. Getting a real implied surface needs a "
        "paid feed (CME DataMine, Refinitiv, Bloomberg).",
        icon="⚠️")
    c = COMMODITIES[commodity]
    mark = require_mark(commodity, marks)
    if mark is None:
        return

    T_m   = st.session_state.get("opt_T_months", 6)
    K_pct = st.session_state.get("opt_K_pct", 100)
    r_pct = st.session_state.get("opt_r_pct", 5)

    rv = realised_vol(c["yf_ticker"], 60)
    default_vol = int((rv or c["vol"]) * 100)

    # Anchor the forward on the matching dated contract where the strip has one.
    strip = fetch_forward_strip(commodity)
    F_default, F_label = mark, "front month"
    if not strip.empty:
        near = strip.iloc[(strip["T"] - T_m/12).abs().argsort()].iloc[0]
        F_default, F_label = float(near["price"]), f"{near['label']} (strip)"

    col1, col2, col3 = st.columns(3)
    F     = col1.number_input(f"Forward F — {F_label}", value=float(F_default),
                              step=float(F_default*0.005), format="%.4f")
    K     = col2.number_input("Strike K", value=float(F_default*K_pct/100),
                              step=float(F_default*0.005), format="%.4f")
    vol_p = col3.number_input("Vol σ (%)", value=default_vol, step=1)

    T, sigma, r = T_m/12, vol_p/100, r_pct/100
    call = black76(F, K, T, r, sigma, "call")
    put  = black76(F, K, T, r, sigma, "put")
    pcp  = call["price"] - put["price"] - math.exp(-r*T)*(F - K)
    mny  = "ITM" if F > K else "OTM" if F < K else "ATM"

    rv_badge = f'<span class="badge">RV60={rv*100:.1f}%</span>' if rv else ""
    st.markdown(
        f'<span class="badge badge-amber">{mny} call</span>'
        f'<span class="badge">T={T:.3f}y</span>'
        f'<span class="badge">σ={sigma*100:.1f}%</span>{rv_badge}'
        f'<span class="badge badge-green">put-call parity ✓ {pcp:.2e}</span>',
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    mult = price_multiplier(commodity)
    tab_c, tab_p = st.tabs(["Call", "Put"])
    for tab, g, acc in [(tab_c, call, AMBER), (tab_p, put, BLUE)]:
        with tab:
            cols = st.columns(6)
            for col, l, k in zip(cols, ["Premium", "Delta", "Gamma", "Vega", "Theta", "Rho"],
                                 ["price", "delta", "gamma", "vega", "theta", "rho"]):
                col.markdown(kpi(l, f"{g[k]:.5f}", "", acc), unsafe_allow_html=True)
            st.markdown("<br>**Per lot (cash)**", unsafe_allow_html=True)
            cols = st.columns(4)
            cols[0].markdown(kpi("Premium/lot", f"${g['price']*mult:,.0f}", "", acc),
                             unsafe_allow_html=True)
            cols[1].markdown(kpi("Delta/lot", f"${g['delta']*mult:,.0f}", "per unit move", acc),
                             unsafe_allow_html=True)
            cols[2].markdown(kpi("Vega/lot", f"${g['vega']*mult:,.0f}", "per vol point", acc),
                             unsafe_allow_html=True)
            cols[3].markdown(kpi("Theta/lot", f"${g['theta']*mult:,.0f}", "per day", acc),
                             unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    strikes = np.linspace(F*0.55, F*1.45, 80)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=np.maximum(strikes-K, 0) - call["price"],
                             name="Long call", line=dict(color=GREEN, width=2.5)))
    fig.add_trace(go.Scatter(x=strikes, y=np.maximum(K-strikes, 0) - put["price"],
                             name="Long put", line=dict(color=RED, width=2.5)))
    fig.add_hline(y=0, line=dict(color=GRAY, dash="dash", width=1))
    fig.add_vline(x=K, line=dict(color=AMBER, dash="dot", width=1.5), annotation_text="K")
    fig.add_vline(x=F, line=dict(color=BLUE, dash="dot", width=1.5), annotation_text="F")
    fig.update_layout(title="Expiry payoff, net of premium", xaxis_title=c["unit"])
    st.plotly_chart(_styled(fig, 360), use_container_width=True)

    st.subheader("Greeks vs Strike")
    ks = np.linspace(F*0.7, F*1.3, 60)
    t1, t2, t3 = st.tabs(["Delta", "Gamma", "Vega"])
    for tab, gk, col_ in [(t1, "delta", AMBER), (t2, "gamma", PURPLE), (t3, "vega", TEAL)]:
        with tab:
            f_ = go.Figure()
            f_.add_trace(go.Scatter(x=ks, y=[black76(F, k, T, r, sigma, "call")[gk] for k in ks],
                                    name=f"Call {gk}", line=dict(color=col_, width=2)))
            f_.add_trace(go.Scatter(x=ks, y=[black76(F, k, T, r, sigma, "put")[gk] for k in ks],
                                    name=f"Put {gk}", line=dict(color=BLUE, width=2)))
            f_.add_vline(x=K, line=dict(color=AMBER, dash="dot"))
            st.plotly_chart(_styled(f_, 280), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: VOL SURFACE
# ══════════════════════════════════════════════════════════════════════════════
def page_vol_surface(commodity, marks):
    st.title(f"Vol Surface — {commodity}")
    st.warning(
        "**Parametric shape, not a calibrated market.** No listed chain is pulled (Yahoo serves "
        "chains for equities, not futures). ATM is seeded from 60-day realised; skew, curvature "
        "and vol-of-vol are your inputs. Use it to stress a book, not to quote one.",
        icon="⚠️")
    c = COMMODITIES[commodity]
    mark = require_mark(commodity, marks)
    if mark is None:
        return

    rv = realised_vol(c["yf_ticker"], 60)
    seed = int((rv or c["vol"]) * 100)

    col1, col2, col3, col4 = st.columns(4)
    atm  = col1.slider("ATM σ (%)", 5, 120, seed, key="vs_atm") / 100
    skew = col2.slider("Skew ×100", -20, 20, -5, key="vs_skew") / 100
    curv = col3.slider("Curvature ×100", 0, 10, 2, key="vs_curv") / 100
    vov  = col4.slider("Vol-of-vol", 0, 100, 15, key="vs_vov") / 100

    if rv:
        st.caption(f"ATM seeded from RV60 = {rv*100:.1f}%. Registry prior: {c['vol']*100:.0f}%.")

    mats, Kgrid, Z = vol_surface_fn(mark, atm, skew, curv, vov)
    labels = ["1M", "2M", "3M", "6M", "9M", "12M", "18M", "24M"]

    fig = go.Figure(data=go.Surface(
        z=Z, x=np.log(Kgrid/mark), y=[m*12 for m in mats],
        colorscale=[[0, BLUE], [0.5, PURPLE], [1, AMBER]],
        colorbar=dict(title="σ", tickfont=dict(color=TEXT))))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="ln(K/F)", color=GRAY, gridcolor=BORDER, backgroundcolor=BG),
            yaxis=dict(title="Tenor (months)", color=GRAY, gridcolor=BORDER, backgroundcolor=BG),
            zaxis=dict(title="σ", color=GRAY, gridcolor=BORDER, backgroundcolor=BG), bgcolor=BG),
        paper_bgcolor="rgba(0,0,0,0)", height=520,
        margin=dict(l=10, r=10, t=10, b=10), font=dict(color=TEXT))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Smile by Tenor")
    fig2 = go.Figure()
    for row, lab, col_ in zip(Z, labels, [AMBER, BLUE, GREEN, RED, PURPLE, TEAL, GRAY, TEXT]):
        fig2.add_trace(go.Scatter(x=np.log(Kgrid/mark), y=row*100, name=lab,
                                  line=dict(color=col_, width=2)))
    fig2.update_layout(xaxis_title="ln(K/F)", yaxis_title="σ (%)")
    st.plotly_chart(_styled(fig2, 360), use_container_width=True)

    fig3 = go.Figure(go.Bar(x=labels, y=Z[:, Z.shape[1]//2]*100, marker_color=AMBER))
    fig3.update_layout(title="ATM term structure", yaxis_title="σ (%)")
    st.plotly_chart(_styled(fig3, 260), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: BLOTTER
# ══════════════════════════════════════════════════════════════════════════════
def page_blotter(marks):
    st.title("Blotter")
    st.caption(
        "Futures and options, in **lots**, marked to the live front month. P&L is cash. "
        "The book persists to disk — it survives a refresh, and can be exported to move "
        "between machines."
    )

    # Load from disk once per session.
    if "positions" not in st.session_state:
        st.session_state["positions"] = blotter_load()
        if st.session_state["positions"]:
            st.success(f"Restored {len(st.session_state['positions'])} position(s) from disk.",
                       icon="💾")

    tab_f, tab_o = st.tabs(["Book future", "Book option"])

    with tab_f:
        c1, c2, c3, c4, c5 = st.columns(5)
        name = c1.selectbox("Contract", list(COMMODITIES.keys()), key="pos_name")
        side = c2.selectbox("Side", ["Long", "Short"], key="pos_side")
        lots = c3.number_input("Lots", value=1, step=1, min_value=1, key="pos_lots")
        mk   = marks.get(name)
        entry = c4.number_input("Entry", value=float(mk) if mk else 0.0, step=0.01,
                                format="%.4f", key="pos_entry", disabled=(mk is None))
        if c5.button("Book", use_container_width=True, type="primary",
                     disabled=(mk is None), key="book_fut"):
            st.session_state["positions"].append(dict(
                kind="future", commodity=name, side=side, lots=int(lots),
                entry=float(entry), vol=COMMODITIES[name]["vol"]))
            blotter_save(st.session_state["positions"])
            st.rerun()
        if mk is None:
            st.caption(f"⚠️ {name} has no live mark — cannot book.")
        else:
            st.caption(f"1 lot = {COMMODITIES[name]['contract_size']:,} "
                       f"{COMMODITIES[name]['size_unit']} ≈ "
                       f"${notional_per_lot(name, mk):,.0f} notional at {mk:,.4f}")

    with tab_o:
        st.caption(
            "Options are priced Black-76 off the live forward. **σ is your input** — no listed "
            "chain exists on a free feed, so the vol you book is an assumption, not a market."
        )
        o1, o2, o3, o4 = st.columns(4)
        oname = o1.selectbox("Contract", list(COMMODITIES.keys()), key="opt_name")
        oside = o2.selectbox("Side", ["Long", "Short"], key="opt_side")
        otype = o3.selectbox("Type", ["call", "put"], key="opt_type")
        olots = o4.number_input("Lots", value=1, step=1, min_value=1, key="opt_lots")

        omk = marks.get(oname)
        p1, p2, p3, p4 = st.columns(4)
        ostrike = p1.number_input("Strike", value=float(omk) if omk else 0.0,
                                  step=0.01, format="%.4f", key="opt_strike",
                                  disabled=(omk is None))
        otenor = p2.slider("Tenor (months)", 1, 24, 6, key="opt_tenor") / 12
        orv = realised_vol(COMMODITIES[oname]["yf_ticker"], 60) if omk else None
        ovol = p3.number_input("σ (%)", value=int((orv or COMMODITIES[oname]["vol"])*100),
                               step=1, key="opt_vol") / 100

        prem = 0.0
        if omk:
            prem = black76(omk, ostrike, otenor, 0.05, ovol, otype)["price"]
        oentry = p4.number_input("Premium paid", value=float(round(prem, 4)),
                                 step=0.0001, format="%.4f", key="opt_entry",
                                 disabled=(omk is None))

        if st.button("Book option", type="primary", disabled=(omk is None), key="book_opt"):
            st.session_state["positions"].append(dict(
                kind="option", commodity=oname, side=oside, lots=int(olots),
                entry=float(oentry), opt_type=otype, strike=float(ostrike),
                tenor=float(otenor), vol=float(ovol)))
            blotter_save(st.session_state["positions"])
            st.rerun()
        if omk:
            st.caption(f"Theoretical premium at σ={ovol*100:.0f}%: {prem:,.4f} "
                       f"→ ${prem*price_multiplier(oname):,.0f} per lot")

    positions = st.session_state["positions"]
    if not positions:
        st.info("Blotter empty.")
        _blotter_io()
        return

    # ── Mark the book ────────────────────────────────────────────────────────
    rows, tot, gl, gs = [], 0.0, 0.0, 0.0
    for p in positions:
        n = p["commodity"]
        mark = marks.get(n)
        if mark is None:
            continue
        sign = 1 if p["side"] == "Long" else -1
        mult = price_multiplier(n)

        if p.get("kind", "future") == "option":
            theo = black76(mark, p["strike"], p["tenor"], 0.05, p["vol"], p["opt_type"])["price"]
            pnl  = sign * (theo - p["entry"]) * mult * p["lots"]
            notl = theo * mult * p["lots"]
            label = f"{n} {p['opt_type'][:1].upper()}{p['strike']:g} {p['tenor']*12:.0f}m"
            entry_disp, mark_disp = p["entry"], theo
        else:
            pnl  = sign * (mark - p["entry"]) * mult * p["lots"]
            notl = notional_per_lot(n, mark) * p["lots"]
            label = n
            entry_disp, mark_disp = p["entry"], mark

        tot += pnl
        if sign > 0:
            gl += abs(notl)
        else:
            gs += abs(notl)
        rows.append({
            "Position": label, "Kind": p.get("kind", "future"), "Side": p["side"],
            "Lots": p["lots"], "Entry": entry_disp, "Mark": mark_disp,
            "Notional": notl, "P&L": pnl,
            "Return %": sign*(mark_disp - entry_disp)/entry_disp*100 if entry_disp else 0.0,
        })

    cols = st.columns(4)
    cols[0].metric("Gross long", f"${gl:,.0f}")
    cols[1].metric("Gross short", f"${gs:,.0f}")
    cols[2].metric("Net exposure", f"${gl-gs:+,.0f}")
    cols[3].metric("Open P&L", f"${tot:+,.0f}", f"{len(rows)} lines")

    df = pd.DataFrame(rows)

    def _c(v):
        if isinstance(v, (int, float)):
            return f"color:{GREEN}" if v >= 0 else f"color:{RED}"
        return ""

    st.dataframe(
        df.style.format({"Entry": "{:,.4f}", "Mark": "{:,.4f}", "Notional": "${:,.0f}",
                         "P&L": "${:+,.0f}", "Return %": "{:+.2f}%"})
          .map(_c, subset=["P&L", "Return %"]),
        use_container_width=True, hide_index=True)

    # ── Price vs roll decomposition ──────────────────────────────────────────
    st.subheader("P&L Attribution — Price vs Carry")
    st.caption(
        "A single P&L number hides which of two opposite things is happening. **Price P&L** is "
        "the mark moving. **Roll P&L** is what the curve pays (or charges) you to hold the "
        "position: a long in backwardation earns it, a long in contango bleeds it. They have "
        "opposite implications for whether to stay in the trade."
    )
    with st.spinner("Pulling strips for carry attribution…"):
        rp = roll_pnl(positions, marks)

    if not rp:
        st.info("No futures positions to attribute (options carry no roll).")
    else:
        rpdf = pd.DataFrame(rp)
        tot_price = float(rpdf["PricePnL"].sum())
        tot_roll  = float(rpdf["MonthlyRoll"].sum())

        m = st.columns(3)
        m[0].markdown(kpi("Price P&L", f"${tot_price:+,.0f}", "mark vs entry",
                          GREEN if tot_price >= 0 else RED), unsafe_allow_html=True)
        m[1].markdown(kpi("Roll P&L (1 month)", f"${tot_roll:+,.0f}",
                          "curve carry if held & rolled",
                          GREEN if tot_roll >= 0 else RED), unsafe_allow_html=True)
        m[2].markdown(kpi("Combined", f"${tot_price + tot_roll:+,.0f}", "",
                          GREEN if tot_price + tot_roll >= 0 else RED), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.dataframe(
            rpdf.style.format({"PricePnL": "${:+,.0f}", "MonthlyRoll": "${:+,.0f}",
                               "M1M2": "{:+,.4f}", "RollAnnPct": "{:+.1f}%"})
                .map(_c, subset=["PricePnL", "MonthlyRoll"]),
            use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=rpdf["Contract"], y=rpdf["PricePnL"], name="Price P&L",
                             marker_color=AMBER))
        fig.add_trace(go.Bar(x=rpdf["Contract"], y=rpdf["MonthlyRoll"], name="Roll P&L (1m)",
                             marker_color=TEAL))
        fig.add_hline(y=0, line=dict(color=GRAY, dash="dash"))
        fig.update_layout(barmode="group", yaxis_title="$")
        st.plotly_chart(_styled(fig, 320), use_container_width=True)

    # ── Aggregate Greeks ─────────────────────────────────────────────────────
    has_opt = any(p.get("kind") == "option" for p in positions)
    st.subheader("Net Book Greeks")
    if not has_opt:
        st.info(
            "No options booked. Futures carry delta 1.0 and nothing else — net delta is "
            "simply the signed notional. Book an option to see gamma, vega and theta.",
            icon="ℹ️")

    bg = book_greeks(positions, marks)
    t = bg["total"]
    g = st.columns(4)
    g[0].markdown(kpi("Net Delta", f"${t['delta']:+,.0f}", "per 1 unit move", AMBER),
                  unsafe_allow_html=True)
    g[1].markdown(kpi("Net Gamma", f"${t['gamma']:+,.0f}", "per unit²",
                      GREEN if t["gamma"] >= 0 else RED), unsafe_allow_html=True)
    g[2].markdown(kpi("Net Vega", f"${t['vega']:+,.0f}", "per vol point",
                      GREEN if t["vega"] >= 0 else RED), unsafe_allow_html=True)
    g[3].markdown(kpi("Net Theta", f"${t['theta']:+,.0f}", "per day",
                      GREEN if t["theta"] >= 0 else RED), unsafe_allow_html=True)

    if has_opt:
        st.markdown("<br>", unsafe_allow_html=True)
        if t["gamma"] < 0:
            st.warning(
                f"**Short gamma.** The book loses on large moves in either direction and is "
                f"collecting ${t['theta']:+,.0f}/day of theta to compensate. That trade works "
                f"until it doesn't.", icon="⚠️")
        elif t["gamma"] > 0:
            st.info(
                f"**Long gamma.** The book gains on large moves either way and pays "
                f"${abs(t['theta']):,.0f}/day of theta for the privilege.", icon="ℹ️")
        st.dataframe(
            pd.DataFrame(bg["rows"]).style.format({
                "Delta": "${:+,.0f}", "Gamma": "${:+,.2f}",
                "Vega": "${:+,.0f}", "Theta": "${:+,.0f}"}),
            use_container_width=True, hide_index=True)

    _blotter_io()


def _blotter_io():
    """Export / import / flatten controls."""
    st.markdown("---")
    st.subheader("Book Management")
    positions = st.session_state.get("positions", [])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇ Export book (JSON)", data=blotter_serialise(positions),
            file_name=f"blotter_{date.today().isoformat()}.json",
            mime="application/json", use_container_width=True,
            disabled=not positions)
    with c2:
        up = st.file_uploader("⬆ Import book", type="json", key="blotter_up",
                              label_visibility="collapsed")
        if up is not None:
            parsed = blotter_deserialise(up.read().decode("utf-8"))
            if parsed is None:
                st.error("Not a valid blotter file.")
            else:
                st.session_state["positions"] = parsed
                blotter_save(parsed)
                st.success(f"Imported {len(parsed)} position(s).")
                st.rerun()
    with c3:
        if st.button("🗑 Flatten book", use_container_width=True, disabled=not positions):
            st.session_state["positions"] = []
            blotter_save([])
            st.rerun()

    st.caption(f"Book auto-saves to `{BLOTTER_FILE}` on every trade. "
               f"Positions in contracts no longer on the desk are dropped on load.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RISK  (correlation-aware)
# ══════════════════════════════════════════════════════════════════════════════
def page_risk(marks):
    st.title("Risk")
    positions = st.session_state.get("positions", [])
    if not positions:
        st.warning("No positions. Book something on the Blotter first.")
        return

    col1, col2, col3, col4 = st.columns(4)
    conf    = col1.selectbox("Confidence", [0.90, 0.95, 0.99], index=1)
    horizon = col2.slider("Horizon (days)", 1, 30, 1, key="risk_h")
    vol_src = col3.selectbox("Vol source", ["60d realised", "Registry prior"], index=0)
    method  = col4.selectbox("Method", ["Correlated (Σ)", "Undiversified sum"], index=0)

    if vol_src == "60d realised":
        for p in positions:
            rv = realised_vol(COMMODITIES[p["commodity"]]["yf_ticker"], 60)
            p["vol"] = rv if rv else COMMODITIES[p["commodity"]]["vol"]
    else:
        for p in positions:
            p["vol"] = COMMODITIES[p["commodity"]]["vol"]

    with st.spinner("Building correlation matrix…"):
        corr = correlation_matrix("2y", 252)

    risk = portfolio_var(positions, marks, corr, conf=conf, horizon=horizon,
                         diversified=(method == "Correlated (Σ)"))
    if not risk["rows"]:
        st.error("No live marks for any booked position.")
        return

    if risk["corr_used"]:
        st.success(
            f"**Correlation matrix applied.** σₚ = √(wᵀΣw) across {len(risk['rows'])} positions. "
            f"Diversification benefit: **{risk['benefit']:.1f}%** versus summing standalone VaRs. "
            f"A long WTI / short Brent book nets down here — the naive sum cannot see that.",
            icon="🔗")
    else:
        st.warning(
            "**No usable correlation matrix — falling back to the undiversified sum.** "
            "Position VaRs are added, which overstates risk. This desk will not silently assume "
            "independence, since that would *understate* it.",
            icon="⚠️")

    cols = st.columns(5)
    cols[0].metric(f"VaR {int(conf*100)}", f"${risk['var']:,.0f}", f"{horizon}d")
    cols[1].metric(f"ES {int(conf*100)}", f"${risk['es']:,.0f}", "expected shortfall")
    cols[2].metric("Undiversified", f"${risk['undiversified']:,.0f}", "naive sum")
    cols[3].metric("Diversification", f"{risk['benefit']:.1f}%", "benefit")
    cols[4].metric("Gross notional", f"${risk['gross']:,.0f}",
                   f"VaR/gross {risk['var']/risk['gross']*100:.2f}%" if risk["gross"] else "")

    st.subheader("Risk Decomposition")
    st.caption(
        "**Component VaR** is the honest number: it accounts for how each leg interacts with the "
        "rest of the book, and the components sum to the portfolio VaR. A hedged leg can even "
        "show a *negative* contribution — it is removing risk. Standalone VaR ignores all of that."
    )
    rdf = pd.DataFrame(risk["rows"])
    st.dataframe(
        rdf.style.format({"Mark": "{:,.4f}", "Notional": "${:+,.0f}", "Vol": "{:.1f}%",
                          "StandaloneVaR": "${:,.0f}", "StandaloneES": "${:,.0f}",
                          "ComponentVaR": "${:+,.0f}", "PctOfVaR": "{:+.1f}%"}),
        use_container_width=True, hide_index=True)

    cA, cB = st.columns(2)
    with cA:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rdf["Contract"], y=rdf["StandaloneVaR"],
                             name="Standalone", marker_color=GRAY, opacity=0.6))
        fig.add_trace(go.Bar(x=rdf["Contract"], y=rdf["ComponentVaR"],
                             name="Component", marker_color=AMBER))
        fig.update_layout(title="Standalone vs component VaR",
                          yaxis_title="VaR ($)", barmode="group")
        st.plotly_chart(_styled(fig, 340), use_container_width=True)
    with cB:
        if risk["corr_used"] and len(rdf) > 1:
            names = rdf["Contract"].tolist()
            sub = corr.loc[names, names]
            fig2 = go.Figure(go.Heatmap(
                z=sub.values, x=names, y=names,
                colorscale=[[0, RED], [0.5, PANEL], [1, GREEN]], zmid=0, zmin=-1, zmax=1,
                text=np.round(sub.values, 2), texttemplate="%{text}",
                textfont=dict(size=10), colorbar=dict(title="ρ")))
            fig2.update_layout(title="Book correlation", height=340,
                               paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT),
                               margin=dict(l=10, r=10, t=30, b=10))
            fig2.update_xaxes(tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Historical VaR — no distributional assumption ────────────────────────
    st.subheader("Historical Simulation VaR")
    st.caption(
        "Replays the last 500 actual daily return vectors onto today's book. **No normality "
        "assumption, no correlation matrix** — the real joint behaviour, fat tails included, "
        "is already in the data. The gap against the parametric number is the information: "
        "where parametric is lower, it is understating the tail."
    )
    with st.spinner("Replaying history…"):
        hv = historical_var(positions, marks, conf=conf, horizon=horizon)

    if not hv.get("available"):
        st.info("Insufficient overlapping history for a historical simulation.")
    else:
        ratio_var = hv["var"] / risk["var"] if risk["var"] else 0
        ratio_es  = hv["es"] / risk["es"] if risk["es"] else 0

        h = st.columns(4)
        h[0].markdown(kpi(f"Historical VaR {int(conf*100)}", f"${hv['var']:,.0f}",
                          f"{hv['n_days']} days replayed", AMBER), unsafe_allow_html=True)
        h[1].markdown(kpi(f"Historical ES {int(conf*100)}", f"${hv['es']:,.0f}",
                          "mean of the tail", PURPLE), unsafe_allow_html=True)
        h[2].markdown(kpi("Hist / Param VaR", f"{ratio_var:.2f}×",
                          "parametric understates" if ratio_var > 1.05 else "broadly agree",
                          RED if ratio_var > 1.05 else GREEN), unsafe_allow_html=True)
        h[3].markdown(kpi("Worst day in sample", f"${hv['worst_pnl']:+,.0f}",
                          str(hv["worst_date"]), RED), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if ratio_var > 1.05:
            st.warning(
                f"**The parametric model is understating this book by {(ratio_var-1)*100:.0f}%.** "
                f"Its normality assumption cannot see the tail that actually happened. Trust the "
                f"historical number, and the ES over the VaR.", icon="⚠️")

        fig_h = go.Figure(go.Histogram(x=hv["pnl"], nbinsx=60, marker_color=BLUE, opacity=0.75,
                                       name="Daily P&L"))
        fig_h.add_vline(x=-hv["var"], line=dict(color=AMBER, width=2),
                        annotation_text=f"Hist VaR {int(conf*100)}")
        fig_h.add_vline(x=-risk["var"], line=dict(color=GREEN, width=2, dash="dot"),
                        annotation_text="Param VaR")
        fig_h.add_vline(x=-hv["es"], line=dict(color=RED, width=2, dash="dash"),
                        annotation_text="Hist ES")
        fig_h.update_layout(title="Distribution of replayed daily book P&L", xaxis_title="$")
        st.plotly_chart(_styled(fig_h, 340), use_container_width=True)

    # ── Dated stress episodes ────────────────────────────────────────────────
    st.subheader("Historical Stress Episodes")
    st.caption(
        "A parallel ±30% shock is a weak test — real dislocations are not parallel. These replay "
        "the **actual, per-contract** moves of dated windows onto the current book. Note the "
        "2022 LME nickel squeeze is absent: nickel is not on this desk and never will be on a "
        "free feed."
    )
    ep = st.selectbox("Episode", list(STRESS_EPISODES.keys()), key="stress_ep")
    s_start, s_end = STRESS_EPISODES[ep]

    with st.spinner(f"Replaying {ep}…"):
        sr = stress_replay(positions, marks, s_start, s_end)

    if not sr.get("available"):
        st.info(
            f"No overlapping history for **{ep}**. Contracts added recently, or a feed gap, "
            f"can leave a window empty. Nothing is extrapolated to fill it."
        )
    else:
        srdf = pd.DataFrame(sr["rows"])
        st.markdown(
            kpi(f"Book P&L — {ep}", f"${sr['total']:+,.0f}",
                f"{sr['start']} → {sr['end']}",
                GREEN if sr["total"] >= 0 else RED),
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        fig_s = go.Figure(go.Bar(
            x=srdf["Contract"], y=srdf["PnL"],
            marker_color=np.where(srdf["PnL"] >= 0, GREEN, RED),
            text=[f"{m:+.1f}%" for m in srdf["Move"]], textposition="outside"))
        fig_s.update_layout(title="P&L by leg (label = actual move in the episode)",
                            yaxis_title="$")
        st.plotly_chart(_styled(fig_s, 320), use_container_width=True)

        st.dataframe(
            srdf.style.format({"Move": "{:+.2f}%", "PnL": "${:+,.0f}"})
                .map(lambda v: (f"color:{GREEN}" if v >= 0 else f"color:{RED}")
                     if isinstance(v, (int, float)) else "", subset=["PnL"]),
            use_container_width=True, hide_index=True)

    # ── Parallel shock (kept, but framed honestly) ───────────────────────────
    st.subheader("Book-Wide Parallel Shock")
    st.caption("Every mark moved by the same percentage. Kept for reference, but a parallel "
               "move is itself a strong assumption — the dated episodes above are the better test.")
    shocks = [-30, -20, -10, -5, 0, 5, 10, 20, 30]
    pnls = []
    for sh in shocks:
        t = 0.0
        for p in positions:
            if p.get("kind") == "option":
                continue
            mark = marks.get(p["commodity"])
            if mark is None:
                continue
            sign = 1 if p["side"] == "Long" else -1
            t += sign * (mark*(1 + sh/100) - p["entry"]) * price_multiplier(p["commodity"]) * p["lots"]
        pnls.append(t)

    fig3 = go.Figure(go.Bar(x=[f"{s:+d}%" for s in shocks], y=pnls,
                            marker_color=np.where(np.array(pnls) >= 0, GREEN, RED),
                            text=[f"${v:+,.0f}" for v in pnls], textposition="outside"))
    fig3.update_layout(yaxis_title="Book P&L ($)")
    st.plotly_chart(_styled(fig3, 320), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MONTE CARLO  (OU / mean-reverting)
# ══════════════════════════════════════════════════════════════════════════════
def page_mc(commodity, marks):
    st.title(f"Monte Carlo — {commodity}")
    c = COMMODITIES[commodity]
    mark = require_mark(commodity, marks)
    if mark is None:
        return

    default_hl = c.get("mr_halflife")
    st.caption(
        "Monthly steps from the live front month. **GBM is the wrong model for most of these "
        "markets**: at nat-gas vol over three years it puts the P95 near 3× spot and the P5 near "
        "zero, which no storable commodity does — inventory arbitrage pulls price back. The "
        "Schwartz one-factor model adds log-price mean reversion and fixes the tails. Gold and "
        "silver are the exception: they behave like financial assets and default to GBM."
    )

    col1, col2, col3, col4 = st.columns(4)
    n_paths = col1.slider("Paths", 200, 5000, 1500, 100, key="mc_n")
    rv = realised_vol(c["yf_ticker"], 60)
    vol_p = col2.slider("σ (%)", 5, 150, int((rv or c["vol"])*100), key="mc_vol")
    horizon = col3.slider("Horizon (months)", 3, 36, 18, 3, key="mc_h")
    model = col4.selectbox(
        "Model",
        ["Schwartz 1-factor (mean-reverting)", "GBM (no reversion)"],
        index=0 if default_hl else 1, key="mc_model")

    if model.startswith("Schwartz"):
        hl = st.slider("Mean-reversion half-life (years)", 0.25, 5.0,
                       float(default_hl or 1.5), 0.25, key="mc_hl",
                       help="Time for a shock to decay by half. Gas ~0.75y (fast, storage-driven). "
                            "Crude ~2y. Metals barely revert at all.")
    else:
        hl = None
        if default_hl:
            st.warning(
                f"**{commodity} has a registry half-life of {default_hl:.2f}y** — it is a storable "
                f"whose price reverts. Running it as GBM will produce tails that are too fat. "
                f"Shown for comparison only.", icon="⚠️")

    with st.spinner(f"Simulating {n_paths:,} paths…"):
        res = simulate(mark, vol_p/100, n_paths, horizon, halflife=hl)

    st.markdown(f'<span class="badge badge-amber">{res["model"]}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    mult = price_multiplier(commodity)
    cols = st.columns(5)
    cols[0].metric("Spot (t=0)", f"{mark:,.4f}", c["unit"])
    cols[1].metric("Median", f"{res['median']:,.4f}", f"{(res['median']/mark-1)*100:+.1f}%")
    cols[2].metric("P5", f"{res['p5']:,.4f}", f"{(res['p5']/mark-1)*100:+.1f}%")
    cols[3].metric("P95", f"{res['p95']:,.4f}", f"{(res['p95']/mark-1)*100:+.1f}%")
    cols[4].metric("P5 loss/lot", f"${(res['p5']-mark)*mult:,.0f}", "if long 1 lot")

    fan = res["fan"]
    fig = go.Figure()
    for y, nm, col_, dash in [("p95", "P95", GREEN, "dot"), ("p75", "P75", GREEN, "solid"),
                              ("p50", "Median", AMBER, "solid"), ("p25", "P25", RED, "solid"),
                              ("p5", "P5", RED, "dot")]:
        fig.add_trace(go.Scatter(
            x=fan["date"], y=fan[y], name=nm,
            line=dict(color=col_, width=2.5 if y == "p50" else 1, dash=dash),
            fill="tonexty" if y != "p95" else None,
            fillcolor="rgba(240,165,0,0.05)"))
    fig.update_layout(title="Price fan", yaxis_title=c["unit"])
    st.plotly_chart(_styled(fig, 420), use_container_width=True)

    # Side-by-side comparison — makes the point better than any caption.
    if default_hl:
        st.subheader("Why the model matters")
        st.caption("Same spot, same vol, same horizon. The only difference is mean reversion.")
        cmp_gbm = simulate(mark, vol_p/100, n_paths, horizon, halflife=None)
        cmp_ou  = simulate(mark, vol_p/100, n_paths, horizon, halflife=default_hl)
        cdf = pd.DataFrame([
            dict(Model="GBM (no reversion)", P5=cmp_gbm["p5"], Median=cmp_gbm["median"],
                 P95=cmp_gbm["p95"], **{"P95/Spot": cmp_gbm["p95"]/mark}),
            dict(Model=f"Schwartz (hl={default_hl:.2f}y)", P5=cmp_ou["p5"], Median=cmp_ou["median"],
                 P95=cmp_ou["p95"], **{"P95/Spot": cmp_ou["p95"]/mark}),
        ])
        st.dataframe(
            cdf.style.format({"P5": "{:,.4f}", "Median": "{:,.4f}", "P95": "{:,.4f}",
                              "P95/Spot": "{:.2f}×"}),
            use_container_width=True, hide_index=True)

    fig_h = go.Figure(go.Bar(x=res["hist_x"], y=res["hist_y"],
                             marker_color=AMBER, opacity=0.75))
    fig_h.add_vline(x=mark, line=dict(color=TEXT, dash="dash"), annotation_text="Spot")
    fig_h.add_vline(x=res["p5"], line=dict(color=RED, dash="dot"), annotation_text="P5")
    fig_h.add_vline(x=res["p95"], line=dict(color=GREEN, dash="dot"), annotation_text="P95")
    fig_h.update_layout(title=f"Terminal distribution at {horizon}m", xaxis_title=c["unit"])
    st.plotly_chart(_styled(fig_h, 280), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MACRO
# ══════════════════════════════════════════════════════════════════════════════
def page_signals(marks):
    st.title("Signals")
    st.caption(
        "Every contract, every signal, one screen. This desk had sixteen pages and no single "
        "place that answered *where should I be looking today*. Nothing here is new maths — it "
        "is the curve, vol, momentum and seasonal pages, aggregated. All of it is live-derived."
    )

    with st.spinner("Scanning the board… (first run pulls every strip — ~30s)"):
        sig = build_signals()

    if sig.empty:
        st.error("Scanner returned nothing — the feed is down.")
        return

    sectors = st.multiselect("Sectors", ALL_SECTORS, default=ALL_SECTORS, key="sig_sectors")
    view = sig[sig["Sector"].isin(sectors)].copy() if sectors else sig.copy()
    if view.empty:
        return

    # ── Headline reads ───────────────────────────────────────────────────────
    backw   = view[view["Structure"] == "BACKWARD"]
    conta   = view[view["Structure"] == "CONTANGO"]
    vol_up  = view[view["VolRegime"] > 1.2]
    stretch = view[(view["Px%ile1y"] > 90) | (view["Px%ile1y"] < 10)]

    c = st.columns(4)
    c[0].markdown(kpi("Backwardated", str(len(backw)),
                      "tight prompt — physical squeeze", GREEN), unsafe_allow_html=True)
    c[1].markdown(kpi("Contango", str(len(conta)),
                      "carry market — storage pays", RED), unsafe_allow_html=True)
    c[2].markdown(kpi("Vol expanding", str(len(vol_up)),
                      "RV60 > 1.2× RV252", PURPLE), unsafe_allow_html=True)
    c[3].markdown(kpi("Price stretched", str(len(stretch)),
                      ">90th or <10th %ile of 1y", AMBER), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── The board ────────────────────────────────────────────────────────────
    st.subheader("The Board")
    disp = view[["Contract", "Sector", "Mark", "Structure", "Carry%", "M1M2",
                 "RV60", "VolRegime", "Chg1M", "Chg3M", "Px%ile1y",
                 "SeasonMed", "SeasonHit"]].copy()
    disp.columns = ["Contract", "Sector", "Mark", "Structure", "Carry %", "M1–M2",
                    "RV60 %", "Vol regime", "1M %", "3M %", "Px %ile 1y",
                    "Season med %", "Season hit %"]

    def _struct(v):
        if v == "BACKWARD":
            return f"color:{GREEN};font-weight:600"
        if v == "CONTANGO":
            return f"color:{RED};font-weight:600"
        return f"color:{GRAY}"

    def _signed(v):
        if isinstance(v, (int, float)) and pd.notna(v):
            return f"color:{GREEN}" if v >= 0 else f"color:{RED}"
        return ""

    def _regime(v):
        if isinstance(v, (int, float)) and pd.notna(v):
            if v > 1.2:
                return f"color:{PURPLE};font-weight:600"
            if v < 0.8:
                return f"color:{GRAY}"
        return ""

    def _pctile(v):
        if isinstance(v, (int, float)) and pd.notna(v):
            if v > 90:
                return f"color:{RED};font-weight:600"
            if v < 10:
                return f"color:{GREEN};font-weight:600"
        return ""

    st.dataframe(
        disp.style
            .format({"Mark": "{:,.4f}", "Carry %": "{:+.2f}", "M1–M2": "{:+,.4f}",
                     "RV60 %": "{:.1f}", "Vol regime": "{:.2f}×",
                     "1M %": "{:+.1f}", "3M %": "{:+.1f}", "Px %ile 1y": "{:.0f}",
                     "Season med %": "{:+.1f}", "Season hit %": "{:.0f}"}, na_rep="—")
            .map(_struct, subset=["Structure"])
            .map(_signed, subset=["Carry %", "M1–M2", "1M %", "3M %", "Season med %"])
            .map(_regime, subset=["Vol regime"])
            .map(_pctile, subset=["Px %ile 1y"]),
        use_container_width=True, hide_index=True, height=560)

    st.caption(
        "**Vol regime** = RV60 ÷ RV252. Above 1.2× (purple) vol is expanding; below 0.8× it is "
        "compressing. **Px %ile** red above 90 / green below 10 flags a stretched level. "
        "**Season** columns are the 10-year median return and hit-rate for the *current* "
        "calendar month, and are blank for contracts with no real seasonality."
    )

    # ── Cross-sectional views ────────────────────────────────────────────────
    st.subheader("Cross-Section")
    t1, t2, t3 = st.tabs(["Carry", "Momentum vs Vol", "Seasonal (this month)"])

    with t1:
        cv = view.dropna(subset=["Carry%"]).sort_values("Carry%")
        fig = go.Figure(go.Bar(
            x=cv["Contract"], y=cv["Carry%"],
            marker_color=np.where(cv["Carry%"] >= 0, RED, GREEN),
            text=[f"{v:+.1f}%" for v in cv["Carry%"]], textposition="outside"))
        fig.add_hline(y=0, line=dict(color=GRAY, dash="dash"))
        fig.update_layout(title="Curve carry, M1 → back (green = backwardation, pays the long)",
                          yaxis_title="%")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(_styled(fig, 400), use_container_width=True)

    with t2:
        mv = view.dropna(subset=["Chg3M", "RV60"])
        fig2 = go.Figure()
        for s in mv["Sector"].unique():
            d = mv[mv["Sector"] == s]
            fig2.add_trace(go.Scatter(
                x=d["RV60"], y=d["Chg3M"], mode="markers+text", name=s,
                text=[COMMODITIES[n]["ticker"] for n in d["Contract"]],
                textposition="top center", textfont=dict(size=9),
                marker=dict(size=12, line=dict(color=BORDER, width=1))))
        fig2.add_hline(y=0, line=dict(color=GRAY, dash="dash"))
        fig2.update_layout(title="3-month return vs realised vol",
                           xaxis_title="RV60 (%)", yaxis_title="3M return (%)")
        st.plotly_chart(_styled(fig2, 400), use_container_width=True)
        st.caption("Top-right = trending and volatile. Bottom-right = falling hard. "
                   "Left = quiet, and quiet markets are where carry trades live.")

    with t3:
        sv = view.dropna(subset=["SeasonMed"]).sort_values("SeasonMed")
        if sv.empty:
            st.info("No seasonal contracts in the current sector filter.")
        else:
            fig3 = go.Figure(go.Bar(
                x=sv["Contract"], y=sv["SeasonMed"],
                marker_color=np.where(sv["SeasonMed"] >= 0, GREEN, RED),
                text=[f"{h:.0f}% hit" for h in sv["SeasonHit"]], textposition="outside"))
            fig3.add_hline(y=0, line=dict(color=GRAY, dash="dash"))
            fig3.update_layout(
                title=f"10y median return for {MONTH_NAMES[date.today().month-1]}",
                yaxis_title="%")
            fig3.update_xaxes(tickangle=-45)
            st.plotly_chart(_styled(fig3, 380), use_container_width=True)
            st.caption("A strong median with a 55% hit rate is noise. Look for both together.")


def page_macro():
    st.title("Macro")

    key = st.session_state.get("fred_key", "")
    if not key:
        st.warning(
            "**No FRED API key set.** Enter one in the sidebar to pull real CPI, policy rates "
            "and GDP. The key is free at "
            "[fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html). "
            "Without it this page stays empty — **this desk no longer ships model-generated "
            "macro series.** They were the last fabricated data here, and they are gone.",
            icon="🔑")
        return

    if not REQUESTS_AVAILABLE:
        st.error("`requests` is not installed. Run `pip install requests`.")
        return

    st.caption("Live FRED series. Same rule as the price feed: real data, or an empty screen.")

    tab_c, tab_x = st.tabs(["By country", "Commodity context"])

    with tab_c:
        col1, col2 = st.columns([2, 5])
        primary = col1.selectbox("Country", list(FRED_SERIES.keys()), key="macro_primary")
        compare = col2.multiselect(
            "Compare", [c for c in FRED_SERIES if c != primary],
            default=[c for c in FRED_SERIES if c != primary][:2], key="macro_cmp")

        countries = [primary] + compare
        for metric, meta in MACRO_METRICS.items():
            st.subheader(meta["label"])
            st.caption(meta["note"])
            fig = go.Figure()
            any_data = False
            for i, ctry in enumerate(countries):
                sid = FRED_SERIES.get(ctry, {}).get(metric)
                if not sid:
                    continue
                df = fetch_fred(sid, key)
                if df.empty:
                    continue
                any_data = True
                y = df["value"]
                # CPI comes as an index; a commodity desk wants the YoY rate.
                if metric == "cpi_yoy":
                    y = df["value"].pct_change(12) * 100
                fig.add_trace(go.Scatter(
                    x=df.index, y=y, name=ctry,
                    line=dict(color=[AMBER, BLUE, GREEN, RED, PURPLE, TEAL][i % 6], width=2)))
            if not any_data:
                st.info(f"FRED has no **{meta['label']}** series mapped for the selected "
                        f"countries. Nothing is substituted.")
                continue
            ylab = "YoY %" if metric == "cpi_yoy" else meta["label"]
            fig.update_layout(yaxis_title=ylab)
            st.plotly_chart(_styled(fig, 340), use_container_width=True)

    with tab_x:
        st.caption(
            "The macro series a commodity desk actually watches. The dollar and real yields "
            "drive the whole complex — a stronger dollar is a headwind for every dollar-priced "
            "commodity, and breakevens are the market's own inflation view."
        )
        for label, sid in FRED_COMMODITY_CONTEXT.items():
            df = fetch_fred(sid, key, start="2018-01-01")
            if df.empty:
                st.error(f"**{label}** — FRED returned nothing for `{sid}`.")
                continue
            last = float(df["value"].iloc[-1])
            chg = last - float(df["value"].iloc[-22]) if len(df) > 22 else 0.0

            cc = st.columns([1, 4])
            cc[0].metric(label, f"{last:,.2f}", f"{chg:+.2f} (1m)")
            with cc[1]:
                fig = go.Figure(go.Scatter(x=df.index, y=df["value"],
                                           line=dict(color=AMBER, width=2)))
                st.plotly_chart(_styled(fig, 200), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CALENDAR
# ══════════════════════════════════════════════════════════════════════════════
def page_events():
    st.title("Calendar")
    st.caption("Scheduled prints that move these markets. Know when you're carrying risk into one.")
    today = date.today()
    df = pd.DataFrame([
        dict(Date=str(e["date"]), In=f"{(e['date']-today).days}d",
             Event=e["event"], Tags=", ".join(e["tags"]))
        for e in sorted(EVENTS, key=lambda x: x["date"])])
    st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
def page_about():
    st.title("About")
    status = (f"yfinance {yf.__version__} — feed active" if YF_AVAILABLE
              else "yfinance not installed")
    counts = {s: sum(1 for c in COMMODITIES.values() if c["sector"] == s) for s in ALL_SECTORS}
    cov = " · ".join(f"{s} {n}" for s, n in counts.items())

    st.markdown(f"""
**Commodity Trading Desk — live mark-to-market build.**

All **{len(COMMODITIES)} contracts** carry both a live front-month settle and a live dated
forward strip. No hardcoded marks, no cost-of-carry curve standing in for a market, no
contract proxied off another exchange. If the feed fails, the screen says so.

**Coverage** — {cov}

**Data**
- Front month: Yahoo continuous contract, 5-min cache
- Forward strip: dated exchange codes (`CLZ26.NYM`, `GCG27.CMX`…), 1-hour cache
- Correlation: 252-day window on daily log returns, 2-year sample
- Realised vol: 60-day close-to-close, annualised
- EIA: US crude, product and gas inventories (free key required)

**Analytics**
- Black-76, full Greeks, per-unit and per-lot cash
- Calendar spreads and roll yield read directly off the strip, with 2y percentiles
- Crack (3-2-1, RB-CL, HO-CL), board crush, WTI-Brent arb, gold/silver ratio
- **Correlation-aware VaR** — σₚ = √(wᵀΣw), with component VaR decomposition
- **Schwartz one-factor Monte Carlo** — log-price mean reversion, per-contract half-life
- Seasonality — 10y monthly distributions on the contracts that actually have a season

**Deliberately excluded**

Nine contracts were removed because they could not be marked honestly:

| Contract | Reason |
|---|---|
| LME Copper | Was proxied off COMEX `HG=F` ($/lb) but labelled $/mt — ~2200× error |
| ICE Gasoil | Live front, no dated strip — the curve was a model |
| EUA, Coal API2 | No feed — marks were constants |
| LME Aluminium, Nickel | No feed — marks were constants |
| Capesize, Panamax | No feed, and cost-of-carry on a non-storable TC route is meaningless |

Each needs a paid feed (LME, ICE, Baltic Exchange) to return.

**Known limits — read these**
- **No listed option chain.** Yahoo serves chains for equities and ETFs, not futures.
  Implied vol here is an *input*, seeded from realised. The surface is a shape, not a market.
  A real implied surface needs CME DataMine, Refinitiv or Bloomberg.
- **Correlation is historical.** It rises toward 1 in a crisis, exactly when the
  diversification benefit is being relied on. Watch the undiversified number too.
- **Parametric VaR assumes normal returns.** Commodities have fat tails; ES is the
  better guide, and even that understates a genuine dislocation.
- **Regional balances** are static estimates covering 3 contracts.
- **Supply/demand and macro pages** are scenario sandboxes, not balances.

**Status:** {status}

---
**[CFCAP](https://aeg-cfcap.streamlit.app)** — curve analytics: PCA, Schwartz-Smith, 51 signals
**[CODAP](https://aeg-codap.streamlit.app)** — derivatives pricing: Asian, crack, calendar, barrier
**[Portfolio Optimizer](https://aeg-markowitz.streamlit.app)** — Markowitz allocation

**Adam EL GBOURI** · [github.com/adamelgbouri](https://github.com/adamelgbouri)
""")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not YF_AVAILABLE:
        st.error("**yfinance is not installed.** This desk has no fallback prices by design. "
                 "Run `pip install yfinance` and reload.")
        st.stop()

    with st.spinner("Pulling live marks…"):
        marks = fetch_live_marks()

    n_live = sum(1 for v in marks.values() if v is not None)
    if n_live == 0:
        st.error("**Feed down.** No contract could be marked. Nothing is displayed rather than "
                 "showing stale or modelled prices.")
        st.stop()
    if n_live < len(COMMODITIES):
        missing = [k for k, v in marks.items() if v is None]
        st.warning(f"**{len(missing)} contract(s) unmarked:** {', '.join(missing)}. "
                   f"Their pages will error rather than substitute a price.", icon="⚠️")

    page, commodity = render_sidebar(marks)
    render_header(commodity, marks.get(commodity))

    dispatch = {
        "📡 Signals":           lambda: page_signals(marks),
        "📊 Dashboard":         lambda: page_dashboard(commodity, marks),
        "📈 Forward Curve":     lambda: page_curve(commodity, marks),
        "🔀 Spreads & Roll":    lambda: page_spreads(commodity, marks),
        "⚗️ Crack & Crush":     lambda: page_structures(marks),
        "🔗 Correlation":       page_correlation,
        "🛢️ EIA Fundamentals":  lambda: page_eia(commodity, marks),
        "🗓️ Seasonality":       lambda: page_seasonality(commodity, marks),
        "🌍 Regional Balances": lambda: page_regional(commodity, marks),
        "📅 Calendar":          page_events,
        "🎯 Options & Greeks":  lambda: page_options(commodity, marks),
        "📉 Vol Surface":       lambda: page_vol_surface(commodity, marks),
        "💼 Blotter":           lambda: page_blotter(marks),
        "🛡️ Risk":              lambda: page_risk(marks),
        "🎲 Monte Carlo":       lambda: page_mc(commodity, marks),
        "🌐 Macro":             page_macro,
        "ℹ️ About":             page_about,
    }
    dispatch.get(page, lambda: st.error("Page not found"))()


if __name__ == "__main__":
    main()
