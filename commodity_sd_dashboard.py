"""
Commodity Supply & Demand Analytics Desk - single-file Streamlit app.

Run with:
    pip install streamlit pandas numpy plotly scikit-learn statsmodels openpyxl yfinance
    streamlit run commodity_sd_dashboard.py

Self-contained:
- synthetic data generators for Crude Oil, Natural Gas, Copper, Wheat
- balance engine, seasonality, inventory, elasticity, lagged response
- scenario engine (Bull/Base/Bear), regional flows, futures curve
- macro overlay, Monte Carlo, sensitivities, fair value, positioning
- dark trading-desk Plotly theme + sidebar navigation
"""

from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression


# =============================================================================
# CONFIG
# =============================================================================

DARK_BG = "#0e1117"
PANEL_BG = "#161b22"
ACCENT = "#00d4ff"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
GREY = "#9ca3af"

COLORS = {
    "supply": GREEN,
    "demand": RED,
    "stocks": ACCENT,
    "price": AMBER,
    "fair_value": "#a78bfa",
    "bull": GREEN,
    "base": ACCENT,
    "bear": RED,
    "neutral": GREY,
}


@dataclass(frozen=True)
class CommodityTemplate:
    """Static template describing the structural characteristics of a commodity.

    Includes *benchmark / ideal metrics* that the UI compares live values against:
        days_cover_target   - normal inventory cover in days
        ideal_utilization_pct - typical "healthy" storage utilisation
        typical_monthly_vol_pct - 1σ monthly price volatility band
        normal_yoy_demand_pct - structural demand growth in a healthy market
        ideal_mm_pct_of_oi  - normal managed-money positioning as % of OI
    """

    key: str
    name: str
    unit: str
    inventory_unit: str
    ticker: str
    base_supply: float
    base_demand: float
    base_price: float
    price_band: tuple
    storage_capacity: float
    days_cover_target: float
    price_unit: str = "$"        # quote unit: $/bbl, $/oz, ¢/lb, $/t, ¢/bu, $/MMBtu...
    seasonal_demand: List[float] = field(default_factory=list)
    seasonal_supply: List[float] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    region_weights: List[float] = field(default_factory=list)
    elasticity_alpha: float = 0.0
    elasticity_beta: float = 0.0
    supply_lag_months: int = 6
    # benchmark / ideal metrics
    ideal_utilization_pct: float = 65.0
    typical_monthly_vol_pct: float = 6.0
    normal_yoy_demand_pct: float = 1.5
    ideal_mm_pct_of_oi: float = 15.0
    sector: str = "Energy"          # Energy / Metals / Ags / Softs / Precious
    # Live futures-curve metadata (inspired by cfcap):
    yf_fmt: str = ""                # Yahoo contract pattern e.g. "CL{M}{YY}.NYM"
    active_months: str = "FGHJKMNQUVXZ"   # delivery months actually listed
    liquid_months: int = 12         # how many forward contracts are liquid
    storage_cost_per_yr: float = 0.06     # for synthetic carry curve fallback


# Seasonality patterns - multiplicative factors around 1.0 across Jan..Dec.
_FLAT = [1.00] * 12

# Energy
_OIL_DEMAND_SEAS = [1.02, 1.00, 0.99, 0.98, 0.99, 1.01, 1.03, 1.04, 1.02, 1.00, 0.99, 1.03]
_OIL_SUPPLY_SEAS = _FLAT
_GAS_DEMAND_SEAS = [1.35, 1.25, 1.10, 0.90, 0.80, 0.85, 0.95, 0.95, 0.85, 0.90, 1.10, 1.30]
_GAS_SUPPLY_SEAS = _FLAT
_GASOLINE_DEMAND = [0.92, 0.90, 0.94, 0.98, 1.04, 1.10, 1.13, 1.12, 1.06, 1.00, 0.95, 0.92]
_GASOLINE_SUPPLY = [0.95, 0.93, 0.96, 1.00, 1.04, 1.06, 1.06, 1.06, 1.04, 1.00, 0.96, 0.94]

# Industrial metals
_COPPER_DEMAND_SEAS = [0.95, 0.97, 1.02, 1.04, 1.05, 1.03, 1.00, 0.99, 1.02, 1.04, 1.00, 0.89]
_COPPER_SUPPLY_SEAS = [0.97, 0.96, 1.00, 1.02, 1.03, 1.03, 1.02, 1.02, 1.01, 1.01, 0.98, 0.95]
_ALUMINUM_DEMAND = [1.00, 0.98, 1.02, 1.04, 1.05, 1.03, 1.00, 0.99, 1.01, 1.02, 1.00, 0.92]
_ALUMINUM_SUPPLY = _FLAT
_NICKEL_DEMAND = [0.95, 0.96, 1.00, 1.03, 1.05, 1.04, 1.02, 1.01, 1.02, 1.03, 1.00, 0.89]
_NICKEL_SUPPLY = [0.98, 0.97, 1.00, 1.01, 1.02, 1.02, 1.01, 1.01, 1.00, 1.00, 0.99, 0.99]
_IRON_ORE_DEMAND = [1.08, 0.92, 1.05, 1.06, 1.04, 1.02, 1.00, 0.99, 1.01, 1.03, 1.00, 0.95]
_IRON_ORE_SUPPLY = [0.92, 0.96, 1.02, 1.05, 1.06, 1.04, 1.02, 1.02, 1.00, 1.00, 0.96, 0.95]

# Precious metals
_GOLD_DEMAND = [1.15, 1.08, 0.95, 0.92, 0.94, 0.92, 0.95, 0.98, 1.00, 1.10, 1.05, 1.10]
_GOLD_SUPPLY = _FLAT
_SILVER_DEMAND = [1.05, 1.00, 0.98, 0.96, 0.97, 0.98, 1.00, 1.02, 1.03, 1.05, 1.02, 1.00]
_SILVER_SUPPLY = _FLAT

# Grains / oilseeds (northern hemisphere harvest concentrated)
_WHEAT_DEMAND_SEAS = _FLAT
_WHEAT_SUPPLY_SEAS = [0.60, 0.50, 0.60, 0.80, 1.10, 1.60, 1.90, 1.70, 1.30, 0.90, 0.70, 0.60]
_CORN_DEMAND = _FLAT
_CORN_SUPPLY = [0.50, 0.40, 0.50, 0.70, 1.10, 1.50, 1.40, 1.10, 1.50, 1.80, 1.40, 0.70]
_SOY_DEMAND = _FLAT
_SOY_SUPPLY = [0.60, 1.20, 1.50, 1.40, 0.70, 0.60, 0.60, 0.60, 1.40, 1.80, 1.20, 0.80]

# Softs
_COFFEE_DEMAND = [1.05, 1.03, 1.00, 0.98, 0.95, 0.95, 0.97, 0.99, 1.02, 1.04, 1.04, 1.05]
_COFFEE_SUPPLY = [0.50, 0.55, 0.65, 0.85, 1.20, 1.50, 1.55, 1.45, 1.30, 1.00, 0.75, 0.60]
_SUGAR_DEMAND = [1.05, 1.05, 1.10, 1.05, 1.00, 0.95, 0.95, 0.98, 1.00, 1.00, 1.00, 1.05]
_SUGAR_SUPPLY = [0.70, 0.65, 0.80, 0.95, 1.10, 1.30, 1.40, 1.25, 1.05, 1.00, 0.95, 0.85]


COMMODITY_TEMPLATES: Dict[str, CommodityTemplate] = {
    # ---------- ENERGY ----------
    "crude_oil": CommodityTemplate(
        key="crude_oil", name="Crude Oil", unit="mb/d", inventory_unit="mb",
        ticker="CL=F", base_supply=101.0, base_demand=100.5, base_price=78.0,
        price_unit="$/bbl",
        price_band=(40.0, 130.0), storage_capacity=4200.0, days_cover_target=30.0,
        seasonal_demand=_OIL_DEMAND_SEAS, seasonal_supply=_OIL_SUPPLY_SEAS,
        regions=["US", "Europe", "China", "Middle East", "Rest of World"],
        region_weights=[0.21, 0.14, 0.15, 0.09, 0.41],
        elasticity_alpha=0.06, elasticity_beta=0.10, supply_lag_months=6,
        ideal_utilization_pct=70, typical_monthly_vol_pct=8,
        normal_yoy_demand_pct=1.2, ideal_mm_pct_of_oi=15, sector="Energy",
        yf_fmt="CL{M}{YY}.NYM", active_months="FGHJKMNQUVXZ", liquid_months=12, storage_cost_per_yr=0.072,
    ),
    "natural_gas": CommodityTemplate(
        key="natural_gas", name="Natural Gas", unit="bcf/d", inventory_unit="bcf",
        ticker="NG=F", base_supply=105.0, base_demand=104.0, base_price=3.20,
        price_unit="$/MMBtu",
        price_band=(1.50, 9.00), storage_capacity=4200.0, days_cover_target=35.0,
        seasonal_demand=_GAS_DEMAND_SEAS, seasonal_supply=_GAS_SUPPLY_SEAS,
        regions=["US", "Europe", "Asia LNG", "Rest of World"],
        region_weights=[0.30, 0.18, 0.20, 0.32],
        elasticity_alpha=0.18, elasticity_beta=0.08, supply_lag_months=4,
        ideal_utilization_pct=75, typical_monthly_vol_pct=12,
        normal_yoy_demand_pct=1.8, ideal_mm_pct_of_oi=18, sector="Energy",
        yf_fmt="NG{M}{YY}.NYM", active_months="FGHJKMNQUVXZ", liquid_months=12, storage_cost_per_yr=0.12,
    ),
    "gasoline": CommodityTemplate(
        key="gasoline", name="Gasoline (RBOB)", unit="mb/d", inventory_unit="mb",
        ticker="RB=F", base_supply=27.0, base_demand=26.5, base_price=2.40,
        price_unit="$/gal",
        price_band=(1.50, 4.50), storage_capacity=280.0, days_cover_target=23.0,
        seasonal_demand=_GASOLINE_DEMAND, seasonal_supply=_GASOLINE_SUPPLY,
        regions=["US", "Europe", "Asia", "Rest of World"],
        region_weights=[0.34, 0.22, 0.28, 0.16],
        elasticity_alpha=0.05, elasticity_beta=0.08, supply_lag_months=3,
        ideal_utilization_pct=80, typical_monthly_vol_pct=9,
        normal_yoy_demand_pct=0.5, ideal_mm_pct_of_oi=14, sector="Energy",
        yf_fmt="RB{M}{YY}.NYM", active_months="FGHJKMNQUVXZ", liquid_months=12, storage_cost_per_yr=0.084,
    ),

    # ---------- INDUSTRIAL METALS ----------
    "copper": CommodityTemplate(
        key="copper", name="Copper", unit="kt/mo", inventory_unit="kt",
        ticker="HG=F", base_supply=1900.0, base_demand=1910.0, base_price=4.20,
        price_unit="$/lb",
        price_band=(2.50, 6.00), storage_capacity=1400.0, days_cover_target=20.0,
        seasonal_demand=_COPPER_DEMAND_SEAS, seasonal_supply=_COPPER_SUPPLY_SEAS,
        regions=["China", "Europe", "US", "Rest of World"],
        region_weights=[0.55, 0.15, 0.10, 0.20],
        elasticity_alpha=0.04, elasticity_beta=0.07, supply_lag_months=12,
        ideal_utilization_pct=50, typical_monthly_vol_pct=6,
        normal_yoy_demand_pct=2.5, ideal_mm_pct_of_oi=20, sector="Metals",
        yf_fmt="HG{M}{YY}.CMX", active_months="HKNUZ", liquid_months=8, storage_cost_per_yr=0.048,
    ),
    "aluminum": CommodityTemplate(
        key="aluminum", name="Aluminum", unit="kt/mo", inventory_unit="kt",
        ticker="ALI=F", base_supply=5800.0, base_demand=5750.0, base_price=2300.0,
        price_unit="$/t",
        price_band=(1700.0, 3500.0), storage_capacity=4000.0, days_cover_target=25.0,
        seasonal_demand=_ALUMINUM_DEMAND, seasonal_supply=_ALUMINUM_SUPPLY,
        regions=["China", "Europe", "US", "Rest of World"],
        region_weights=[0.58, 0.14, 0.10, 0.18],
        elasticity_alpha=0.04, elasticity_beta=0.05, supply_lag_months=24,
        ideal_utilization_pct=50, typical_monthly_vol_pct=5,
        normal_yoy_demand_pct=3.0, ideal_mm_pct_of_oi=14, sector="Metals",
        yf_fmt="", active_months="FGHJKMNQUVXZ", liquid_months=12, storage_cost_per_yr=0.048,
    ),
    "nickel": CommodityTemplate(
        key="nickel", name="Nickel", unit="kt/mo", inventory_unit="kt",
        ticker="NI=F", base_supply=270.0, base_demand=265.0, base_price=18000.0,
        price_unit="$/t",
        price_band=(12000.0, 50000.0), storage_capacity=250.0, days_cover_target=20.0,
        seasonal_demand=_NICKEL_DEMAND, seasonal_supply=_NICKEL_SUPPLY,
        regions=["China", "Europe", "Indonesia", "Rest of World"],
        region_weights=[0.55, 0.12, 0.18, 0.15],
        elasticity_alpha=0.05, elasticity_beta=0.04, supply_lag_months=18,
        ideal_utilization_pct=50, typical_monthly_vol_pct=12,
        normal_yoy_demand_pct=5.0, ideal_mm_pct_of_oi=18, sector="Metals",
        yf_fmt="", active_months="FGHJKMNQUVXZ", liquid_months=12, storage_cost_per_yr=0.048,
    ),
    "iron_ore": CommodityTemplate(
        key="iron_ore", name="Iron Ore", unit="mt/mo", inventory_unit="mt",
        ticker="TIO=F", base_supply=130.0, base_demand=128.0, base_price=110.0,
        price_unit="$/t",
        price_band=(50.0, 230.0), storage_capacity=250.0, days_cover_target=25.0,
        seasonal_demand=_IRON_ORE_DEMAND, seasonal_supply=_IRON_ORE_SUPPLY,
        regions=["China", "Europe", "Japan/Korea", "Rest of World"],
        region_weights=[0.70, 0.10, 0.10, 0.10],
        elasticity_alpha=0.05, elasticity_beta=0.06, supply_lag_months=18,
        ideal_utilization_pct=65, typical_monthly_vol_pct=10,
        normal_yoy_demand_pct=1.5, ideal_mm_pct_of_oi=12, sector="Metals",
        yf_fmt="", active_months="FGHJKMNQUVXZ", liquid_months=12, storage_cost_per_yr=0.06,
    ),

    # ---------- PRECIOUS METALS ----------
    "gold": CommodityTemplate(
        key="gold", name="Gold", unit="t/mo", inventory_unit="t",
        ticker="GC=F", base_supply=305.0, base_demand=300.0, base_price=2000.0,
        price_unit="$/oz",
        price_band=(1200.0, 3000.0), storage_capacity=5000.0, days_cover_target=90.0,
        seasonal_demand=_GOLD_DEMAND, seasonal_supply=_GOLD_SUPPLY,
        regions=["China", "India", "OECD ETFs", "Rest of World"],
        region_weights=[0.25, 0.22, 0.28, 0.25],
        elasticity_alpha=0.03, elasticity_beta=0.02, supply_lag_months=24,
        ideal_utilization_pct=60, typical_monthly_vol_pct=4,
        normal_yoy_demand_pct=1.0, ideal_mm_pct_of_oi=22, sector="Precious",
        yf_fmt="GC{M}{YY}.CMX", active_months="GJMQVZ", liquid_months=8, storage_cost_per_yr=0.024,
    ),
    "silver": CommodityTemplate(
        key="silver", name="Silver", unit="t/mo", inventory_unit="t",
        ticker="SI=F", base_supply=2400.0, base_demand=2500.0, base_price=25.0,
        price_unit="$/oz",
        price_band=(15.0, 50.0), storage_capacity=30000.0, days_cover_target=90.0,
        seasonal_demand=_SILVER_DEMAND, seasonal_supply=_SILVER_SUPPLY,
        regions=["China", "India", "OECD", "Rest of World"],
        region_weights=[0.30, 0.20, 0.30, 0.20],
        elasticity_alpha=0.05, elasticity_beta=0.03, supply_lag_months=18,
        ideal_utilization_pct=60, typical_monthly_vol_pct=7,
        normal_yoy_demand_pct=2.0, ideal_mm_pct_of_oi=20, sector="Precious",
        yf_fmt="SI{M}{YY}.CMX", active_months="HKNUZ", liquid_months=6, storage_cost_per_yr=0.036,
    ),

    # ---------- AGRICULTURE / GRAINS ----------
    "wheat": CommodityTemplate(
        key="wheat", name="Wheat", unit="mt/mo", inventory_unit="mt",
        ticker="ZW=F", base_supply=65.0, base_demand=64.5, base_price=620.0,
        price_unit="¢/bu",
        price_band=(380.0, 1100.0), storage_capacity=320.0, days_cover_target=70.0,
        seasonal_demand=_WHEAT_DEMAND_SEAS, seasonal_supply=_WHEAT_SUPPLY_SEAS,
        regions=["US", "EU", "Black Sea", "China", "Rest of World"],
        region_weights=[0.10, 0.15, 0.12, 0.18, 0.45],
        elasticity_alpha=0.05, elasticity_beta=0.03, supply_lag_months=9,
        ideal_utilization_pct=60, typical_monthly_vol_pct=7,
        normal_yoy_demand_pct=1.0, ideal_mm_pct_of_oi=15, sector="Ags",
        yf_fmt="ZW{M}{YY}.CBT", active_months="HKNUZ", liquid_months=8, storage_cost_per_yr=0.06,
    ),
    "corn": CommodityTemplate(
        key="corn", name="Corn", unit="mt/mo", inventory_unit="mt",
        ticker="ZC=F", base_supply=100.0, base_demand=99.0, base_price=450.0,
        price_unit="¢/bu",
        price_band=(330.0, 800.0), storage_capacity=700.0, days_cover_target=80.0,
        seasonal_demand=_CORN_DEMAND, seasonal_supply=_CORN_SUPPLY,
        regions=["US", "China", "Brazil", "Rest of World"],
        region_weights=[0.32, 0.27, 0.10, 0.31],
        elasticity_alpha=0.05, elasticity_beta=0.04, supply_lag_months=9,
        ideal_utilization_pct=55, typical_monthly_vol_pct=6,
        normal_yoy_demand_pct=1.2, ideal_mm_pct_of_oi=16, sector="Ags",
        yf_fmt="ZC{M}{YY}.CBT", active_months="HKNUZ", liquid_months=8, storage_cost_per_yr=0.06,
    ),
    "soybeans": CommodityTemplate(
        key="soybeans", name="Soybeans", unit="mt/mo", inventory_unit="mt",
        ticker="ZS=F", base_supply=32.0, base_demand=31.5, base_price=1200.0,
        price_unit="¢/bu",
        price_band=(900.0, 1800.0), storage_capacity=220.0, days_cover_target=85.0,
        seasonal_demand=_SOY_DEMAND, seasonal_supply=_SOY_SUPPLY,
        regions=["US", "Brazil", "China", "Rest of World"],
        region_weights=[0.28, 0.34, 0.06, 0.32],
        elasticity_alpha=0.06, elasticity_beta=0.04, supply_lag_months=8,
        ideal_utilization_pct=55, typical_monthly_vol_pct=7,
        normal_yoy_demand_pct=2.5, ideal_mm_pct_of_oi=18, sector="Ags",
        yf_fmt="ZS{M}{YY}.CBT", active_months="FHKNQUX", liquid_months=8, storage_cost_per_yr=0.06,
    ),

    # ---------- SOFTS ----------
    "coffee": CommodityTemplate(
        key="coffee", name="Coffee (Arabica)", unit="kt/mo", inventory_unit="kt",
        ticker="KC=F", base_supply=880.0, base_demand=870.0, base_price=250.0,
        price_unit="¢/lb",
        price_band=(120.0, 500.0), storage_capacity=1800.0, days_cover_target=60.0,
        seasonal_demand=_COFFEE_DEMAND, seasonal_supply=_COFFEE_SUPPLY,
        regions=["Brazil", "Vietnam", "Europe (consumer)", "US (consumer)", "Rest of World"],
        region_weights=[0.35, 0.18, 0.20, 0.15, 0.12],
        elasticity_alpha=0.06, elasticity_beta=0.04, supply_lag_months=24,
        ideal_utilization_pct=60, typical_monthly_vol_pct=9,
        normal_yoy_demand_pct=2.0, ideal_mm_pct_of_oi=22, sector="Softs",
        yf_fmt="KC{M}{YY}.NYB", active_months="HKNUZ", liquid_months=6, storage_cost_per_yr=0.048,
    ),
    "sugar": CommodityTemplate(
        key="sugar", name="Sugar (Raw #11)", unit="mt/mo", inventory_unit="mt",
        ticker="SB=F", base_supply=15.0, base_demand=14.8, base_price=22.0,
        price_unit="¢/lb",
        price_band=(12.0, 40.0), storage_capacity=80.0, days_cover_target=90.0,
        seasonal_demand=_SUGAR_DEMAND, seasonal_supply=_SUGAR_SUPPLY,
        regions=["Brazil", "India", "Thailand", "EU", "Rest of World"],
        region_weights=[0.35, 0.20, 0.10, 0.10, 0.25],
        elasticity_alpha=0.05, elasticity_beta=0.04, supply_lag_months=12,
        ideal_utilization_pct=60, typical_monthly_vol_pct=8,
        normal_yoy_demand_pct=1.5, ideal_mm_pct_of_oi=18, sector="Softs",
        yf_fmt="SB{M}{YY}.NYB", active_months="HKNV", liquid_months=6, storage_cost_per_yr=0.048,
    ),
}


SCENARIO_PRESETS: Dict[str, Dict[str, float]] = {
    "Bull": {"supply_shock_pct": -2.0, "demand_shock_pct": 1.5, "gdp_growth_pct": 3.5,
             "weather_shock_pct": 1.0, "fx_usd_pct": -2.0, "probability": 0.25},
    "Base": {"supply_shock_pct": 0.0, "demand_shock_pct": 0.0, "gdp_growth_pct": 2.5,
             "weather_shock_pct": 0.0, "fx_usd_pct": 0.0, "probability": 0.50},
    "Bear": {"supply_shock_pct": 2.0, "demand_shock_pct": -2.0, "gdp_growth_pct": 0.5,
             "weather_shock_pct": -1.0, "fx_usd_pct": 3.0, "probability": 0.25},
}


# =============================================================================
# LOGGING
# =============================================================================

def get_logger(name: str = "commodity_sd") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(os.environ.get("COMMODITY_SD_LOG", "INFO").upper())
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)
    logger.propagate = False
    return logger


logger = get_logger()


# =============================================================================
# IO HELPERS
# =============================================================================

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def df_to_excel_bytes(frames: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet, df in frames.items():
            df.to_excel(writer, sheet_name=sheet[:31])
    return buf.getvalue()


def params_to_json(params: Dict) -> str:
    return json.dumps(params, indent=2, default=str)


def params_from_json(blob: str) -> Dict:
    try:
        return json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid parameter JSON: {exc}") from exc


# =============================================================================
# DATA - SYNTHETIC GENERATORS
# =============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def get_sd_dataset(commodity_key: str, start: str = "2018-01-01",
                   forecast_months: int = 24, seed: int = 42) -> pd.DataFrame:
    """Generate a monthly synthetic S&D table for a commodity."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    end = pd.Timestamp(date.today()).strftime("%Y-%m-01")
    idx = pd.date_range(start=start, end=end, freq="MS")
    horizon = pd.date_range(
        start=idx[-1] + pd.offsets.MonthBegin(1),
        periods=forecast_months, freq="MS",
    )
    full_idx = idx.append(horizon)

    rng = np.random.default_rng(seed)
    n = len(full_idx)
    months = full_idx.month - 1

    seas_d = np.array(tpl.seasonal_demand)[months]
    seas_s = np.array(tpl.seasonal_supply)[months]

    t = np.linspace(0, 1, n)
    demand_trend = tpl.base_demand * (1 + 0.010 * t * 6)
    supply_trend = tpl.base_supply * (1 + 0.012 * t * 6)

    cycle = 0.02 * np.sin(2 * np.pi * np.arange(n) / 48)
    demand = demand_trend * seas_d * (1 + cycle + rng.normal(0, 0.012, n))
    supply = supply_trend * seas_s * (1 + cycle * 0.5 + rng.normal(0, 0.015, n))

    imports = np.abs(rng.normal(tpl.base_demand * 0.20, tpl.base_demand * 0.02, n))
    exports = np.abs(rng.normal(tpl.base_demand * 0.18, tpl.base_demand * 0.02, n))

    days_in_month = full_idx.days_in_month.to_numpy()
    if tpl.unit.endswith("/d"):
        supply_monthly = supply * days_in_month
        demand_monthly = demand * days_in_month
    else:
        supply_monthly = supply
        demand_monthly = demand

    # Anchor the structural balance: rescale supply so the historic window
    # averages a tiny ~0.2% surplus over demand. Removes runaway stock builds
    # caused by base_supply/base_demand calibration drift.
    n_hist = len(idx)
    hist_supply_mean = float(np.mean(supply_monthly[:n_hist]))
    hist_demand_mean = float(np.mean(demand_monthly[:n_hist]))
    if hist_supply_mean > 0:
        scale = (hist_demand_mean * 1.002) / hist_supply_mean
        supply = supply * scale
        supply_monthly = supply_monthly * scale

    if tpl.unit.endswith("/d"):
        starting_stocks = tpl.base_demand * tpl.days_cover_target
    else:
        starting_stocks = tpl.base_demand * (tpl.days_cover_target / 30.0)
    # Keep starting stocks within a sensible band relative to capacity.
    starting_stocks = min(max(starting_stocks, 0.2 * tpl.storage_capacity),
                          0.7 * tpl.storage_capacity)

    stocks = np.empty(n)
    stocks[0] = starting_stocks
    cap_soft = tpl.storage_capacity * 1.2  # soft cap to keep series finite
    for i in range(1, n):
        candidate = stocks[i - 1] + (supply_monthly[i] - demand_monthly[i])
        stocks[i] = float(np.clip(candidate, 0.0, cap_soft))

    avg_daily_demand = demand_monthly / days_in_month
    days_cover = stocks / np.where(avg_daily_demand > 0, avg_daily_demand, 1)
    norm_dc = (days_cover - days_cover.mean()) / max(days_cover.std(), 1e-6)
    lo, hi = tpl.price_band
    mid = (lo + hi) / 2.0
    price = mid * np.exp(-0.18 * norm_dc) * (1 + rng.normal(0, 0.04, n))
    price = np.clip(price, lo * 0.7, hi * 1.3)

    gdp_index = 100 * np.cumprod(1 + rng.normal(0.002, 0.004, n))
    weather_index = 50 + 10 * np.sin(2 * np.pi * np.arange(n) / 12 + 1.1) + rng.normal(0, 2, n)
    refinery_runs = np.clip(85 + 8 * seas_d / seas_d.mean() + rng.normal(0, 1.5, n), 70, 98)

    df = pd.DataFrame({
        "date": full_idx, "supply": supply, "demand": demand,
        "imports": imports, "exports": exports, "stocks": stocks,
        "days_cover": days_cover, "price": price, "gdp_index": gdp_index,
        "weather_index": weather_index, "refinery_runs": refinery_runs,
        "is_forecast": [d in horizon for d in full_idx],
    }).set_index("date")
    return df


@st.cache_data(ttl=600, show_spinner=False)
def get_regional_dataset(commodity_key: str, seed: int = 7) -> pd.DataFrame:
    """Synthetic regional split using template `region_weights` (falls back to uniform)."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    rng = np.random.default_rng(seed)
    regions = tpl.regions
    n = len(regions)

    if tpl.region_weights and len(tpl.region_weights) == n:
        weights = np.array(tpl.region_weights, dtype=float)
    else:
        weights = np.ones(n)
    weights = weights / weights.sum()

    demand = tpl.base_demand * weights * (1 + rng.normal(0, 0.03, n))
    skew = rng.normal(1.0, 0.20, n)
    supply = tpl.base_supply * weights * skew
    supply = supply * (tpl.base_supply / supply.sum())
    return pd.DataFrame({
        "region": regions, "supply": supply, "demand": demand,
        "net_trade": supply - demand,
    })


# ---------------------------------------------------------------------------
# Live futures curve (inspired by cfcap / tst3.py)
# ---------------------------------------------------------------------------
_MONTH_CODES = list("FGHJKMNQUVXZ")  # CME futures month codes Jan..Dec
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _build_contract_tickers(tpl: CommodityTemplate, n_max: int) -> List[Dict]:
    """
    Build the list of *active* (non-expired) contracts for a commodity.

    Walks calendar months forward, keeps those whose delivery letter is in
    `active_months`, and skips contracts whose expiry has already passed
    (rule of thumb: contract expires ~20th of the month preceding delivery).
    """
    if not tpl.yf_fmt:
        return []
    from datetime import datetime
    now = datetime.now()
    contracts: List[Dict] = []
    offset = 0
    while len(contracts) < n_max and offset < n_max * 4:
        m = (now.month - 1 + offset) % 12
        year = now.year + (now.month - 1 + offset) // 12
        offset += 1
        if _MONTH_CODES[m] not in tpl.active_months:
            continue
        # expiry ~20th of month preceding delivery
        exp_m = m - 1 if m > 0 else 11
        exp_y = year if m > 0 else year - 1
        if now > datetime(exp_y, exp_m + 1, 20):
            continue
        yr2 = f"{year}"[-2:]
        ticker = tpl.yf_fmt.replace("{M}", _MONTH_CODES[m]).replace("{YY}", yr2)
        contracts.append({
            "ticker": ticker,
            "label": f"{_MONTH_NAMES[m]}-{year}",
            "month_code": _MONTH_CODES[m],
            "tenor_month": len(contracts) + 1,
        })
    return contracts


@st.cache_data(ttl=900, show_spinner=False)
def get_live_futures_curve(commodity_key: str, n_max: int = 12
                           ) -> Optional[pd.DataFrame]:
    """
    Fetch the live forward curve from Yahoo, contract by contract.

    Returns a DataFrame with columns tenor_month, label, ticker, price, expiry,
    or None if no template/yfinance/connectivity.  Same approach as cfcap:
    download each maturity, keep only those with a usable close, renumber
    tenors so the result has no gaps.
    """
    tpl = COMMODITY_TEMPLATES[commodity_key]
    if not tpl.yf_fmt or os.environ.get("COMMODITY_SD_DISABLE_YF") == "1":
        return None
    contracts = _build_contract_tickers(tpl, n_max=n_max)
    if not contracts:
        return None
    try:
        import yfinance as yf
        tickers = [c["ticker"] for c in contracts]
        raw = yf.download(tickers, period="5d", auto_adjust=True,
                          progress=False, group_by="ticker")
        if raw is None or raw.empty:
            return None
        results = []
        for c in contracts:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    close = raw[c["ticker"]]["Close"].dropna()
                else:
                    close = raw["Close"].dropna()
                if close.empty:
                    continue
                results.append({**c, "price": float(close.iloc[-1])})
            except Exception:
                continue
        if len(results) < 2:
            return None
        df = pd.DataFrame(results).sort_values("tenor_month").reset_index(drop=True)
        df["tenor_month"] = range(1, len(df) + 1)
        # Derive an approximate expiry date for plotting (1st of delivery month)
        df["expiry"] = pd.date_range(
            start=pd.Timestamp.today().normalize() + pd.offsets.MonthBegin(1),
            periods=len(df), freq="MS",
        )
        return df
    except Exception as exc:
        logger.warning("Live futures curve fetch failed for %s: %s",
                       commodity_key, exc)
        return None


@st.cache_data(ttl=600, show_spinner=False)
def get_synthetic_futures_curve(commodity_key: str, structure: str = "contango",
                                months: int = 24, seed: int = 5) -> pd.DataFrame:
    """Cost-of-carry style synthetic curve used as fallback."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    spot = tpl.base_price
    rng = np.random.default_rng(seed)
    tenors = np.arange(1, months + 1)
    slope = {"contango": 0.005, "backwardation": -0.006, "flat": 0.0}.get(structure, 0.0)
    log_price = np.log(spot) + slope * tenors + rng.normal(0, 0.005, months)
    prices = np.exp(log_price)
    dates = pd.date_range(start=pd.Timestamp.today().normalize() + pd.offsets.MonthBegin(1),
                          periods=months, freq="MS")
    return pd.DataFrame({"tenor_month": tenors, "expiry": dates, "price": prices})


# Keep the old name as a thin wrapper so existing callers don't break.
def get_futures_curve(commodity_key: str, structure: str = "contango",
                      months: int = 24, seed: int = 5) -> pd.DataFrame:
    return get_synthetic_futures_curve(commodity_key, structure, months, seed)


@st.cache_data(ttl=600, show_spinner=False)
def get_high_frequency(commodity_key: str, days: int = 120, seed: int = 11) -> pd.DataFrame:
    tpl = COMMODITY_TEMPLATES[commodity_key]
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    vessels = np.clip(70 + 5 * np.sin(np.arange(days) / 8) + rng.normal(0, 3, days), 40, 110)
    refinery_util = np.clip(88 + 2 * np.sin(np.arange(days) / 14) + rng.normal(0, 1.2, days), 75, 98)
    power_burn = np.clip(40 + 5 * np.sin(np.arange(days) / 7 + 0.4) + rng.normal(0, 1.5, days), 25, 60)
    weather_hdd = np.clip(15 + 10 * np.sin(np.arange(days) / 30) + rng.normal(0, 2, days), 0, 40)
    sat_prod = np.clip(tpl.base_supply * (1 + rng.normal(0, 0.005, days)), 0, None)
    return pd.DataFrame({
        "date": idx, "vessels_tracked": vessels.astype(int),
        "refinery_util_pct": refinery_util, "power_burn": power_burn,
        "weather_hdd": weather_hdd, "sat_production_est": sat_prod,
    }).set_index("date")


@st.cache_data(ttl=600, show_spinner=False)
def get_macro_panel(months: int = 84, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.today().normalize().replace(day=1),
                        periods=months, freq="MS")
    gdp = 100 * np.cumprod(1 + rng.normal(0.0018, 0.003, months))
    pmi = np.clip(50 + 3 * np.sin(np.arange(months) / 8) + rng.normal(0, 1.2, months), 40, 60)
    usd = 100 * np.cumprod(1 + rng.normal(0.0005, 0.005, months))
    rates = np.clip(2 + 1.5 * np.sin(np.arange(months) / 24) + rng.normal(0, 0.2, months), 0, 7)
    return pd.DataFrame({
        "date": idx, "gdp_index": gdp, "pmi": pmi,
        "usd_index": usd, "policy_rate": rates,
    }).set_index("date")


@st.cache_data(ttl=600, show_spinner=False)
def get_positioning(commodity_key: str, weeks: int = 120, seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=weeks, freq="W-TUE")
    trend = np.cumsum(rng.normal(0, 1, weeks))
    managed_money_net = 80_000 + 30_000 * np.sin(np.arange(weeks) / 12) + 6_000 * np.sign(trend)
    open_interest = 1_500_000 + 100_000 * np.sin(np.arange(weeks) / 24) + rng.normal(0, 20_000, weeks)
    cta_signal = np.tanh(np.gradient(managed_money_net) / 8_000)
    sentiment = np.clip(50 + 25 * np.tanh(managed_money_net / 80_000), 0, 100)
    return pd.DataFrame({
        "date": idx, "managed_money_net": managed_money_net,
        "open_interest": open_interest, "cta_signal": cta_signal,
        "sentiment_score": sentiment,
    }).set_index("date")


def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    return df


@st.cache_data(ttl=300, show_spinner=False)
def get_yahoo_history(commodity_key: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Récupère l'historique de prix Yahoo Finance pour la commodité, ou None si indisponible."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    if os.environ.get("COMMODITY_SD_DISABLE_YF") == "1":
        return None
    try:
        import yfinance as yf
        hist = yf.Ticker(tpl.ticker).history(period=period, auto_adjust=False)
        if hist is None or hist.empty:
            return None
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as exc:
        logger.warning("Yahoo fetch failed: %s", exc)
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_live_spot(commodity_key: str) -> Optional[Dict[str, float]]:
    """Renvoie le dernier prix coté en direct (Yahoo) avec date et variation 1j."""
    hist = get_yahoo_history(commodity_key, period="5d")
    if hist is None or hist.empty:
        return None
    close = float(hist["Close"].iloc[-1])
    prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else close
    change_pct = (close - prev) / prev * 100 if prev else 0.0
    return {
        "price": close,
        "change_pct": change_pct,
        "asof": hist.index[-1].strftime("%Y-%m-%d"),
    }


def fmt_price(value: float, unit: str) -> str:
    """Formate un prix avec son unité, ex: '78.42 $/bbl' ou '2.40 $/gal'."""
    if not isinstance(value, (int, float)) or pd.isna(value):
        return "—"
    if "¢" in unit:
        return f"{value:,.2f} {unit}"
    return f"{value:,.2f} {unit}"


# =============================================================================
# MODELS - BALANCE ENGINE
# =============================================================================

Frequency = Literal["M", "Q", "Y"]


@dataclass
class BalanceAssumptions:
    """User-tunable assumptions for the balance engine."""

    beginning_stocks: Optional[float] = None
    supply_adj_pct: float = 0.0
    demand_adj_pct: float = 0.0
    imports_adj_pct: float = 0.0
    exports_adj_pct: float = 0.0
    refinery_runs_pct: float = 0.0
    weather_pct: float = 0.0
    gdp_growth_pct: float = 0.0
    storage_capacity: Optional[float] = None
    forecast_months: int = 24
    extra: Dict[str, float] = field(default_factory=dict)


def _apply_adjustments(df: pd.DataFrame, tpl: CommodityTemplate,
                       a: BalanceAssumptions) -> pd.DataFrame:
    out = df.copy()
    mask = out["is_forecast"]
    out.loc[mask, "supply"] *= 1 + a.supply_adj_pct / 100.0
    base_demand_mult = (
        1 + a.demand_adj_pct / 100.0
        + a.weather_pct / 100.0
        + (a.gdp_growth_pct - 2.5) / 100.0 * 0.6
    )
    out.loc[mask, "demand"] *= base_demand_mult
    if tpl.key == "crude_oil":
        out.loc[mask, "demand"] *= 1 + a.refinery_runs_pct / 100.0 * 0.4
    out.loc[mask, "imports"] *= 1 + a.imports_adj_pct / 100.0
    out.loc[mask, "exports"] *= 1 + a.exports_adj_pct / 100.0
    return out


def run_balance(df: pd.DataFrame, commodity_key: str,
                assumptions: Optional[BalanceAssumptions] = None,
                frequency: Frequency = "M") -> pd.DataFrame:
    """Run the balance engine over a S&D dataframe."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    a = assumptions or BalanceAssumptions()
    adj = _apply_adjustments(df, tpl, a)

    days = adj.index.days_in_month.to_numpy()
    if tpl.unit.endswith("/d"):
        supply_total = adj["supply"].to_numpy() * days
        demand_total = adj["demand"].to_numpy() * days
    else:
        supply_total = adj["supply"].to_numpy()
        demand_total = adj["demand"].to_numpy()

    imports = adj["imports"].to_numpy()
    exports = adj["exports"].to_numpy()
    net_trade = imports - exports

    start = (a.beginning_stocks if a.beginning_stocks is not None
             else float(adj["stocks"].iloc[0]))
    storage_cap = a.storage_capacity or tpl.storage_capacity
    # Soft cap of 1.3× capacity to represent the real-world release valves
    # (floating storage, ad-hoc exports, rationing) and keep synthetic series
    # from drifting unbounded. The Inventories page enforces a hard cap.
    soft_cap = storage_cap * 1.3
    stocks = np.empty(len(adj))
    stocks[0] = float(np.clip(start, 0.0, soft_cap))
    for i in range(1, len(adj)):
        delta = supply_total[i] - demand_total[i] + (net_trade[i] - net_trade.mean()) * 0.05
        stocks[i] = float(np.clip(stocks[i - 1] + delta, 0.0, soft_cap))
    avg_daily_demand = demand_total / days
    avg_daily_demand = np.where(avg_daily_demand <= 0, 1e-6, avg_daily_demand)
    days_cover = stocks / avg_daily_demand

    out = adj.copy()
    out["supply_adj"] = adj["supply"]
    out["demand_adj"] = adj["demand"]
    out["net_trade"] = net_trade
    out["build_draw"] = supply_total - demand_total
    out["stocks_model"] = stocks
    out["days_cover_model"] = days_cover
    out["surplus_deficit"] = supply_total - demand_total
    out["capacity_pct"] = (stocks / storage_cap) * 100.0

    if frequency == "M":
        return out
    rule = "QE" if frequency == "Q" else "YE"
    agg = {
        "supply": "mean", "demand": "mean", "imports": "mean", "exports": "mean",
        "supply_adj": "mean", "demand_adj": "mean", "net_trade": "mean",
        "build_draw": "sum", "stocks_model": "last", "days_cover_model": "mean",
        "surplus_deficit": "sum", "capacity_pct": "last", "price": "mean",
        "stocks": "last", "days_cover": "mean", "gdp_index": "last",
        "weather_index": "mean", "refinery_runs": "mean", "is_forecast": "max",
    }
    cols = [c for c in agg if c in out.columns]
    return out[cols].resample(rule).agg({c: agg[c] for c in cols})


# =============================================================================
# MODELS - SEASONALITY
# =============================================================================

def monthly_profile(series: pd.Series, years: int = 5) -> pd.DataFrame:
    s = series.dropna()
    cutoff = s.index.max() - pd.DateOffset(years=years)
    recent = s[s.index >= cutoff]
    grouped = recent.groupby(recent.index.month)
    df = pd.DataFrame({
        "mean": grouped.mean(), "std": grouped.std(),
        "min": grouped.min(), "max": grouped.max(),
    })
    current_year = s.index.max().year
    cur = s[s.index.year == current_year]
    df["current"] = cur.groupby(cur.index.month).mean()
    df.index.name = "month"
    return df


def rolling_seasonal_average(series: pd.Series, window: int = 12) -> pd.Series:
    return series.rolling(window=window, min_periods=max(3, window // 3)).mean()


def normalize_seasonal(series: pd.Series) -> pd.Series:
    monthly_mean = series.groupby(series.index.month).transform("mean")
    return series / monthly_mean


def decompose(series: pd.Series, period: int = 12) -> Tuple[pd.Series, pd.Series, pd.Series]:
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        s = series.dropna()
        if len(s) < 2 * period:
            raise ValueError("series too short")
        res = seasonal_decompose(s, model="additive", period=period, extrapolate_trend="freq")
        return res.trend, res.seasonal, res.resid
    except Exception:
        trend = series.rolling(period, min_periods=period // 2, center=True).mean()
        detrended = series - trend
        seasonal = detrended.groupby(detrended.index.month).transform("mean")
        residual = series - trend - seasonal
        return trend, seasonal, residual


def year_over_year_pivot(series: pd.Series) -> pd.DataFrame:
    s = series.dropna()
    pivot = pd.DataFrame({"year": s.index.year, "month": s.index.month, "value": s.values})
    return pivot.pivot_table(index="year", columns="month", values="value", aggfunc="mean")


# =============================================================================
# MODELS - INVENTORY / STORAGE
# =============================================================================

@dataclass
class StorageConfig:
    capacity: Optional[float] = None
    floating_buffer_pct: float = 5.0
    allow_negative: bool = False


def project_inventory(df: pd.DataFrame, commodity_key: str,
                      config: Optional[StorageConfig] = None) -> pd.DataFrame:
    tpl = COMMODITY_TEMPLATES[commodity_key]
    cfg = config or StorageConfig()
    cap = cfg.capacity or tpl.storage_capacity
    floating_cap = cap * (cfg.floating_buffer_pct / 100.0)

    out = df.copy()
    stocks = np.empty(len(out))
    floating = np.zeros(len(out))
    stocks[0] = float(out["stocks_model"].iloc[0])
    bd = out["build_draw"].to_numpy()
    for i in range(1, len(out)):
        candidate = stocks[i - 1] + bd[i]
        if not cfg.allow_negative:
            candidate = max(candidate, 0.0)
        if candidate > cap:
            spillover = candidate - cap
            floating[i] = min(spillover, floating_cap)
            stocks[i] = cap
        else:
            stocks[i] = candidate
    out["stocks_capped"] = stocks
    out["overflow_floating"] = floating
    out["utilization_pct"] = (stocks / cap) * 100.0
    return out


def draw_build_waterfall(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    s = df["build_draw"].iloc[-n:]
    return pd.DataFrame({"period": s.index.strftime("%b-%y"), "delta": s.values})


def days_of_forward_cover(df: pd.DataFrame) -> pd.Series:
    forward = df["demand"].rolling(window=2, min_periods=1).mean()
    days = df.index.days_in_month
    daily = forward / days
    cover = df["stocks_model"] / daily.replace(0, np.nan)
    return cover.ffill().clip(lower=0)


# =============================================================================
# MODELS - ELASTICITY
# =============================================================================

@dataclass
class ElasticityParams:
    alpha: float = 0.06
    beta: float = 0.10
    base_price: float = 0.0
    d0: float = 0.0
    s0: float = 0.0


def demand_curve(prices: np.ndarray, p: ElasticityParams) -> np.ndarray:
    return p.d0 * (1.0 - p.alpha * (prices - p.base_price) / p.base_price)


def supply_curve(prices: np.ndarray, p: ElasticityParams) -> np.ndarray:
    return p.s0 * (1.0 + p.beta * (prices - p.base_price) / p.base_price)


def equilibrium(p: ElasticityParams) -> Tuple[float, float]:
    denom = p.d0 * p.alpha + p.s0 * p.beta
    if abs(denom) < 1e-12:
        return p.base_price, p.d0
    p_eq = p.base_price * (1.0 + (p.d0 - p.s0) / denom)
    q_eq = demand_curve(np.array([p_eq]), p)[0]
    return float(p_eq), float(q_eq)


def build_curves(commodity_key: str, alpha: float, beta: float,
                 price_band_pct: float = 0.5, n: int = 80) -> pd.DataFrame:
    tpl = COMMODITY_TEMPLATES[commodity_key]
    p = ElasticityParams(alpha=alpha, beta=beta, base_price=tpl.base_price,
                         d0=tpl.base_demand, s0=tpl.base_supply)
    prices = np.linspace(tpl.base_price * (1 - price_band_pct),
                         tpl.base_price * (1 + price_band_pct), n)
    return pd.DataFrame({
        "price": prices, "demand": demand_curve(prices, p),
        "supply": supply_curve(prices, p),
    })


# =============================================================================
# MODELS - LAGGED RESPONSE
# =============================================================================

@dataclass
class LaggedRegressionResult:
    coefficients: np.ndarray
    intercept: float
    r_squared: float
    lag: int


def fit_lagged_supply(df: pd.DataFrame, lag_months: int = 6,
                      price_col: str = "price",
                      supply_col: str = "supply") -> LaggedRegressionResult:
    s = df[[price_col, supply_col]].dropna()
    if len(s) <= lag_months + 5:
        raise ValueError("Not enough observations for the requested lag")
    X = pd.concat([s[price_col].shift(i) for i in range(1, lag_months + 1)], axis=1)
    X.columns = [f"price_lag_{i}" for i in range(1, lag_months + 1)]
    y = s[supply_col]
    data = pd.concat([X, y], axis=1).dropna()
    model = LinearRegression().fit(data[X.columns], data[supply_col])
    r2 = model.score(data[X.columns], data[supply_col])
    return LaggedRegressionResult(
        coefficients=model.coef_, intercept=float(model.intercept_),
        r_squared=float(r2), lag=lag_months,
    )


def project_lagged_response(base_supply: float, price_shock_pct: float,
                            lag_months: int, horizon: int = 24,
                            response_strength: float = 0.08) -> pd.Series:
    months = np.arange(horizon)
    delayed = 1.0 / (1.0 + np.exp(-(months - lag_months) / 2.0))
    response_pct = response_strength * (price_shock_pct / 100.0) * delayed
    series = base_supply * (1.0 + response_pct)
    idx = pd.date_range(start=pd.Timestamp.today().normalize().replace(day=1),
                        periods=horizon, freq="MS")
    return pd.Series(series, index=idx, name="lagged_supply")


# =============================================================================
# MODELS - FAIR VALUE
# =============================================================================

def _dc_col(df: pd.DataFrame) -> str:
    return "days_cover_model" if "days_cover_model" in df.columns else "days_cover"


def fit_fair_value(df: pd.DataFrame) -> Dict:
    dc = _dc_col(df)
    hist = df[~df["is_forecast"]].dropna(subset=["price", dc])
    X = hist[[dc]].to_numpy()
    y = np.log(hist["price"].to_numpy())
    model = LinearRegression().fit(X, y)
    return {
        "intercept": float(model.intercept_), "slope": float(model.coef_[0]),
        "r_squared": float(model.score(X, y)),
        "mean_dc": float(hist[dc].mean()),
        "mean_price": float(hist["price"].mean()),
    }


def estimate_fair_value(df: pd.DataFrame, commodity_key: str) -> pd.DataFrame:
    tpl = COMMODITY_TEMPLATES[commodity_key]
    dc = _dc_col(df)
    try:
        fit = fit_fair_value(df)
        log_fv = fit["intercept"] + fit["slope"] * df[dc]
        fv = np.exp(log_fv)
    except Exception:
        lo, hi = tpl.price_band
        mid = (lo + hi) / 2
        fv = mid * (1 - 0.15 * (df[dc] - df[dc].mean())
                    / max(df[dc].std(), 1e-6))
    out = df.copy()
    out["fair_value_price"] = fv
    out["fv_residual_pct"] = (out["price"] - fv) / fv * 100.0
    out["fv_signal"] = np.where(
        out["fv_residual_pct"] > 10, "Overvalued",
        np.where(out["fv_residual_pct"] < -10, "Undervalued", "Fair"),
    )
    return out


def marginal_cost_curve(commodity_key: str, n_quantiles: int = 10) -> pd.DataFrame:
    tpl = COMMODITY_TEMPLATES[commodity_key]
    lo, hi = tpl.price_band
    q = np.linspace(0.02, 0.98, n_quantiles)
    costs = lo + (hi - lo) * q ** 1.4
    return pd.DataFrame({"cum_share_pct": q * 100.0, "marginal_cost": costs})


# =============================================================================
# MODELS - SCENARIO ENGINE
# =============================================================================

def build_assumptions_from_preset(name: str,
                                  base: BalanceAssumptions) -> BalanceAssumptions:
    preset = SCENARIO_PRESETS[name]
    return BalanceAssumptions(
        beginning_stocks=base.beginning_stocks,
        supply_adj_pct=preset["supply_shock_pct"],
        demand_adj_pct=preset["demand_shock_pct"],
        weather_pct=preset["weather_shock_pct"],
        gdp_growth_pct=preset["gdp_growth_pct"],
        storage_capacity=base.storage_capacity,
        forecast_months=base.forecast_months,
        extra={"fx_usd_pct": preset["fx_usd_pct"]},
    )


def run_scenarios(df: pd.DataFrame, commodity_key: str,
                  base_assumptions: BalanceAssumptions,
                  scenarios: Iterable[str] = ("Bull", "Base", "Bear"),
                  ) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    for name in scenarios:
        a = build_assumptions_from_preset(name, base_assumptions)
        bal = run_balance(df, commodity_key, a)
        bal["fair_value_price"] = estimate_fair_value(bal, commodity_key)["fair_value_price"]
        results[name] = bal
    return results


def scenario_summary(results: Dict[str, pd.DataFrame], commodity_key: str) -> pd.DataFrame:
    tpl = COMMODITY_TEMPLATES[commodity_key]
    rows = []
    for name, bal in results.items():
        last = bal.iloc[-1]
        rows.append({
            "Scenario": name, "Probability": SCENARIO_PRESETS[name]["probability"],
            "End Stocks": last["stocks_model"], "Days Cover": last["days_cover_model"],
            "Build/Draw (last 12M)": bal["build_draw"].iloc[-12:].sum(),
            "Avg Price (forecast)": bal.loc[bal["is_forecast"], "price"].mean(),
            "Fair Value (end)": last["fair_value_price"], "Unit": tpl.inventory_unit,
        })
    return pd.DataFrame(rows).set_index("Scenario")


def probability_weighted_price(results: Dict[str, pd.DataFrame]) -> float:
    pw = 0.0
    for name, bal in results.items():
        prob = SCENARIO_PRESETS[name]["probability"]
        pw += prob * bal.loc[bal["is_forecast"], "price"].mean()
    return float(pw)


# =============================================================================
# MODELS - REGIONAL
# =============================================================================

def regional_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["balance"] = out["supply"] - out["demand"]
    out["status"] = np.where(out["balance"] > 0, "Exporter",
                             np.where(out["balance"] < 0, "Importer", "Balanced"))
    return out


def build_trade_flows(df: pd.DataFrame) -> Tuple[List[str], List[int], List[int], List[float]]:
    rs = regional_summary(df)
    exporters = rs[rs["balance"] > 0].copy()
    importers = rs[rs["balance"] < 0].copy()
    if exporters.empty or importers.empty:
        return [], [], [], []
    importers["deficit"] = -importers["balance"]
    importers["share"] = importers["deficit"] / importers["deficit"].sum()

    nodes = list(exporters["region"]) + list(importers["region"])
    sources, targets, values = [], [], []
    for i, ex_row in enumerate(exporters.itertuples(index=False)):
        for j, im_row in enumerate(importers.itertuples(index=False)):
            flow = ex_row.balance * im_row.share
            if flow > 0:
                sources.append(i)
                targets.append(len(exporters) + j)
                values.append(float(flow))
    return nodes, sources, targets, values


def arbitrage_signals(df: pd.DataFrame) -> pd.DataFrame:
    rs = regional_summary(df)
    rs["arb_signal"] = np.where(rs["balance"] > 0, "Export Arb",
                                np.where(rs["balance"] < 0, "Import Need", "Neutral"))
    return rs


# =============================================================================
# MODELS - FUTURES CURVE
# =============================================================================

def classify_structure(curve: pd.DataFrame) -> str:
    if len(curve) < 2:
        return "Mixed"
    diffs = np.diff(curve["price"].to_numpy())
    if np.all(diffs > 0):
        return "Contango"
    if np.all(diffs < 0):
        return "Backwardation"
    if diffs.mean() > 0:
        return "Contango (Mixed)"
    if diffs.mean() < 0:
        return "Backwardation (Mixed)"
    return "Flat"


def calendar_spreads(curve: pd.DataFrame) -> pd.DataFrame:
    p = curve.set_index("tenor_month")["price"]

    def diff(a: int, b: int) -> float:
        if a in p.index and b in p.index:
            return float(p.loc[a] - p.loc[b])
        return float("nan")

    rows = [("m1 - m2", diff(1, 2)), ("m1 - m6", diff(1, 6)),
            ("m1 - m12", diff(1, 12)), ("m6 - m12", diff(6, 12))]
    return pd.DataFrame(rows, columns=["Spread", "Value"]).set_index("Spread")


def storage_economics(curve: pd.DataFrame, storage_cost_per_month: float,
                      financing_rate_pct: float) -> pd.DataFrame:
    p = curve.set_index("tenor_month")["price"]
    base = float(p.iloc[0])
    out = curve.copy().set_index("tenor_month")
    out["carry"] = (storage_cost_per_month
                    + (financing_rate_pct / 100.0 / 12.0) * base) * out.index
    out["contango_premium"] = out["price"] - base
    out["positive_carry"] = out["contango_premium"] > out["carry"]
    return out.reset_index()


def inventory_curve_relationship(curve: pd.DataFrame, days_cover: float) -> Tuple[str, float]:
    structure = classify_structure(curve)
    z = (35 - days_cover) / 20.0
    score = float(np.clip(z, -1.0, 1.0))
    label = ("Tight (supports backwardation)" if score > 0.3
             else "Loose (supports contango)" if score < -0.3
             else "Balanced")
    return f"{label} - observed: {structure}", score


# =============================================================================
# MODELS - MACRO
# =============================================================================

def align_macro(sd: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    macro = macro.copy()
    macro.index = pd.to_datetime(macro.index)
    sd = sd.drop(columns=[c for c in sd.columns if c in macro.columns], errors="ignore")
    return sd.join(macro, how="inner")


def correlation_matrix(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    available = [c for c in cols if c in df.columns]
    return df[available].corr().round(3)


def rolling_correlation(s1: pd.Series, s2: pd.Series, window: int = 24) -> pd.Series:
    return s1.rolling(window=window, min_periods=window // 2).corr(s2)


def regression_summary(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> Dict:
    data = df[[y_col] + x_cols].dropna()
    if len(data) < len(x_cols) + 3:
        raise ValueError("Not enough data to fit regression")
    model = LinearRegression().fit(data[x_cols], data[y_col])
    pred = model.predict(data[x_cols])
    resid = data[y_col].to_numpy() - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((data[y_col] - data[y_col].mean()) ** 2))
    return {
        "coefficients": dict(zip(x_cols, model.coef_)),
        "intercept": float(model.intercept_),
        "r_squared": float(1 - ss_res / max(ss_tot, 1e-12)),
        "n_obs": int(len(data)),
    }


# =============================================================================
# MODELS - MONTE CARLO
# =============================================================================

@dataclass
class MCConfig:
    n_paths: int = 500
    supply_sigma_pct: float = 1.5
    demand_sigma_pct: float = 1.2
    weather_sigma_pct: float = 1.0
    outage_prob: float = 0.05
    outage_size_pct: float = 4.0
    seed: int = 2024


def run_monte_carlo(df: pd.DataFrame, commodity_key: str,
                    base_assumptions: BalanceAssumptions, cfg: MCConfig) -> Dict:
    tpl = COMMODITY_TEMPLATES[commodity_key]
    rng = np.random.default_rng(cfg.seed)
    base_bal = run_balance(df, commodity_key, base_assumptions)
    fc_mask = base_bal["is_forecast"]
    fc_idx = base_bal.index[fc_mask]
    n_fc = int(fc_mask.sum())

    end_stocks = np.empty(cfg.n_paths)
    avg_price = np.empty(cfg.n_paths)
    build_draw = np.empty(cfg.n_paths)
    paths_price = np.empty((cfg.n_paths, n_fc))
    paths_stocks = np.empty((cfg.n_paths, n_fc))

    for p in range(cfg.n_paths):
        a = BalanceAssumptions(
            beginning_stocks=base_assumptions.beginning_stocks,
            supply_adj_pct=base_assumptions.supply_adj_pct + rng.normal(0.0, cfg.supply_sigma_pct),
            demand_adj_pct=base_assumptions.demand_adj_pct + rng.normal(0.0, cfg.demand_sigma_pct),
            imports_adj_pct=base_assumptions.imports_adj_pct,
            exports_adj_pct=base_assumptions.exports_adj_pct,
            refinery_runs_pct=base_assumptions.refinery_runs_pct,
            weather_pct=base_assumptions.weather_pct + rng.normal(0.0, cfg.weather_sigma_pct),
            gdp_growth_pct=base_assumptions.gdp_growth_pct,
            storage_capacity=base_assumptions.storage_capacity,
            forecast_months=base_assumptions.forecast_months,
        )
        bal = run_balance(df, commodity_key, a)
        if rng.random() < cfg.outage_prob * n_fc:
            month = rng.integers(0, max(n_fc, 1))
            bal.iloc[-(n_fc - month):, bal.columns.get_loc("supply_adj")] *= (
                1.0 - cfg.outage_size_pct / 100.0
            )
            days = bal.index.days_in_month.to_numpy()
            mul = days if tpl.unit.endswith("/d") else np.ones_like(days)
            bd = bal["supply_adj"].to_numpy() * mul - bal["demand_adj"].to_numpy() * mul
            stocks = bal["stocks_model"].to_numpy().copy()
            for i in range(1, len(stocks)):
                stocks[i] = max(stocks[i - 1] + bd[i], 0.0)
            bal["stocks_model"] = stocks
        bal = estimate_fair_value(bal, commodity_key)
        end_stocks[p] = float(bal["stocks_model"].iloc[-1])
        avg_price[p] = float(bal.loc[fc_mask, "fair_value_price"].mean())
        build_draw[p] = float(bal.loc[fc_mask, "build_draw"].sum())
        paths_price[p, :] = bal.loc[fc_mask, "fair_value_price"].to_numpy()
        paths_stocks[p, :] = bal.loc[fc_mask, "stocks_model"].to_numpy()

    pp = pd.DataFrame(paths_price.T, index=fc_idx,
                      columns=[f"p{i}" for i in range(cfg.n_paths)])
    ps = pd.DataFrame(paths_stocks.T, index=fc_idx,
                      columns=[f"p{i}" for i in range(cfg.n_paths)])
    pct = pd.DataFrame({
        "p5_price": pp.quantile(0.05, axis=1),
        "p50_price": pp.quantile(0.50, axis=1),
        "p95_price": pp.quantile(0.95, axis=1),
        "p5_stocks": ps.quantile(0.05, axis=1),
        "p50_stocks": ps.quantile(0.50, axis=1),
        "p95_stocks": ps.quantile(0.95, axis=1),
    })
    return {"end_stocks": end_stocks, "avg_price": avg_price, "build_draw": build_draw,
            "paths_price": pp, "paths_stocks": ps, "percentiles": pct}


def value_at_risk(losses: np.ndarray, alpha: float = 0.95) -> float:
    return float(np.quantile(losses, alpha))


# =============================================================================
# MODELS - SENSITIVITY
# =============================================================================

@dataclass
class SensitivityVar:
    name: str
    attr: str
    low: float
    high: float


def _eval(df: pd.DataFrame, commodity_key: str, a: BalanceAssumptions, metric: str) -> float:
    bal = run_balance(df, commodity_key, a)
    if metric == "end_stocks":
        return float(bal["stocks_model"].iloc[-1])
    if metric == "avg_fc_price":
        return float(bal.loc[bal["is_forecast"], "price"].mean())
    if metric == "build_draw_sum":
        return float(bal["build_draw"].iloc[-12:].sum())
    raise ValueError(f"Unknown metric: {metric}")


def tornado(df: pd.DataFrame, commodity_key: str, base: BalanceAssumptions,
            variables: Iterable[SensitivityVar], metric: str = "end_stocks") -> pd.DataFrame:
    base_val = _eval(df, commodity_key, base, metric)
    rows = []
    for v in variables:
        low_val = _eval(df, commodity_key, replace(base, **{v.attr: v.low}), metric)
        high_val = _eval(df, commodity_key, replace(base, **{v.attr: v.high}), metric)
        rows.append({"variable": v.name, "low": low_val - base_val,
                     "high": high_val - base_val,
                     "low_input": v.low, "high_input": v.high})
    out = pd.DataFrame(rows)
    out["range"] = out["high"].abs() + out["low"].abs()
    return out.sort_values("range", ascending=True)


def stress_matrix(df: pd.DataFrame, commodity_key: str, base: BalanceAssumptions,
                  var_a: SensitivityVar, var_b: SensitivityVar,
                  grid: int = 5, metric: str = "end_stocks") -> pd.DataFrame:
    a_grid = np.linspace(var_a.low, var_a.high, grid)
    b_grid = np.linspace(var_b.low, var_b.high, grid)
    out = pd.DataFrame(index=a_grid, columns=b_grid, dtype=float)
    out.index.name = var_a.name
    out.columns.name = var_b.name
    for av in a_grid:
        for bv in b_grid:
            a = replace(base, **{var_a.attr: av, var_b.attr: bv})
            out.loc[av, bv] = _eval(df, commodity_key, a, metric)
    return out


# =============================================================================
# MODELS - POSITIONING
# =============================================================================

def positioning_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mm_net_4w_avg"] = out["managed_money_net"].rolling(4).mean()
    out["oi_4w_avg"] = out["open_interest"].rolling(4).mean()
    z = (out["managed_money_net"] - out["managed_money_net"].rolling(52).mean()) / \
        out["managed_money_net"].rolling(52).std()
    out["mm_z_score"] = z
    return out


def sentiment_label(score: float) -> str:
    if score >= 75: return "Bullish"
    if score >= 55: return "Mildly Bullish"
    if score >= 45: return "Neutral"
    if score >= 25: return "Mildly Bearish"
    return "Bearish"


# =============================================================================
# THEME
# =============================================================================

def register_theme() -> None:
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
        font=dict(family="Inter, Helvetica Neue, Arial", color="#e5e7eb", size=12),
        xaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937",
                   linecolor="#374151", ticks="outside"),
        yaxis=dict(gridcolor="#1f2937", zerolinecolor="#1f2937",
                   linecolor="#374151", ticks="outside"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#374151",
                    borderwidth=0, orientation="h", y=1.08, x=0),
        margin=dict(l=40, r=20, t=40, b=40),
        colorway=[ACCENT, GREEN, AMBER, "#a78bfa", RED, GREY, "#f97316", "#06b6d4"],
        hoverlabel=dict(bgcolor="#111827", font_size=11),
    )
    pio.templates["trading_desk"] = tpl
    pio.templates.default = "trading_desk"


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.5rem; padding-bottom: 1.5rem;}
            section[data-testid="stSidebar"] > div {background: #0b0f14;}
            div[data-testid="stMetric"] {
                background: #161b22;
                border: 1px solid #1f2937;
                border-radius: 8px;
                padding: 0.55rem 0.8rem;
            }
            /* Smaller KPI value font + tighter label + smaller delta */
            div[data-testid="stMetricValue"] {
                color: #e5e7eb;
                font-size: 1.15rem !important;
                line-height: 1.25 !important;
            }
            div[data-testid="stMetricLabel"] {
                font-size: 0.78rem !important;
                color: #9ca3af !important;
            }
            div[data-testid="stMetricDelta"] {
                font-size: 0.72rem !important;
            }
            h1, h2, h3, h4 {color: #f3f4f6 !important;}
            /* Italic light-gray chart purpose caption */
            .chart-purpose {
                color: #9ca3af;
                font-style: italic;
                font-size: 0.85rem;
                margin: -0.25rem 0 0.45rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# CHART BUILDERS
# =============================================================================

def supply_demand_chart(df: pd.DataFrame, unit: str = "") -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df.index, y=df["supply_adj"] if "supply_adj" in df else df["supply"],
        mode="lines", name="Supply",
        line=dict(color=COLORS["supply"], width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["demand_adj"] if "demand_adj" in df else df["demand"],
        mode="lines", name="Demand",
        line=dict(color=COLORS["demand"], width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["stocks_model"] if "stocks_model" in df else df["stocks"],
        mode="lines", name="Stocks",
        line=dict(color=COLORS["stocks"], width=1.5, dash="dot")), secondary_y=True)
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
    fig.add_trace(go.Scatter(
        x=df.index, y=df["stocks_model"], mode="lines",
        name="Modeled Stocks", line=dict(color=COLORS["stocks"], width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.15)",
    ))
    fig.update_layout(title="Inventory Trajectory", yaxis_title=f"Stocks ({unit})", height=320)
    return fig


def balance_bars(df: pd.DataFrame) -> go.Figure:
    last = df.tail(24)
    colors = np.where(last["build_draw"] >= 0, COLORS["supply"], COLORS["demand"])
    fig = go.Figure(data=[go.Bar(x=last.index, y=last["build_draw"],
                                  marker_color=colors, name="Build/Draw")])
    fig.update_layout(title="Monthly Build / Draw (Last 24M)", yaxis_title="Δ stocks", height=320)
    return fig


def days_cover_chart(df: pd.DataFrame, target: Optional[float] = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["days_cover_model"], mode="lines",
                             name="Days of Cover",
                             line=dict(color=COLORS["fair_value"], width=2)))
    if target is not None:
        fig.add_hline(y=target, line=dict(color=COLORS["price"], dash="dash"),
                      annotation_text=f"Target {target:.0f}d",
                      annotation_position="top right")
    fig.update_layout(title="Days of Forward Cover", yaxis_title="days", height=300)
    return fig


def seasonal_lines(profile: pd.DataFrame) -> go.Figure:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=profile["mean"], name="5y Avg",
                             line=dict(color=COLORS["stocks"], width=2)))
    if "current" in profile:
        fig.add_trace(go.Scatter(x=months, y=profile["current"], name="Current Year",
                                 line=dict(color=COLORS["price"], width=2, dash="dot")))
    fig.add_trace(go.Scatter(
        x=months + months[::-1],
        y=list(profile["max"]) + list(profile["min"][::-1]),
        fill="toself", fillcolor="rgba(156,163,175,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="5y Range",
    ))
    fig.update_layout(title="Monthly Seasonal Profile", height=320)
    return fig


def seasonal_heatmap(pivot: pd.DataFrame, title: str = "Seasonal Heatmap") -> go.Figure:
    fig = px.imshow(pivot.values, x=pivot.columns, y=pivot.index, aspect="auto",
                    color_continuous_scale="RdBu_r", origin="lower",
                    labels=dict(x="Month", y="Year", color="Value"))
    fig.update_layout(title=title, height=340)
    return fig


def waterfall_chart(deltas: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Waterfall(
        x=deltas["period"], y=deltas["delta"],
        measure=["relative"] * len(deltas),
        connector=dict(line=dict(color="#374151")),
        increasing=dict(marker=dict(color=COLORS["supply"])),
        decreasing=dict(marker=dict(color=COLORS["demand"])),
    ))
    fig.update_layout(title="Inventory Build / Draw Waterfall", height=320)
    return fig


def utilization_gauge(util_pct: float, target_pct: float = 80.0) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=util_pct,
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLORS["stocks"]},
            "steps": [
                {"range": [0, 40], "color": "#1f2937"},
                {"range": [40, 75], "color": "#0b3b48"},
                {"range": [75, 100], "color": "#173b1f"},
            ],
            "threshold": {"line": {"color": COLORS["price"], "width": 3},
                          "thickness": 0.75, "value": target_pct},
        },
        title={"text": "Storage Utilisation"},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def elasticity_chart(df: pd.DataFrame, eq_price: float, eq_qty: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["price"], y=df["demand"], name="Demand",
                             line=dict(color=COLORS["demand"], width=2)))
    fig.add_trace(go.Scatter(x=df["price"], y=df["supply"], name="Supply",
                             line=dict(color=COLORS["supply"], width=2)))
    fig.add_trace(go.Scatter(x=[eq_price], y=[eq_qty], name="Equilibrium",
                             mode="markers+text",
                             marker=dict(color=COLORS["price"], size=12, symbol="diamond"),
                             text=[f" ({eq_price:.1f}, {eq_qty:.1f})"],
                             textposition="top right"))
    fig.update_layout(title="Price-Elasticity Curves",
                      xaxis_title="Price", yaxis_title="Quantity", height=380)
    return fig


def scenario_paths(results: Dict, value_col: str = "stocks_model",
                   title: str = "Scenario Paths") -> go.Figure:
    fig = go.Figure()
    cmap = {"Bull": COLORS["bull"], "Base": COLORS["base"], "Bear": COLORS["bear"]}
    for name, bal in results.items():
        fig.add_trace(go.Scatter(x=bal.index, y=bal[value_col], name=name,
                                 line=dict(color=cmap.get(name, COLORS["neutral"]), width=2)))
    fig.update_layout(title=title, height=380)
    return fig


def fan_chart(percentiles: pd.DataFrame, value: str = "price") -> go.Figure:
    p5 = percentiles[f"p5_{value}"]
    p50 = percentiles[f"p50_{value}"]
    p95 = percentiles[f"p95_{value}"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p95.index.tolist() + p5.index[::-1].tolist(),
        y=p95.tolist() + p5[::-1].tolist(),
        fill="toself", fillcolor="rgba(0,212,255,0.18)",
        line=dict(color="rgba(0,0,0,0)"), name="P5-P95 band"))
    fig.add_trace(go.Scatter(x=p50.index, y=p50, name="Median",
                             line=dict(color=COLORS["stocks"], width=2)))
    fig.update_layout(title=f"Probabilistic {value.title()} Fan Chart", height=380)
    return fig


def tornado_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df["variable"], x=df["low"], orientation="h",
                         name="Low", marker_color=COLORS["bear"]))
    fig.add_trace(go.Bar(y=df["variable"], x=df["high"], orientation="h",
                         name="High", marker_color=COLORS["bull"]))
    fig.update_layout(barmode="relative", title="Tornado Sensitivity",
                      xaxis_title="Δ vs Base", height=380)
    return fig


def regional_bar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Supply", x=df["region"], y=df["supply"],
                         marker_color=COLORS["supply"]))
    fig.add_trace(go.Bar(name="Demand", x=df["region"], y=df["demand"],
                         marker_color=COLORS["demand"]))
    fig.update_layout(barmode="group", title="Regional Supply vs Demand", height=320)
    return fig


def sankey_chart(nodes: List[str], sources: List[int], targets: List[int],
                 values: List[float]) -> go.Figure:
    if not nodes:
        fig = go.Figure()
        fig.update_layout(title="No trade flow data", height=320)
        return fig
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=14, thickness=20,
                  line=dict(color="#374151", width=0.4),
                  label=nodes,
                  color=["#22c55e"] * (len(nodes) - sum(1 for s in sources))
                        + ["#ef4444"] * sum(1 for s in sources)),
        link=dict(source=sources, target=targets, value=values,
                  color="rgba(0,212,255,0.35)"),
    ))
    fig.update_layout(title="Inter-Regional Trade Flows (Implied)", height=420)
    return fig


def futures_curve_chart(curve: pd.DataFrame, structure: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["tenor_month"], y=curve["price"],
                             mode="lines+markers", name=structure,
                             line=dict(color=COLORS["price"], width=2)))
    fig.update_layout(title=f"Futures Curve - {structure}",
                      xaxis_title="Contract Month", yaxis_title="Price", height=320)
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr.values, x=corr.columns, y=corr.index,
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    aspect="auto", text_auto=True)
    fig.update_layout(title="Correlation Matrix", height=340)
    return fig


def scatter_with_fit(x: pd.Series, y: pd.Series, x_label: str, y_label: str) -> go.Figure:
    common = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
    try:
        import statsmodels.api  # noqa: F401  - required for trendline="ols"
        fig = px.scatter(common, x="x", y="y", trendline="ols",
                         labels={"x": x_label, "y": y_label})
    except ImportError:
        fig = px.scatter(common, x="x", y="y",
                         labels={"x": x_label, "y": y_label})
        if len(common) >= 2:
            slope, intercept = np.polyfit(common["x"], common["y"], 1)
            xs = np.linspace(common["x"].min(), common["x"].max(), 50)
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept,
                                     mode="lines", name="OLS fit",
                                     line=dict(color=COLORS["price"], width=1.5)))
    fig.update_layout(title=f"{y_label} vs {x_label}", height=320)
    return fig


def rolling_corr_chart(s: pd.Series, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines",
                             line=dict(color=COLORS["stocks"], width=2),
                             name="rolling corr"))
    fig.add_hline(y=0, line=dict(color="#9ca3af", dash="dot"))
    fig.update_layout(title=f"Rolling Correlation: {label}",
                      yaxis_range=[-1, 1], height=300)
    return fig


def histogram(arr: np.ndarray, title: str, x_label: str = "value") -> go.Figure:
    fig = go.Figure(go.Histogram(x=arr, nbinsx=40,
                                  marker_color=COLORS["stocks"],
                                  marker_line=dict(color="#1f2937", width=0.4)))
    p5, p50, p95 = np.quantile(arr, [0.05, 0.5, 0.95])
    fig.add_vline(x=p5, line=dict(color=COLORS["bear"], dash="dot"), annotation_text="P5")
    fig.add_vline(x=p50, line=dict(color=COLORS["price"], dash="dash"), annotation_text="P50")
    fig.add_vline(x=p95, line=dict(color=COLORS["bull"], dash="dot"), annotation_text="P95")
    fig.update_layout(title=title, xaxis_title=x_label, height=320)
    return fig


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
        line=dict(color="rgba(0,0,0,0)"), name="±10% Fair Band"))
    fig.update_layout(title="Fair Value vs Observed Price", height=380)
    return fig


def cost_curve_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=df["cum_share_pct"], y=df["marginal_cost"],
                            marker_color=COLORS["fair_value"]))
    fig.update_layout(title="Marginal Cost Curve",
                      xaxis_title="Cumulative Supply Share (%)",
                      yaxis_title="Marginal Cost", height=320)
    return fig


# =============================================================================
# UI HELPERS
# =============================================================================

PAGES = [
    "🏠 Dashboard",
    "⚖️ Supply & Demand",
    "🛢️ Inventories",
    "🌪️ Scenarios",
    "🌍 Regional Flows",
    "📈 Futures Curve",
    "🏦 Macro",
    "🎲 Monte Carlo",
    "📉 Sensitivities",
    "⚙️ Settings",
]


# Per-page help text — simple explanations of where data comes from and what the page does.
HELP_TEXT: Dict[str, str] = {
    "🏠 Dashboard": """
**What is this page for?** A market overview at a glance: price, supply/demand
balance, stocks, telemetry and speculative positioning, all on a single screen.

**Where do the numbers come from?**
- **Spot price**: pulled live from Yahoo Finance via the futures contract ticker
  (CL=F for crude, GC=F for gold, etc.). When no internet is available, the
  app falls back to an internal reference value. The quote unit ($/bbl,
  $/oz, ¢/lb…) is shown next to the price.
- **Supply, demand, stocks**: monthly **synthetic** series built from typical
  market parameters (reference production, consumption, seasonality, trend
  growth). A real-world source (EIA, IEA, USDA, JODI…) can replace the
  generator.
- **Regions**: supply/demand split using each major zone's historical share
  (e.g. China ~55 % of global copper demand).
- **Daily telemetry**: simulated daily tracking — vessels in transit,
  refinery utilization, satellite production estimate.
- **Speculative positioning**: Managed Money net positions and a sentiment
  index, CFTC-style, simulated.

**What you'll find here.**
- Key-indicator strip (price, fair value, stocks, days of cover, storage util).
- "Benchmarks vs Current" table comparing each indicator to its ideal norm.
- Combined Supply / Demand / Stocks chart.
- Observed price vs modeled fair value.
- Inventory trajectory and days of cover.
- Regional snapshot and daily telemetry.
""",
    "⚖️ Supply & Demand": """
**What is this page for?** Build and stress-test a supply/demand balance.
Start from a historical series, apply assumptions (growth, supply shock,
weather…) and watch how stocks evolve.

**Where do the numbers come from?**
- The monthly series are **synthetic** by default. You can upload your own
  CSV (minimum columns: `date`, `supply`, `demand` — optional: `imports`,
  `exports`, `stocks`, `price`).
- The adjustments (production, demand, weather, GDP, refining…) come directly
  from the sidebar sliders.

**Computation logic.**
- Accounting identity: *End stocks = Begin stocks + Supply − Demand*.
  Displayable monthly, quarterly or yearly.
- History is frozen; adjustments apply only to forecast months, so the past
  is never rewritten.

**Page sections.**
- Balance key indicators (end stocks, surplus/deficit, days of cover, storage util).
- Combined Supply/Demand/Stocks chart.
- Histogram of monthly builds/draws.
- Seasonality module (monthly profile, heatmap, trend/seasonal/residual decomposition).
- Elasticity curves (supply and demand as functions of price with the
  equilibrium point).
- Lagged supply response to a price shock (e.g. shale delay, planting lag).
- Exportable table (CSV / Excel).
""",
    "🛢️ Inventories": """
**What is this page for?** Test stocks against capacity constraints, simulate
floating storage, and track days of cover.

**Where do the numbers come from?**
- Same series as the Supply & Demand page (balance computed from the sidebar
  assumptions).
- Storage capacity and the floating-storage share are editable at the top of
  the page.

**Logic.**
- Month by month, stocks are projected from builds/draws.
- If projected stocks exceed capacity, the surplus goes into *floating
  storage* (capped at the configured % of capacity).
- Utilization rate = stocks ÷ capacity × 100. Compared to the commodity's
  ideal norm.

**Sections.**
- Key indicators (current stocks, utilization, floating stocks, days of cover).
- Inventory trajectory.
- Last-12-month delta waterfall.
- Utilization gauge.
- Forward days of cover with target line.
- Stacked area: fixed + floating stocks.
""",
    "🌪️ Scenarios": """
**What is this page for?** Quickly compare three scenarios — Bull, Base,
Bear — to frame a likely price range.

**Where do the numbers come from?**
- Each scenario applies a preset bundle of adjustments (supply shock, demand
  shock, GDP growth, weather, USD) to the base balance. Probabilities are
  editable at the top.
- The balance is re-run per scenario, then fair value is recomputed to give
  each scenario its own implied price trajectory.

**Sections.**
- Stocks paths by scenario.
- Price paths by scenario.
- Summary table (probability, end stocks, days of cover, 12m build/draw,
  average price, end fair value).
- Probability-weighted forecast price.
""",
    "🌍 Regional Flows": """
**What is this page for?** See who produces, who consumes, and infer trade
flows between regions.

**Where do the numbers come from?**
- Regional split uses the **historical weights** of each zone defined in the
  commodity template (e.g. China = 55 % of copper demand).
- Supply and demand are simulated per region with a bit of noise to generate
  realistic surpluses/deficits.

**Logic.**
- *Regional balance* = supply − demand. Positive → net exporter.
  Negative → net importer.
- Sankey flows are built by sending each exporter's surplus to importers in
  proportion to their deficit share.

**Sections.**
- Supply vs Demand bar chart by region.
- Implied trade-flow Sankey diagram.
- Arbitrage signals table (Export Arb / Import Need / Neutral).
""",
    "📈 Futures Curve": """
**What is this page for?** Analyze the term structure: contango vs
backwardation, calendar spreads, storage economics.

**Where do the numbers come from?**
- The curve is built from **live futures contracts on Yahoo Finance** when
  available. For each commodity we walk the active delivery months
  (e.g. `CLM25.NYM` for June 2025 WTI, `GCQ25.CMX` for August 2025 gold),
  skip expired contracts, and pull the latest close. Spot is the front-month
  live close.
- When live data isn't reachable (offline, throttled tickers, no contract
  listing for that product), the page falls back to a synthetic curve whose
  shape — contango, backwardation or flat — you control with a selector.

**Logic.**
- *Contango*: forward price > spot (well-supplied market, low scarcity).
- *Backwardation*: forward price < spot (tight market, premium for holding
  physical today).
- *Storage economics*: checks whether the contango premium covers monthly
  storage + financing cost. If yes → "positive carry": it's profitable to
  buy physical and sell forward.

**Sections.**
- Curve chart (with detected regime label and data-source badge).
- Calendar spreads table (m1-m2, m1-m6, m1-m12, m6-m12).
- Storage economics table (contango premium vs carry cost).
- Market reading: tightness score derived from days of cover.
""",
    "🏦 Macro": """
**What is this page for?** See how the big macro aggregates (GDP, PMI, USD,
policy rate) drive the commodity price.

**Where do the numbers come from?**
- The macro panel is **simulated** over 7 years: GDP index (drift random
  walk), PMI (mean-reverting around 50), USD index, policy rate. Can be
  swapped for FRED, OECD series, etc.
- Commodity price comes from the synthetic series of the Supply & Demand page.

**Available analyses.**
- *Correlation matrix* between price and each macro aggregate.
- *Scatter plots* with regression line: pick two variables and confront them
  with price.
- *Rolling correlations* (24 months) to see how the relationship evolves.
- *Multivariate regression*: log(price) explained by all macro factors at
  once. Returns coefficients, intercept and R².
""",
    "🎲 Monte Carlo": """
**What is this page for?** Generate a probabilistic range of prices and
stocks by drawing random shocks (supply, demand, weather, outages).

**Where do the numbers come from?**
- All central assumptions come from the sidebar sliders. Volatilities and the
  outage profile are configured on the page itself.

**How does it work?**
For each simulated path:
1. Draw a supply shock, a demand shock and a weather perturbation from
   normal distributions centered on the base assumptions.
2. With a configurable probability, inject an outage (e.g. hurricane,
   refinery shutdown) that cuts supply.
3. Re-run the balance and recompute fair value.

**Outputs.**
- Distributions of average forecast price, end-stocks level and cumulative
  build/draw. P5/P50/P95 markers.
- Fan charts (uncertainty cone) for price and stocks.
- VaR 95 %: price loss in the worst 5 % of cases.
""",
    "📉 Sensitivities": """
**What is this page for?** Identify *which* assumptions move the target
metric (end stocks, average price, build/draw) the most.

**Where do the numbers come from?**
- Base: current assumptions (sidebar). Each variable is pushed to its low and
  high extreme, all else equal.

**Sections.**
- *Tornado chart*: variables ranked by impact magnitude. The longest bar is
  the most sensitive variable.
- *2D stress matrix*: pick two variables, grid their values and read the
  metric in each cell. Reveals interaction effects.
""",
    "⚙️ Settings": """
**What is this page for?** Save/load assumptions, flush the cache, browse
commodity templates.

**Content.**
- *Export/import JSON* for assumptions.
- *Cache button*: flushes cached data and recomputes everything.
- *Benchmarks matrix*: all ideal parameters (days-of-cover target, ideal
  utilization, typical volatility, normal demand growth, target MM/OI %)
  for all 14 commodities side by side.
- *Detailed template* per commodity (reference supply/demand, storage
  capacity, reference price, quote unit, price band, region weights,
  elasticities, supply lag, seasonality).
""",
}



def render_page_help(page_name: str) -> None:
    """Affiche un encadré expliquant l'origine des données et l'utilité de la page."""
    body = HELP_TEXT.get(page_name)
    if not body:
        return
    with st.expander("ℹ️ About this page — data sources & method", expanded=False):
        st.markdown(body)


def chart_intro(title: str, purpose: str) -> None:
    """Render a small title and an italic light-gray purpose line BEFORE the chart."""
    st.markdown(f"**{title}**")
    st.markdown(
        f"<div class='chart-purpose'>{purpose}</div>",
        unsafe_allow_html=True,
    )


def interpretation(text: str) -> None:
    """Render an info box with a data-driven reading of the section."""
    st.info(f"💡 **Reading** — {text}")


# ---------------------------------------------------------------------------
# Automatic readings — produce a data-driven interpretation string
# ---------------------------------------------------------------------------
def _label_dc(actual: float, target: float) -> str:
    if actual < target * 0.85:
        return "tight"
    if actual > target * 1.15:
        return "loose"
    return "balanced"


def _label_util(actual: float, ideal: float) -> str:
    if actual < ideal - 12:
        return "under-used"
    if actual > ideal + 12:
        return "saturated"
    return "normal"


def _label_fv(deviation_pct: float) -> str:
    if deviation_pct > 10:
        return "overvalued"
    if deviation_pct < -10:
        return "undervalued"
    return "close to fair value"


def read_dashboard(tpl: "CommodityTemplate", bal: pd.DataFrame, fv: pd.DataFrame,
                   spot: float, fv_now: float) -> str:
    last = bal.iloc[-1]
    dc = float(last["days_cover_model"])
    util = float(last["capacity_pct"])
    dev = (spot - fv_now) / fv_now * 100 if fv_now else 0.0
    parts = [
        f"With a spot price at **{fmt_price(spot, tpl.price_unit)}** and a "
        f"modeled fair value of **{fmt_price(fv_now, tpl.price_unit)}**, the "
        f"market is **{_label_fv(dev)}** (deviation {dev:+.1f} %).",
        f"Stocks cover **{dc:.0f} days** of demand (target "
        f"{tpl.days_cover_target:.0f} d) → balance is **{_label_dc(dc, tpl.days_cover_target)}**.",
        f"Storage utilization is **{util:.0f} %** "
        f"(ideal {tpl.ideal_utilization_pct:.0f} %) → regime "
        f"**{_label_util(util, tpl.ideal_utilization_pct)}**.",
    ]
    return " ".join(parts)


def read_balance(bal: pd.DataFrame, tpl: "CommodityTemplate") -> str:
    last = bal.iloc[-1]
    sd = float(last["surplus_deficit"])
    bd12 = float(bal["build_draw"].iloc[-12:].sum())
    avg_demand = float(bal["demand"].mean())
    sd_pct = sd / avg_demand * 100 if avg_demand else 0.0
    sense = "surplus" if sd > 0 else "deficit"
    cumul = "build" if bd12 > 0 else "draw"
    return (
        f"In the last projected month the market is in **{sense}** of "
        f"{abs(sd):,.1f} {tpl.unit} (~{abs(sd_pct):.1f} % of demand). "
        f"Over the trailing 12 months, inventory shows a cumulative **{cumul}** "
        f"of {abs(bd12):,.0f} {tpl.inventory_unit}."
    )


def read_inventory(inv: pd.DataFrame, tpl: "CommodityTemplate") -> str:
    last = inv.iloc[-1]
    util = float(last["utilization_pct"])
    floating = float(last["overflow_floating"])
    msg = (
        f"Warehouse utilization is at **{util:.0f} %** "
        f"(ideal {tpl.ideal_utilization_pct:.0f} %) → "
        f"**{_label_util(util, tpl.ideal_utilization_pct)}**. "
    )
    if floating > 1:
        msg += (f"Floating storage is being used ({floating:,.0f} "
                f"{tpl.inventory_unit}) — a sign that fixed capacity is saturated.")
    else:
        msg += "No floating storage in use: fixed capacity is sufficient."
    return msg


def read_scenarios(results: Dict[str, pd.DataFrame], tpl: "CommodityTemplate") -> str:
    fc = lambda b: b.loc[b["is_forecast"], "price"].mean()
    bull = fc(results["Bull"]); base = fc(results["Base"]); bear = fc(results["Bear"])
    spread_pct = (bull - bear) / base * 100 if base else 0.0
    return (
        f"The bull scenario projects an average price of "
        f"**{fmt_price(bull, tpl.price_unit)}**, the central case "
        f"**{fmt_price(base, tpl.price_unit)}** and the bear "
        f"**{fmt_price(bear, tpl.price_unit)}**. "
        f"The bull–bear range is **{spread_pct:.1f} %** of the central price: "
        f"the wider it is, the more the market is exposed to shocks."
    )


def read_regional(rs: pd.DataFrame) -> str:
    exporters = rs[rs["balance"] > 0]
    importers = rs[rs["balance"] < 0]
    if exporters.empty or importers.empty:
        return "All regions are balanced — no structural flows visible."
    top_exp = exporters.sort_values("balance", ascending=False).iloc[0]
    top_imp = importers.sort_values("balance").iloc[0]
    return (
        f"**{top_exp['region']}** is the dominant exporter ({top_exp['balance']:+,.1f}); "
        f"**{top_imp['region']}** is the largest importer "
        f"({top_imp['balance']:+,.1f}). "
        "Sankey links show who feeds whom, in proportion to deficits."
    )


def read_curve(structure: str, score: float) -> str:
    if score > 0.3:
        market = "tight (low stocks)"
        consistent = "backwardation"
    elif score < -0.3:
        market = "loose (abundant stocks)"
        consistent = "contango"
    else:
        market = "balanced"
        consistent = "flat curve"
    coherent = (
        "consistent" if (consistent in structure.lower()
                         or (consistent == "flat curve" and "flat" in structure.lower()))
        else "inconsistent"
    )
    return (
        f"Stocks → **{market}** market, suggesting a **{consistent}** structure. "
        f"Observed: **{structure}** ({coherent} with the fundamental). "
        "When the curve diverges from the fundamentals, there is often a "
        "convergence trade to play."
    )


def read_macro(corr: pd.DataFrame) -> str:
    if "price" not in corr.columns:
        return "Not enough data to interpret."
    line = corr["price"].drop("price", errors="ignore")
    top = line.abs().sort_values(ascending=False)
    if top.empty:
        return ""
    name = top.index[0]
    val = line[name]
    direction = "positive" if val > 0 else "negative"
    label = {"gdp_index": "GDP", "pmi": "PMI", "usd_index": "USD",
             "policy_rate": "policy rate"}.get(name, name)
    return (
        f"The macro factor most tied to price is **{label}** "
        f"(correlation {val:+.2f}, **{direction}**). "
        "A positive sign means price moves with the indicator; negative means "
        "the opposite. Rolling correlations help check whether the relationship "
        "is stable over time."
    )


def read_monte_carlo(avg_price: np.ndarray, end_stocks: np.ndarray,
                     tpl: "CommodityTemplate") -> str:
    p5, p50, p95 = np.quantile(avg_price, [0.05, 0.5, 0.95])
    width_pct = (p95 - p5) / p50 * 100 if p50 else 0.0
    return (
        f"Expected average price sits around "
        f"**{fmt_price(p50, tpl.price_unit)}** "
        f"(95 % of paths between {fmt_price(p5, tpl.price_unit)} and "
        f"{fmt_price(p95, tpl.price_unit)} — a **{width_pct:.0f} %** width of "
        f"the median). The wider the range, the riskier the market."
    )


def read_tornado(tornado_df: pd.DataFrame) -> str:
    if tornado_df.empty:
        return ""
    top = tornado_df.iloc[-1]
    return (
        f"The most sensitive variable is **{top['variable']}**: a move between "
        f"{top['low_input']:+.1f} and {top['high_input']:+.1f} swings the metric "
        f"by {top['range']:,.1f} units. This is the assumption to watch first."
    )


def init_session_defaults() -> None:
    ss = st.session_state
    ss.setdefault("commodity_key", "crude_oil")
    ss.setdefault("horizon_months", 24)
    ss.setdefault("history_start", "2018-01-01")
    ss.setdefault("seed", 42)
    ss.setdefault("assumptions", BalanceAssumptions(forecast_months=24, gdp_growth_pct=2.5))
    ss.setdefault("page", PAGES[0])


def sidebar_controls() -> None:
    init_session_defaults()
    with st.sidebar:
        st.markdown("### 🛢️ Commodity S&D Desk")

        st.session_state["page"] = st.radio("Navigate", PAGES,
                                            index=PAGES.index(st.session_state["page"]))
        st.divider()

        keys = list(COMMODITY_TEMPLATES.keys())
        st.session_state["commodity_key"] = st.selectbox(
            "Commodity", options=keys,
            format_func=lambda k: COMMODITY_TEMPLATES[k].name,
            index=keys.index(st.session_state["commodity_key"]),
        )
        st.session_state["horizon_months"] = st.slider(
            "Forecast horizon (months)", 6, 36,
            st.session_state["horizon_months"], step=3,
        )
        st.session_state["history_start"] = st.text_input(
            "History start (YYYY-MM-DD)", st.session_state["history_start"]
        )
        st.divider()

        with st.expander("Assumptions", expanded=False):
            a: BalanceAssumptions = st.session_state["assumptions"]
            a.supply_adj_pct = st.slider("Supply Δ %", -10.0, 10.0, a.supply_adj_pct, 0.1)
            a.demand_adj_pct = st.slider("Demand Δ %", -10.0, 10.0, a.demand_adj_pct, 0.1)
            a.weather_pct = st.slider("Weather Δ %", -5.0, 5.0, a.weather_pct, 0.1)
            a.gdp_growth_pct = st.slider("GDP growth %", -2.0, 6.0,
                                         a.gdp_growth_pct or 2.5, 0.1)
            a.imports_adj_pct = st.slider("Imports Δ %", -20.0, 20.0, a.imports_adj_pct, 0.5)
            a.exports_adj_pct = st.slider("Exports Δ %", -20.0, 20.0, a.exports_adj_pct, 0.5)
            if st.session_state["commodity_key"] == "crude_oil":
                a.refinery_runs_pct = st.slider("Refinery runs Δ %", -10.0, 10.0,
                                                a.refinery_runs_pct, 0.1)
            a.forecast_months = st.session_state["horizon_months"]
            st.session_state["assumptions"] = a

        st.divider()
        st.caption("Streamlit · live data (Yahoo) with synthetic fallback")


def kpi_row(items: List[Tuple[str, str, Optional[str]]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col:
            st.metric(label, value, delta=delta)


# =============================================================================
# PAGE RENDERERS
# =============================================================================

def page_dashboard(tpl: CommodityTemplate, df: pd.DataFrame, bal: pd.DataFrame,
                   fv: pd.DataFrame) -> None:
    st.title(f"🏠 {tpl.name} — Dashboard")
    st.caption(f"Sector: **{tpl.sector}** · Futures contract: **{tpl.ticker}** · "
               f"Flow unit: {tpl.unit} · Inventory: {tpl.inventory_unit} · "
               f"Quote: {tpl.price_unit}")
    render_page_help("🏠 Dashboard")

    last_h = bal[~bal["is_forecast"]].iloc[-1]
    last_f = bal.iloc[-1]

    # Spot live (Yahoo) si disponible, sinon spot synthétique de référence
    live = get_live_spot(st.session_state["commodity_key"])
    if live is not None:
        spot = float(live["price"])
        spot_label = f"{fmt_price(spot, tpl.price_unit)}"
        spot_delta = f"{live['change_pct']:+.2f} % (1j) · live au {live['asof']}"
    else:
        spot = float(last_h["price"])
        spot_label = f"{fmt_price(spot, tpl.price_unit)}"
        spot_delta = "internal reference (live unavailable)"

    fv_now = float(fv.loc[fv.index == last_h.name, "fair_value_price"].iloc[0])
    delta_fv_pct = (spot - fv_now) / fv_now * 100.0 if fv_now else 0.0
    yoy_idx = max(-13, -len(bal))
    yoy_pct = (last_h["price"] - bal["price"].iloc[yoy_idx]) / bal["price"].iloc[yoy_idx] * 100

    # --- Bandeau d'indicateurs clés
    chart_intro("Key indicators",
                "Market snapshot at a glance — price, fair value, stocks and "
                "storage utilization, compared to their targets.")
    kpi_row([
        ("Spot price", spot_label, spot_delta),
        ("Fair value", fmt_price(fv_now, tpl.price_unit),
         f"{delta_fv_pct:+.1f} % vs spot"),
        (f"End stocks ({tpl.inventory_unit})",
         f"{last_f['stocks_model']:,.0f}",
         f"YoY prix {yoy_pct:+.1f} %"),
        ("Days of cover", f"{last_f['days_cover_model']:.1f}",
         f"target {tpl.days_cover_target:.0f}"),
        ("Storage util", f"{last_f['capacity_pct']:.1f} %",
         f"ideal {tpl.ideal_utilization_pct:.0f} %"),
    ])

    interpretation(read_dashboard(tpl, bal, fv, spot, fv_now))

    # --- Benchmarks
    chart_intro("📐 Ideal benchmarks vs current reading",
                "For each indicator we show the 'normal' value for this product "
                "and the current value. The *Reading* column tells whether "
                "you're in range, above, or below.")
    yoy_demand = (bal["demand"].iloc[-1] / bal["demand"].iloc[max(-13, -len(bal))] - 1) * 100
    rolling_vol = bal["price"].pct_change().tail(12).std() * 100
    bench_rows = [
        {"Metric": "Jours de couverture", "Ideal": f"{tpl.days_cover_target:.0f} d",
         "Current": f"{last_f['days_cover_model']:.1f} d",
         "Reading": "Tight" if last_f["days_cover_model"] < tpl.days_cover_target * 0.85
                    else "Loose" if last_f["days_cover_model"] > tpl.days_cover_target * 1.15
                    else "Balanced"},
        {"Metric": "Utilisation stockage", "Ideal": f"{tpl.ideal_utilization_pct:.0f} %",
         "Current": f"{last_f['capacity_pct']:.1f} %",
         "Reading": "Under-used" if last_f["capacity_pct"] < tpl.ideal_utilization_pct - 10
                    else "Saturated" if last_f["capacity_pct"] > tpl.ideal_utilization_pct + 10
                    else "Healthy"},
        {"Metric": "Monthly vol (12m)",
         "Ideal": f"{tpl.typical_monthly_vol_pct:.1f} %",
         "Current": f"{rolling_vol:.1f} %",
         "Reading": "Elevated" if rolling_vol > tpl.typical_monthly_vol_pct * 1.3
                    else "Subdued" if rolling_vol < tpl.typical_monthly_vol_pct * 0.7
                    else "Normal"},
        {"Metric": "YoY demand growth",
         "Ideal": f"{tpl.normal_yoy_demand_pct:+.1f} %",
         "Current": f"{yoy_demand:+.1f} %",
         "Reading": "Above trend" if yoy_demand > tpl.normal_yoy_demand_pct + 1
                    else "Below trend" if yoy_demand < tpl.normal_yoy_demand_pct - 1
                    else "On trend"},
    ]
    st.dataframe(pd.DataFrame(bench_rows), hide_index=True, use_container_width=True)

    st.markdown("---")

    # --- Bloc graphiques principaux
    left, right = st.columns([3, 2])
    with left:
        chart_intro("Supply, demand & stocks",
                    "Monthly evolution of supply and demand (left axis) plus "
                    "the inventory level (right axis). The vertical line "
                    "marks the start of the forecast horizon.")
        st.plotly_chart(supply_demand_chart(bal, unit=tpl.unit), use_container_width=True)

        chart_intro("Observed price vs fair value",
                    "Fair value is estimated from a stocks↔price regression. "
                    "The violet band is the ±10 % range considered 'fairly "
                    "valued'.")
        st.plotly_chart(fair_value_chart(fv), use_container_width=True)
    with right:
        chart_intro("Inventory trajectory",
                    "Stocks projected month by month through the forecast horizon.")
        st.plotly_chart(inventory_chart(bal, unit=tpl.inventory_unit),
                        use_container_width=True)

        chart_intro("Forward days of cover",
                    "How many days of demand stocks would cover. The dashed "
                    "line is the normal target for this product.")
        st.plotly_chart(days_cover_chart(bal, target=tpl.days_cover_target),
                        use_container_width=True)

    # --- Régional
    st.markdown("### Regional snapshot")
    reg = get_regional_dataset(st.session_state["commodity_key"])
    rs = regional_summary(reg)
    c1, c2 = st.columns([3, 2])
    with c1:
        chart_intro("Supply vs demand by region",
                    "Grouped bars. A green bar larger than red = net exporter "
                    "region ; the opposite = net importer.")
        st.plotly_chart(regional_bar(reg), use_container_width=True)
    with c2:
        chart_intro("Regional detail",
                    "Balance = Supply − Demand. *Exporter* / *Importer* per sign.")
        st.dataframe(rs[["region", "supply", "demand", "balance", "status"]].round(2),
                     use_container_width=True, hide_index=True)
    interpretation(read_regional(rs))

    # --- Télémétrie
    st.markdown("### Daily telemetry")
    chart_intro("High-frequency indicators",
                "Three proxy indicators tracking real-world activity day by day. "
                "The 7-day change shows the recent trend.")
    hf = get_high_frequency(st.session_state["commodity_key"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Vessels tracked", int(hf["vessels_tracked"].iloc[-1]),
              f"{hf['vessels_tracked'].iloc[-1] - hf['vessels_tracked'].iloc[-8]:+d} on 7d")
    c2.metric("Refinery util %", f"{hf['refinery_util_pct'].iloc[-1]:.1f}",
              f"{hf['refinery_util_pct'].iloc[-1] - hf['refinery_util_pct'].iloc[-8]:+.2f}")
    c3.metric("Sat production est", f"{hf['sat_production_est'].iloc[-1]:,.1f}",
              f"{(hf['sat_production_est'].iloc[-1] / hf['sat_production_est'].iloc[-8] - 1) * 100:+.2f} %")
    interpretation(
        "Rising refinery utilization or more vessels in transit signal **robust "
        "demand**. A sustained decline often precedes a loosening market."
    )

    # --- Positionnement
    st.markdown("### Speculative positioning")
    chart_intro("Hedge funds & sentiment",
                "Managed Money net position (longs − shorts) plus a global "
                "sentiment index. A *soft* indicator: can stay disconnected "
                "from fundamentals for long stretches.")
    pos = get_positioning(st.session_state["commodity_key"])
    c1, c2 = st.columns(2)
    mm_now = float(pos['managed_money_net'].iloc[-1])
    sent_now = float(pos['sentiment_score'].iloc[-1])
    c1.metric("Managed Money net", f"{mm_now:,.0f}")
    c2.metric("Sentiment score",
              f"{sent_now:.0f}/100 · {sentiment_label(sent_now)}")
    interpretation(
        f"Sentiment is at **{sent_now:.0f}/100** ({sentiment_label(sent_now)}). "
        f"Extreme sentiment (>80 or <20) is often a contrarian signal — the "
        f"consensus is already priced and the next surprise tends to go the "
        f"other way."
    )


def page_supply_demand(tpl: CommodityTemplate, df: pd.DataFrame) -> None:
    st.title(f"⚖️ {tpl.name} — Supply/Demand Balance")
    render_page_help("⚖️ Supply & Demand")

    with st.expander("Upload a custom CSV (columns: date, supply, demand…)"):
        file = st.file_uploader("CSV file", type=["csv"])
        if file is not None:
            df = load_csv(file)

    chart_intro("Aggregation frequency",
                "Monthly (default), quarterly or yearly. Coarser cadence "
                "smooths out seasonality.")
    freq = st.radio("Frequency", ["Monthly", "Quarterly", "Yearly"], horizontal=True)
    freq_map = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
    bal = run_balance(df, st.session_state["commodity_key"],
                      st.session_state["assumptions"], frequency=freq_map[freq])

    last = bal.iloc[-1]
    chart_intro("Balance key indicators",
                "State of the market at the end of the projected period.")
    kpi_row([
        (f"End stocks ({tpl.inventory_unit})", f"{last['stocks_model']:,.0f}", None),
        ("Surplus / Deficit", f"{last['surplus_deficit']:+,.1f}", None),
        ("Days of cover", f"{last['days_cover_model']:.1f}",
         f"target {tpl.days_cover_target:.0f}"),
        ("Storage util", f"{last['capacity_pct']:.1f} %",
         f"ideal {tpl.ideal_utilization_pct:.0f} %"),
    ])
    interpretation(read_balance(bal, tpl))

    chart_intro("Offre, demande & stocks",
                "L'écart entre les deux lignes pleines (offre vs demande) "
                "explique la pente des stocks (ligne pointillée).")
    st.plotly_chart(supply_demand_chart(bal, unit=tpl.unit), use_container_width=True)

    chart_intro("Monthly builds & draws",
                "Green bars = inventory rising (build), red = falling (draw). "
                "Last 24 months.")
    st.plotly_chart(balance_bars(bal), use_container_width=True)

    st.markdown("### Seasonality")
    chart_intro("Average seasonal profile",
                "For each month, displays the 5-year historical mean, the "
                "min-max range, and the current year as a dotted line.")
    profile = monthly_profile(df["demand"])
    piv = year_over_year_pivot(df["demand"])
    c1, c2 = st.columns([2, 3])
    with c1:
        st.plotly_chart(seasonal_lines(profile), use_container_width=True)
    with c2:
        chart_intro("Seasonal heatmap",
                    "Year × month heatmap for spotting structural anomalies.")
        st.plotly_chart(seasonal_heatmap(piv, "Demand heatmap (Year × Month)"),
                        use_container_width=True)

    # Lecture saisonnalité
    peak_month = int(profile["mean"].idxmax())
    trough_month = int(profile["mean"].idxmin())
    months_en = ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]
    interpretation(
        f"The seasonal demand peak is in **{months_en[peak_month - 1]}** "
        f"({profile.loc[peak_month, 'mean']:,.1f}), the trough in "
        f"**{months_en[trough_month - 1]}** ({profile.loc[trough_month, 'mean']:,.1f}). "
        "Knowing this cycle helps anticipate structural builds/draws and "
        "avoid confusing a seasonal move with a real fundamental signal."
    )

    with st.expander("Seasonal decomposition (trend / seasonal / residual)"):
        st.markdown("We split the series into 3 additive components: **trend** (long term), "
                    "**seasonal** (annual cycle) and **residual** (the rest, which should "
                    "look like noise if the model is good).")
        trend, seasonal, resid = decompose(df["demand"], period=12)
        st.markdown("**Long-term trend**")
        st.line_chart(trend.dropna(), height=180)
        st.markdown("**Seasonal component**")
        st.line_chart(seasonal.dropna(), height=180)
        st.markdown("**Residual**")
        st.line_chart(resid.dropna(), height=180)
        st.markdown("**12-month moving average (smoothing)**")
        st.line_chart(rolling_seasonal_average(df["demand"]).dropna(), height=180)

    st.markdown("### Price elasticity")
    chart_intro("Supply and demand curves",
                "The larger α (alpha), the more demand falls when price rises. "
                "The larger β (beta), the more strongly production responds to price. "
                "The intersection is the market equilibrium.")
    c1, c2 = st.columns(2)
    alpha = c1.slider("Demand elasticity α", 0.0, 0.5, tpl.elasticity_alpha, 0.005)
    beta = c2.slider("Supply elasticity β", 0.0, 0.5, tpl.elasticity_beta, 0.005)
    curves = build_curves(st.session_state["commodity_key"], alpha=alpha, beta=beta)
    p = ElasticityParams(alpha=alpha, beta=beta, base_price=tpl.base_price,
                         d0=tpl.base_demand, s0=tpl.base_supply)
    eq_p, eq_q = equilibrium(p)
    st.plotly_chart(elasticity_chart(curves, eq_p, eq_q), use_container_width=True)
    interpretation(
        f"Equilibrium price is **{fmt_price(eq_p, tpl.price_unit)}** for a quantity "
        f"of **{eq_q:,.2f} {tpl.unit}**. "
        f"When α and β are small, demand and supply are **rigid**: large price "
        "moves are needed to rebalance the market. "
        "When α and β are high, the market is **flexible** and absorbs shocks "
        "with less volatility."
    )

    st.markdown("### Lagged supply response")
    chart_intro("Effect of a price shock on future production",
                "Simulates how long production takes to react to a price move. "
                "Shale is ~6 months ; copper takes years (mining capex).")
    c1, c2 = st.columns(2)
    lag = c1.slider("Reaction lag (months)", 1, 18, tpl.supply_lag_months)
    shock = c2.slider("Price shock %", -50.0, 50.0, 20.0, 5.0)
    try:
        result = fit_lagged_supply(df, lag_months=lag)
        st.caption(f"Fit quality (R²) on history: {result.r_squared:.3f}")
    except ValueError as exc:
        st.warning(str(exc))
    proj = project_lagged_response(tpl.base_supply, shock, lag)
    st.line_chart(proj, height=220)
    interpretation(
        f"With a {lag}-month lag and a {shock:+.0f} % price shock, production "
        f"converges gradually to a new level. The S-curve shows reaction is "
        "zero until the lag has elapsed, then accelerates, then stabilizes."
    )

    st.markdown("### Balance table & export")
    chart_intro("Month-by-month detail",
                "The last 36 rows of the full balance, exportable.")
    st.dataframe(bal.tail(36).round(2), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.download_button("Download balance (CSV)", df_to_csv_bytes(bal),
                       file_name=f"{st.session_state['commodity_key']}_balance.csv",
                       mime="text/csv")
    c2.download_button("Download Excel workbook",
                       df_to_excel_bytes({"balance": bal, "seasonal_profile": profile}),
                       file_name=f"{st.session_state['commodity_key']}_workbook.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def page_inventories(tpl: CommodityTemplate, df: pd.DataFrame, bal: pd.DataFrame) -> None:
    st.title(f"🛢️ {tpl.name} — Inventory & Storage")
    render_page_help("🛢️ Inventories")

    chart_intro("Storage parameters",
                "Edit the storage capacity and the share allowed in floating "
                "storage (vessels, rail). If stocks exceed fixed capacity, "
                "the surplus is absorbed up to this limit.")
    c1, c2, c3 = st.columns(3)
    cap = c1.number_input(f"Capacity ({tpl.inventory_unit})",
                          value=float(tpl.storage_capacity),
                          step=10.0, min_value=10.0)
    floating = c2.slider("Floating storage (% of capacity)", 0.0, 25.0, 5.0, 0.5)
    allow_neg = c3.checkbox("Allow negative stocks (debug)", value=False)
    inv = project_inventory(bal, st.session_state["commodity_key"],
                            StorageConfig(capacity=cap, floating_buffer_pct=floating,
                                          allow_negative=allow_neg))

    last = inv.iloc[-1]
    chart_intro("Inventory key indicators",
                "Reading at the end of the projected period.")
    kpi_row([
        (f"Current stocks ({tpl.inventory_unit})",
         f"{last['stocks_capped']:,.0f}", None),
        ("Storage util", f"{last['utilization_pct']:.1f} %",
         f"ideal {tpl.ideal_utilization_pct:.0f} %"),
        ("Floating stocks", f"{last['overflow_floating']:,.0f}", None),
        ("Days of cover", f"{last['days_cover_model']:.1f}",
         f"target {tpl.days_cover_target:.0f}"),
    ])
    interpretation(read_inventory(inv, tpl))

    chart_intro("Trajectoire des stocks",
                "Stocks projetés mois par mois, plafonnés par la capacité fixe.")
    st.plotly_chart(inventory_chart(inv, unit=tpl.inventory_unit),
                    use_container_width=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        chart_intro("Builds/draws waterfall (12 months)",
                    "For each recent month, shows the inventory delta. A run "
                    "of green bars = sustained accumulation.")
        st.plotly_chart(waterfall_chart(draw_build_waterfall(inv, n=12)),
                        use_container_width=True)
    with c2:
        chart_intro("Utilisation gauge",
                    "Quick visual of pressure on storage capacity. Above 80 %, "
                    "the buffer shrinks.")
        st.plotly_chart(utilization_gauge(last["utilization_pct"]),
                        use_container_width=True)

    chart_intro("Forward days of cover",
                "How many days of demand stocks would last with zero new "
                "production. The dashed line is the normal target.")
    st.plotly_chart(days_cover_chart(inv, target=tpl.days_cover_target),
                    use_container_width=True)

    chart_intro("Fixed vs floating stocks",
                "Stacked area. When the light-blue layer (floating) appears, "
                "fixed capacity has been exceeded.")
    st.area_chart(inv[["stocks_capped", "overflow_floating"]].tail(48), height=260)

    chart_intro("Monthly detail",
                "Summary table of the last 24 months.")
    st.dataframe(
        inv[["stocks_model", "stocks_capped", "overflow_floating",
             "utilization_pct", "days_cover_model"]].tail(24).round(2),
        use_container_width=True,
    )


def page_scenarios(tpl: CommodityTemplate, df: pd.DataFrame) -> None:
    st.title(f"🌪️ {tpl.name} — Scenario Engine")
    render_page_help("🌪️ Scenarios")

    chart_intro("Scenario probabilities",
                "Adjust the probability assigned to each scenario. The system "
                "renormalizes them to 100 % automatically.")
    c1, c2, c3 = st.columns(3)
    new_probs = {}
    labels = {"Bull": "Bull (upside)", "Base": "Base (central)", "Bear": "Bear (downside)"}
    for col, name in zip([c1, c2, c3], ["Bull", "Base", "Bear"]):
        new_probs[name] = col.number_input(
            f"Probability — {labels[name]}", min_value=0.0, max_value=1.0,
            value=float(SCENARIO_PRESETS[name]["probability"]), step=0.05,
        )
    total = sum(new_probs.values()) or 1.0
    for n in new_probs:
        SCENARIO_PRESETS[n]["probability"] = new_probs[n] / total

    results = run_scenarios(df, st.session_state["commodity_key"],
                            st.session_state["assumptions"])

    chart_intro("Inventory paths by scenario",
                "How stocks evolve in each scenario. Bear = stocks rising "
                "(surplus) ; Bull = stocks falling (tightness).")
    st.plotly_chart(scenario_paths(results, "stocks_model", "Inventory trajectories"),
                    use_container_width=True)

    chart_intro("Price paths by scenario",
                "Fair-value price implied by each scenario's stocks level.")
    st.plotly_chart(scenario_paths(results, "fair_value_price",
                                    "Fair value trajectories"),
                    use_container_width=True)
    interpretation(read_scenarios(results, tpl))

    chart_intro("Summary table",
                "Per scenario: probability, end stocks, days of cover, "
                "12-month build/draw, average price, end fair value.")
    summary = scenario_summary(results, st.session_state["commodity_key"])
    st.dataframe(summary.round(2), use_container_width=True)
    pw = probability_weighted_price(results)
    st.metric("Probability-weighted forecast price",
              fmt_price(pw, tpl.price_unit))
    interpretation(
        f"The probability-weighted expected price is **{fmt_price(pw, tpl.price_unit)}**. "
        "This serves as an anchor for budgeting or hedging. If the current "
        "market price diverges sharply from this level, it's either an "
        "opportunity or a sign our assumptions need recalibrating."
    )

    with st.expander("Per-scenario assumption details"):
        st.json(SCENARIO_PRESETS)


def page_regional(tpl: CommodityTemplate) -> None:
    st.title(f"🌍 {tpl.name} — Regional Flows")
    render_page_help("🌍 Regional Flows")
    reg = get_regional_dataset(st.session_state["commodity_key"])
    rs = regional_summary(reg)

    c1, c2 = st.columns([3, 2])
    with c1:
        chart_intro("Supply vs demand by region",
                    "Grouped bars. Direct comparison of production capacity "
                    "and consumption for each major zone.")
        st.plotly_chart(regional_bar(reg), use_container_width=True)
    with c2:
        chart_intro("Regional table",
                    "Balance = Supply − Demand. Positive → net exporter.")
        st.dataframe(rs.round(2), hide_index=True, use_container_width=True)
    interpretation(read_regional(rs))

    chart_intro("Implied trade flows (Sankey)",
                "Each net exporter sends its surplus to net importers, in "
                "proportion to each importer's deficit. Link width represents "
                "flow volume. Visualizes trade dependency.")
    nodes, sources, targets, values = build_trade_flows(reg)
    st.plotly_chart(sankey_chart(nodes, sources, targets, values),
                    use_container_width=True)

    chart_intro("Arbitrage signals",
                "Classifies each region: *Export Arb* = surplus to ship out ; "
                "*Import Need* = urgent cargo need. The more extreme the "
                "balance, the wider the regional price differential.")
    arb = arbitrage_signals(reg)
    st.dataframe(arb[["region", "balance", "arb_signal"]].round(2),
                 hide_index=True, use_container_width=True)
    interpretation(
        "When a region flips to *Import Need*, local prices rise until enough "
        "cargos are attracted. This mechanism explains the Brent-Dubai premium, "
        "the TTF-Henry Hub spread, or the Chicago-Heartland corn basis."
    )


def page_futures_curve(tpl: CommodityTemplate, bal: pd.DataFrame) -> None:
    st.title(f"📈 {tpl.name} — Term Structure")
    render_page_help("📈 Futures Curve")

    # Try a live curve from Yahoo first (cfcap-style contract walking).
    ck = st.session_state["commodity_key"]
    live_curve = get_live_futures_curve(ck, n_max=tpl.liquid_months or 12)
    is_live = live_curve is not None and len(live_curve) >= 2

    chart_intro("Curve source & parameters",
                "When live futures contracts can be fetched from Yahoo Finance "
                "(WTI, Henry Hub, RBOB, Gold, Silver, Copper, Wheat, Corn, "
                "Soybeans, Coffee, Sugar…), the curve is **real**. Otherwise "
                "we fall back to a synthetic shape that you choose below — "
                "useful for stress testing and for LME/SGX products that "
                "Yahoo doesn't list.")
    c1, c2, c3 = st.columns(3)
    mode_options = ["Live (Yahoo)" if is_live else "Synthetic (Yahoo unavailable)",
                    "Synthetic — contango",
                    "Synthetic — backwardation",
                    "Synthetic — flat"]
    mode = c1.selectbox("Curve source", mode_options,
                         index=0 if is_live else 1)
    months = c2.slider("Tenors (months)", 6, 36, 18)
    storage_cost = c3.number_input(f"Monthly storage cost ({tpl.price_unit})",
                                    value=0.20, step=0.05)
    financing_rate = st.slider("Financing rate %", 0.0, 12.0, 5.0, 0.25)

    if mode.startswith("Live") and is_live:
        curve = live_curve.copy()
        source_badge = (f"🟢 **Live data** — {len(curve)} contracts pulled from "
                        f"Yahoo Finance (pattern `{tpl.yf_fmt}`, expired "
                        f"contracts skipped).")
    else:
        synth_shape = "contango"
        if "backwardation" in mode:
            synth_shape = "backwardation"
        elif "flat" in mode:
            synth_shape = "flat"
        curve = get_synthetic_futures_curve(ck, structure=synth_shape, months=months)
        reason = ("forced synthetic" if mode.startswith("Synthetic —")
                  else "Yahoo unreachable / no listed contracts")
        source_badge = (f"🟡 **Synthetic curve** ({synth_shape}) — {reason}. "
                        f"Real LME (aluminum, nickel) and SGX (iron ore) feeds "
                        f"are not on Yahoo; a paid data source would be needed.")
    st.caption(source_badge)
    struct_label = classify_structure(curve)

    chart_intro("Term structure",
                "Price per maturity. **Contango** = upward-sloping (forwards "
                "trade above spot — well-supplied market). "
                "**Backwardation** = downward-sloping (forwards below spot — "
                "premium for holding physical today).")
    st.plotly_chart(futures_curve_chart(curve, struct_label), use_container_width=True)

    c1, c2 = st.columns([2, 3])
    with c1:
        chart_intro("Calendar spreads",
                    "Difference between the near maturity and farther "
                    "maturities. A positive near-far spread = contango.")
        st.dataframe(calendar_spreads(curve).round(3), use_container_width=True)
    with c2:
        chart_intro("Storage economics",
                    "For each maturity, compares contango premium to carry "
                    "cost (storage + financing). *positive_carry* = storing "
                    "physical and selling forward is profitable.")
        econ = storage_economics(curve, storage_cost, financing_rate)
        st.dataframe(econ[["tenor_month", "price", "contango_premium",
                           "carry", "positive_carry"]].round(3),
                     hide_index=True, use_container_width=True)
    pos_carry_pct = float(econ["positive_carry"].mean()) * 100
    interpretation(
        f"{pos_carry_pct:.0f} % of maturities offer **positive carry**: at these "
        "tenors, a trader can buy physical today, store, and sell forward for "
        "a mechanical gain. If carry is positive everywhere, the market is "
        "oversupplied."
    )

    st.subheader("Stocks ↔ curve coherence")
    chart_intro("Market reading",
                "Compares the observed curve regime to the regime theoretically "
                "induced by the stocks' days of cover.")
    last_dc = float(bal["days_cover_model"].iloc[-1])
    label, score = inventory_curve_relationship(curve, last_dc)
    kpi_row([
        ("Days of cover", f"{last_dc:.1f}", None),
        ("Tightness score", f"{score:+.2f}",
         "+1 very tight, −1 very loose"),
        ("Diagnosis", label, None),
    ])
    interpretation(read_curve(label, score))

    with st.expander("Raw curve data"):
        st.dataframe(curve.round(3), hide_index=True, use_container_width=True)


def page_macro(tpl: CommodityTemplate, df: pd.DataFrame) -> None:
    st.title(f"🏦 {tpl.name} — Macro Overlay")
    render_page_help("🏦 Macro")
    macro = get_macro_panel(months=84)
    joined = align_macro(df, macro)

    cols = ["price", "gdp_index", "pmi", "usd_index", "policy_rate"]
    corr = correlation_matrix(joined, cols)

    chart_intro("Correlation matrix",
                "Linear relationship between price and each macro aggregate. "
                "Reading: green = positive correlation, red = negative. The "
                "closer to 1 in absolute value, the stronger the link.")
    st.plotly_chart(correlation_heatmap(corr), use_container_width=True)
    interpretation(read_macro(corr))

    chart_intro("Cross-diagnostic: price vs macro factor",
                "Pick two macro variables to confront with price. The "
                "regression line shows the average trend. Wide scatter = "
                "weak relationship ; tight alignment = robust relationship.")
    choices = ["gdp_index", "pmi", "usd_index", "policy_rate"]
    label_map = {"gdp_index": "GDP index", "pmi": "PMI",
                 "usd_index": "USD index", "policy_rate": "Policy rate"}
    c1, c2 = st.columns(2)
    with c1:
        x1 = st.selectbox("Macro variable A", choices, index=0,
                          format_func=lambda x: label_map[x], key="ma")
        st.plotly_chart(scatter_with_fit(joined[x1], joined["price"],
                                          label_map[x1], f"Price ({tpl.price_unit})"),
                        use_container_width=True)
    with c2:
        x2 = st.selectbox("Macro variable B", choices, index=2,
                          format_func=lambda x: label_map[x], key="mb")
        st.plotly_chart(scatter_with_fit(joined[x2], joined["price"],
                                          label_map[x2], f"Price ({tpl.price_unit})"),
                        use_container_width=True)

    st.subheader("Rolling correlations vs price")
    chart_intro("Stability of the relationship over time",
                "Correlation on a rolling 24-month window. If the curve "
                "oscillates around zero, the relationship is unstable ; if it "
                "stays away from zero, the link is durable.")
    c1, c2 = st.columns(2)
    rc1 = rolling_correlation(joined["price"], joined["gdp_index"])
    c1.plotly_chart(rolling_corr_chart(rc1, "Price vs GDP"), use_container_width=True)
    rc2 = rolling_correlation(joined["price"], joined["usd_index"])
    c2.plotly_chart(rolling_corr_chart(rc2, "Price vs USD"), use_container_width=True)
    last_usd_corr = float(rc2.dropna().iloc[-1]) if not rc2.dropna().empty else 0
    last_gdp_corr = float(rc1.dropna().iloc[-1]) if not rc1.dropna().empty else 0
    interpretation(
        f"Recent 24m correlation: Price–GDP = {last_gdp_corr:+.2f}, Price–USD = "
        f"{last_usd_corr:+.2f}. For most commodities, we expect a negative "
        "correlation with USD (dollar-denominated prices) and a positive one "
        "with GDP (demand)."
    )

    st.subheader("Multivariate regression: log(price) ~ macro")
    chart_intro("Multi-factor linear model",
                "Tries to explain log-price using **all** macro factors "
                "simultaneously. R² indicates the variance share captured by "
                "the model (the closer to 1, the better).")
    joined["log_price"] = np.log(joined["price"])
    try:
        res = regression_summary(joined, "log_price",
                                 ["gdp_index", "pmi", "usd_index", "policy_rate"])
        st.json(res)
        interpretation(
            f"The model explains **{res['r_squared'] * 100:.0f} %** of price "
            "variance. The higher, the better macro works as a shortcut for "
            "understanding price. Above 50 %, macro indicators can reasonably "
            "be used to anticipate moves."
        )
    except ValueError as exc:
        st.warning(str(exc))


def page_monte_carlo(tpl: CommodityTemplate, df: pd.DataFrame) -> None:
    st.title(f"🎲 {tpl.name} — Monte Carlo Engine")
    render_page_help("🎲 Monte Carlo")

    chart_intro("Random shock parameters",
                "Volatilities (σ) for supply, demand and weather shocks plus "
                "outage frequency/size. The larger σ, the more dispersed the "
                "trajectories.")
    c1, c2, c3 = st.columns(3)
    n_paths = c1.slider("Number of paths", 100, 2000, 500, step=100)
    sigma_supply = c2.slider("Supply shock σ %", 0.5, 5.0, 1.5, 0.1)
    sigma_demand = c3.slider("Demand shock σ %", 0.5, 5.0, 1.2, 0.1)

    c1, c2, c3 = st.columns(3)
    sigma_weather = c1.slider("Weather shock σ %", 0.0, 3.0, 1.0, 0.1)
    outage_prob = c2.slider("Outage probability (per month)", 0.0, 0.20, 0.05, 0.01)
    outage_size = c3.slider("Outage size %", 0.0, 15.0, 4.0, 0.5)

    cfg = MCConfig(n_paths=n_paths, supply_sigma_pct=sigma_supply,
                   demand_sigma_pct=sigma_demand, weather_sigma_pct=sigma_weather,
                   outage_prob=outage_prob, outage_size_pct=outage_size)

    if st.button("Run Monte Carlo simulation", type="primary"):
        with st.spinner(f"Simulating {n_paths} paths…"):
            out = run_monte_carlo(df, st.session_state["commodity_key"],
                                  st.session_state["assumptions"], cfg)
        end_stocks = out["end_stocks"]
        avg_price = out["avg_price"]
        bd = out["build_draw"]
        base_price = float(np.median(avg_price))
        losses = np.maximum(0, base_price - avg_price)
        var95 = value_at_risk(losses, 0.95)

        chart_intro("Probabilistic indicators",
                    "Synthesis of simulated paths: median, P5-P95 range and "
                    "Value at Risk 95 % (price drop in the worst 5 % of cases).")
        kpi_row([
            ("Median price", fmt_price(float(np.median(avg_price)), tpl.price_unit), None),
            ("P5 – P95 range",
             f"{fmt_price(float(np.quantile(avg_price, 0.05)), tpl.price_unit)} – "
             f"{fmt_price(float(np.quantile(avg_price, 0.95)), tpl.price_unit)}", None),
            (f"Median end stocks ({tpl.inventory_unit})",
             f"{np.median(end_stocks):,.0f}", None),
            (f"VaR 95 % (price drop)",
             fmt_price(var95, tpl.price_unit), None),
        ])
        interpretation(read_monte_carlo(avg_price, end_stocks, tpl))

        c1, c2 = st.columns(2)
        with c1:
            chart_intro("Forecast average price distribution",
                        "Histogram of average prices across N paths. Vertical "
                        "bars at P5/P50/P95 mark the thresholds.")
            st.plotly_chart(histogram(avg_price, "Average price distribution",
                                       x_label=f"Price ({tpl.price_unit})"),
                            use_container_width=True)
        with c2:
            chart_intro("End-of-horizon stocks distribution",
                        "The narrower the distribution, the more robust the "
                        "forecast against shocks.")
            st.plotly_chart(histogram(end_stocks, "End stocks distribution",
                                       x_label=f"Stocks ({tpl.inventory_unit})"),
                            use_container_width=True)
        chart_intro("Cumulative build/draw distribution",
                    "Total stocks variation over the forecast period across "
                    "all paths.")
        st.plotly_chart(histogram(bd, "Cumulative build/draw distribution",
                                   x_label="Δ Stocks"), use_container_width=True)

        st.subheader("Probabilistic fan charts")
        chart_intro("Uncertainty cone",
                    "The shaded band is the P5–P95 interval month by month. "
                    "The center line is the median. The wider the cone, the "
                    "more uncertainty grows with horizon.")
        pct = out["percentiles"]
        c1, c2 = st.columns(2)
        c1.plotly_chart(fan_chart(pct, "price"), use_container_width=True)
        c2.plotly_chart(fan_chart(pct, "stocks"), use_container_width=True)
    else:
        st.info("Set the parameters then click **Run simulation** to start.")


def page_sensitivities(tpl: CommodityTemplate, df: pd.DataFrame) -> None:
    st.title(f"📉 {tpl.name} — Sensitivity Analysis")
    render_page_help("📉 Sensitivities")

    chart_intro("Choice of target metric",
                "Which value to test? End stocks, average forecast price, "
                "or 12-month cumulative build/draw.")
    metric_map = {
        "end_stocks": "End-of-period stocks",
        "avg_fc_price": "Average forecast price",
        "build_draw_sum": "12-month cumulative build/draw",
    }
    metric = st.selectbox("Metric",
                          list(metric_map.keys()),
                          format_func=lambda k: metric_map[k], index=0)

    variables = [
        SensitivityVar("Supply Δ %", "supply_adj_pct", -3.0, 3.0),
        SensitivityVar("Demand Δ %", "demand_adj_pct", -3.0, 3.0),
        SensitivityVar("Weather Δ %", "weather_pct", -2.0, 2.0),
        SensitivityVar("GDP %", "gdp_growth_pct", 0.5, 4.0),
        SensitivityVar("Imports Δ %", "imports_adj_pct", -10.0, 10.0),
        SensitivityVar("Exports Δ %", "exports_adj_pct", -10.0, 10.0),
    ]
    torn = tornado(df, st.session_state["commodity_key"],
                   st.session_state["assumptions"], variables, metric=metric)

    chart_intro("Tornado chart",
                "For each variable, push it to its low and high values and "
                "measure the impact on the metric. The longest bars are the "
                "top levers to watch.")
    st.plotly_chart(tornado_chart(torn), use_container_width=True)
    st.dataframe(torn.round(2), hide_index=True, use_container_width=True)
    interpretation(read_tornado(torn))

    st.subheader("2D stress matrix")
    chart_intro("Interaction effects between two variables",
                "Heatmap where two assumptions are gridded and the metric is "
                "read in each cell. Useful for spotting extreme zones (e.g. "
                "a simultaneous supply + demand shock).")
    c1, c2 = st.columns(2)
    labels = [v.name for v in variables]
    a = c1.selectbox("Variable A", labels, index=0)
    b = c2.selectbox("Variable B", labels, index=1)
    va = next(v for v in variables if v.name == a)
    vb = next(v for v in variables if v.name == b)
    mat = stress_matrix(df, st.session_state["commodity_key"],
                        st.session_state["assumptions"], va, vb, grid=6, metric=metric)
    st.plotly_chart(seasonal_heatmap(mat,
                                      title=f"{metric_map[metric]} — {a} × {b}"),
                    use_container_width=True)
    interpretation(
        "The reddest/bluest corner cells indicate the most dangerous stress "
        "combinations. That's where to build extreme risk scenarios."
    )


def page_settings() -> None:
    st.title("⚙️ Settings")
    render_page_help("⚙️ Settings")

    chart_intro("Active assumptions",
                "The adjustments currently applied to the balance, coming "
                "from the sidebar sliders.")
    a: BalanceAssumptions = st.session_state["assumptions"]
    st.json(asdict(a))

    st.subheader("Save / Load")
    chart_intro("Export / import JSON",
                "Freezes a set of assumptions so you can re-use it later or "
                "share with a colleague.")
    blob = params_to_json(asdict(a))
    st.download_button("Download parameters (JSON)", blob.encode("utf-8"),
                       file_name="commodity_sd_params.json", mime="application/json")
    uploaded = st.file_uploader("Load parameters (JSON)", type=["json"])
    if uploaded is not None:
        try:
            loaded = params_from_json(uploaded.read().decode("utf-8"))
            st.session_state["assumptions"] = BalanceAssumptions(**loaded)
            st.success("Parameters loaded — they apply across all pages.")
        except Exception as exc:
            st.error(f"Load failed: {exc}")

    st.subheader("Cache")
    if st.button("Clear cache & reload"):
        st.cache_data.clear()
        st.rerun()

    st.subheader("Ideal benchmarks per commodity")
    chart_intro("Comparative view",
                "Summary table of normal parameters for each product. Acts "
                "as the reference for reading the other pages.")
    rows = []
    for k, tpl in COMMODITY_TEMPLATES.items():
        rows.append({
            "Commodity": tpl.name, "Sector": tpl.sector, "Ticker": tpl.ticker,
            "Quote": tpl.price_unit, "Days-of-cover target": tpl.days_cover_target,
            "Ideal util %": tpl.ideal_utilization_pct,
            "Monthly vol %": tpl.typical_monthly_vol_pct,
            "YoY demand growth %": tpl.normal_yoy_demand_pct,
            "% MM/OI": tpl.ideal_mm_pct_of_oi,
            "Storage capacity": tpl.storage_capacity,
            "Supply lag (months)": tpl.supply_lag_months,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Detailed commodity templates")
    for k, tpl in COMMODITY_TEMPLATES.items():
        with st.expander(f"{tpl.name} — full template"):
            st.json({
                "key": tpl.key, "name": tpl.name, "sector": tpl.sector,
                "unit": tpl.unit, "inventory_unit": tpl.inventory_unit,
                "ticker": tpl.ticker, "price_unit": tpl.price_unit,
                "base_supply": tpl.base_supply, "base_demand": tpl.base_demand,
                "base_price": tpl.base_price, "price_band": tpl.price_band,
                "storage_capacity": tpl.storage_capacity,
                "days_cover_target": tpl.days_cover_target,
                "ideal_utilization_pct": tpl.ideal_utilization_pct,
                "typical_monthly_vol_pct": tpl.typical_monthly_vol_pct,
                "normal_yoy_demand_pct": tpl.normal_yoy_demand_pct,
                "ideal_mm_pct_of_oi": tpl.ideal_mm_pct_of_oi,
                "regions": tpl.regions, "region_weights": tpl.region_weights,
                "elasticity_alpha": tpl.elasticity_alpha,
                "elasticity_beta": tpl.elasticity_beta,
                "supply_lag_months": tpl.supply_lag_months,
                "seasonal_demand": tpl.seasonal_demand,
                "seasonal_supply": tpl.seasonal_supply,
            })


# =============================================================================
# MAIN APP
# =============================================================================

def main() -> None:
    st.set_page_config(page_title="Commodity S&D Desk", page_icon="🛢️",
                       layout="wide", initial_sidebar_state="expanded")
    register_theme()
    apply_page_style()
    sidebar_controls()

    ck = st.session_state["commodity_key"]
    tpl = COMMODITY_TEMPLATES[ck]
    df = get_sd_dataset(ck, start=st.session_state["history_start"],
                        forecast_months=st.session_state["horizon_months"])
    bal = run_balance(df, ck, st.session_state["assumptions"])
    fv = estimate_fair_value(bal, ck)

    page = st.session_state["page"]
    if page == "🏠 Dashboard":
        page_dashboard(tpl, df, bal, fv)
    elif page == "⚖️ Supply & Demand":
        page_supply_demand(tpl, df)
    elif page == "🛢️ Inventories":
        page_inventories(tpl, df, bal)
    elif page == "🌪️ Scenarios":
        page_scenarios(tpl, df)
    elif page == "🌍 Regional Flows":
        page_regional(tpl)
    elif page == "📈 Futures Curve":
        page_futures_curve(tpl, bal)
    elif page == "🏦 Macro":
        page_macro(tpl, df)
    elif page == "🎲 Monte Carlo":
        page_monte_carlo(tpl, df)
    elif page == "📉 Sensitivities":
        page_sensitivities(tpl, df)
    elif page == "⚙️ Settings":
        page_settings()


if __name__ == "__main__":
    main()
