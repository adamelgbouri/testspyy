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
    price_unit: str = "$"        # unité de cotation: $/bbl, $/oz, ¢/lb, $/t, ¢/bu, $/MMBtu...
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


@st.cache_data(ttl=600, show_spinner=False)
def get_futures_curve(commodity_key: str, structure: str = "contango",
                      months: int = 24, seed: int = 5) -> pd.DataFrame:
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
                padding: 0.6rem 0.9rem;
            }
            div[data-testid="stMetricValue"] {color: #e5e7eb;}
            h1, h2, h3, h4 {color: #f3f4f6 !important;}
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


# Texte d'aide par page — explications simples sur l'origine des données et l'utilité.
HELP_TEXT: Dict[str, str] = {
    "🏠 Dashboard": """
**À quoi sert cette page ?** Vue d'ensemble du marché : un seul écran pour
voir le prix, l'équilibre offre/demande, les stocks, la télémétrie et le
positionnement des spéculateurs.

**D'où viennent les chiffres ?**
- **Prix spot** : récupéré en direct depuis Yahoo Finance via le ticker
  du contrat à terme (CL=F pour le pétrole, GC=F pour l'or, etc.). Si
  l'accès internet est indisponible, on retombe sur une valeur de
  référence interne. L'unité de cotation (\$/bbl, \$/oz, ¢/lb…) est
  affichée à côté.
- **Offre, demande, stocks** : séries mensuelles **simulées** à partir
  des paramètres typiques du marché (production de référence,
  consommation, saisonnalité, croissance tendancielle). Une vraie
  source (EIA, IEA, USDA, JODI…) peut remplacer ce générateur.
- **Régions** : répartition de l'offre et de la demande selon les parts
  historiques de chaque grande zone (par ex. la Chine prend ~55 % de la
  demande mondiale de cuivre).
- **Télémétrie haute fréquence** : suivi quotidien simulé — nombre de
  navires en mer, taux d'utilisation des raffineries, estimation
  satellite de production.
- **Positionnement spéculatif** : positions nettes des hedge funds et
  indice de sentiment, style rapport CFTC, simulés.

**Ce que vous trouvez ici.**
- Bandeau d'indicateurs clés (prix, juste valeur, stocks, jours de
  couverture, taux d'utilisation du stockage).
- Tableau « Benchmarks vs Actuel » qui compare chaque indicateur à sa
  norme idéale pour ce produit.
- Graphique consolidé Offre / Demande / Stocks.
- Comparaison prix observé vs juste valeur modélisée.
- Trajectoire des stocks et jours de couverture.
- Photo régionale et télémétrie quotidienne.
""",
    "⚖️ Supply & Demand": """
**À quoi sert cette page ?** Construire et tester un bilan offre/demande.
On part d'une série historique, on applique des hypothèses (croissance,
choc d'offre, météo…) et on regarde comment les stocks évoluent.

**D'où viennent les chiffres ?**
- Les séries mensuelles sont **synthétiques** par défaut. Vous pouvez
  charger votre propre fichier CSV (colonnes `date`, `supply`, `demand`
  au minimum, plus optionnellement `imports`, `exports`, `stocks`,
  `price`).
- Les ajustements (production, demande, météo, PIB, raffinage…) viennent
  directement des curseurs de la barre latérale.

**Logique de calcul.**
- Identité comptable : *Stocks fin = Stocks début + Offre − Demande*.
  On peut afficher en cadence mensuelle, trimestrielle ou annuelle.
- L'historique reste figé ; les ajustements ne s'appliquent qu'aux mois
  prévisionnels, pour ne pas réécrire le passé.

**Sections de la page.**
- Indicateurs clés du bilan (stocks fin, surplus/déficit, jours de
  couverture, utilisation du stockage).
- Graphique consolidé Offre/Demande/Stocks.
- Histogramme des builds/draws mensuels.
- Module saisonnalité (profil mensuel, heatmap, décomposition
  tendance/saisonnier/résidu).
- Courbes d'élasticité (offre et demande en fonction du prix avec point
  d'équilibre).
- Réponse retardée de l'offre à un choc de prix (par ex. délai
  shale, lag de plantation).
- Tableau exportable en CSV ou Excel.
""",
    "🛢️ Inventories": """
**À quoi sert cette page ?** Mettre les stocks à l'épreuve des
contraintes de capacité, simuler un stockage flottant, et suivre les
jours de couverture.

**D'où viennent les chiffres ?**
- Mêmes séries que la page Supply & Demand (bilan calculé à partir des
  hypothèses de la barre latérale).
- La capacité de stockage et la part de stockage flottant sont
  modifiables en haut de page.

**Logique.**
- Mois par mois, les stocks sont projetés à partir des builds/draws.
- Si les stocks projetés dépassent la capacité, le surplus part en
  *stockage flottant* (limité par le % de la capacité).
- Le taux d'utilisation = stocks ÷ capacité × 100. Comparé à la norme
  idéale de la commodité.

**Sections.**
- Indicateurs clés (stocks courants, utilisation, stocks flottants,
  jours de couverture).
- Trajectoire d'inventaire.
- Waterfall des 12 dernières variations.
- Jauge d'utilisation.
- Jours de couverture forward avec ligne cible.
- Aire empilée stocks terrestres + flottants.
""",
    "🌪️ Scenarios": """
**À quoi sert cette page ?** Comparer rapidement trois scénarios —
Bull, Base, Bear — pour cadrer un range de prix probable.

**D'où viennent les chiffres ?**
- Chaque scénario applique un jeu d'ajustements pré-réglés (choc
  d'offre, choc de demande, croissance PIB, météo, USD) sur le bilan de
  base. Les probabilités sont éditables en haut de page.
- Le bilan est rejoué pour chaque scénario, puis la juste valeur est
  recalculée pour donner une trajectoire de prix implicite.

**Sections.**
- Trajectoires de stocks par scénario.
- Trajectoires de prix par scénario.
- Tableau récapitulatif (probabilité, stocks fin, jours de couverture,
  build/draw 12 mois, prix moyen, juste valeur fin).
- Prix moyen attendu pondéré par les probabilités.
""",
    "🌍 Regional Flows": """
**À quoi sert cette page ?** Voir qui produit, qui consomme, et déduire
les flux commerciaux entre régions.

**D'où viennent les chiffres ?**
- La répartition régionale utilise les **poids historiques** de chaque
  zone définis dans la fiche commodité (par ex. Chine 55 % de la demande
  cuivre).
- L'offre et la demande sont simulées par région avec un peu de bruit
  pour générer des surplus/déficits réalistes.

**Logique.**
- *Balance régionale* = offre − demande. Positive → exportateur net.
  Négative → importateur net.
- Les flux du diagramme Sankey sont calculés en faisant transiter le
  surplus de chaque exportateur vers les importateurs au prorata de
  leur déficit.

**Sections.**
- Bar chart Offre vs Demande par région.
- Sankey des flux commerciaux implicites.
- Tableau de signaux d'arbitrage (Export Arb / Import Need / Neutre).
""",
    "📈 Futures Curve": """
**À quoi sert cette page ?** Analyser la structure à terme : contango
ou backwardation, spreads de calendrier, économie du stockage.

**D'où viennent les chiffres ?**
- La courbe est **simulée** selon la forme choisie (contango,
  backwardation, plate). Une connexion à un vrai feed (CME, ICE) peut
  remplacer ce générateur.
- Le prix spot vient de la page Dashboard (Yahoo en direct ou
  référence interne).

**Logique.**
- *Contango* : prix forward > spot (marché long sur stocks, faible
  rareté).
- *Backwardation* : prix forward < spot (marché tendu, prime à la
  détention physique).
- *Économie du stockage* : on regarde si la prime de contango couvre
  le coût mensuel de stockage + financement. Si oui, "positive carry"
  → on peut stocker physiquement et vendre forward.

**Sections.**
- Graphique de la courbe (avec étiquette du régime détecté).
- Tableau des spreads calendaires (m1-m2, m1-m6, m1-m12, m6-m12).
- Tableau économie du stockage (prime de contango vs coût de portage).
- Lecture du marché : indice de tension dérivé des jours de couverture.
""",
    "🏦 Macro": """
**À quoi sert cette page ?** Voir comment les grands agrégats macro
(PIB, PMI, USD, taux directeur) influencent le prix de la commodité.

**D'où viennent les chiffres ?**
- Le panel macro est **simulé** sur 7 ans : indice PIB (marche
  aléatoire avec dérive), PMI (oscillant autour de 50), indice USD,
  taux directeur. Peut être remplacé par des séries FRED, OCDE, etc.
- Le prix de la commodité vient de la série synthétique de la page
  Supply & Demand.

**Analyses disponibles.**
- *Matrice de corrélation* entre prix et chaque agrégat macro.
- *Scatter plots* avec droite de régression : on choisit deux
  variables et on les confronte au prix.
- *Corrélations roulantes* (24 mois) pour voir comment la relation
  évolue dans le temps.
- *Régression multivariée* : log(prix) expliqué par tous les facteurs
  macro. Retourne les coefficients, la constante et le R².
""",
    "🎲 Monte Carlo": """
**À quoi sert cette page ?** Générer un range probabiliste de prix et
de stocks en tirant des chocs aléatoires (offre, demande, météo,
pannes).

**D'où viennent les chiffres ?**
- Toutes les hypothèses centrales viennent des curseurs de la barre
  latérale. Les volatilités et le profil de pannes sont définis sur la
  page elle-même.

**Comment ça marche ?**
Pour chaque trajectoire simulée :
1. On tire un choc d'offre, un choc de demande et une perturbation
   météo dans des lois normales centrées sur les hypothèses de base.
2. Avec une probabilité paramétrable, on injecte une panne (par ex.
   ouragan, arrêt de raffinerie) qui ampute l'offre.
3. On rejoue le bilan et on recalcule la juste valeur.

**Sorties.**
- Distributions du prix moyen prévisionnel, du niveau de stocks fin et
  du build/draw cumulé. Marqueurs P5/P50/P95.
- Fan charts (cône d'incertitude) pour prix et stocks.
- VaR 95 % : perte de prix dans le pire 5 % des cas.
""",
    "📉 Sensitivities": """
**À quoi sert cette page ?** Identifier *quelles* hypothèses bougent le
plus la métrique cible (stocks fin, prix moyen, build/draw).

**D'où viennent les chiffres ?**
- Base : hypothèses courantes (barre latérale). Chaque variable est
  poussée à son extrême bas et à son extrême haut, toutes choses égales
  par ailleurs.

**Sections.**
- *Tornado chart* : variables classées par amplitude d'impact. Le bar
  le plus long indique la variable la plus sensible.
- *Stress matrix 2D* : on choisit deux variables, on quadrille leurs
  valeurs et on lit la métrique dans chaque cellule. Permet de voir les
  effets d'interaction.
""",
    "⚙️ Settings": """
**À quoi sert cette page ?** Sauvegarder/charger les hypothèses,
nettoyer le cache, consulter les fiches commodité.

**Contenu.**
- *Export/Import JSON* des hypothèses.
- *Bouton de cache* : vide les données mémorisées et recalcule tout.
- *Matrice de benchmarks* : tous les paramètres idéaux (jours de
  couverture cible, utilisation idéale, volatilité typique, croissance
  normale de la demande, % MM/OI cible) pour les 14 commodités côte à
  côte.
- *Fiche détaillée* par commodité (offre/demande de référence, capacité
  de stockage, prix de référence, unité de cotation, bande de prix,
  poids régionaux, élasticités, délai de réponse de l'offre,
  saisonnalités).
""",
}


def render_page_help(page_name: str) -> None:
    """Affiche un encadré expliquant l'origine des données et l'utilité de la page."""
    body = HELP_TEXT.get(page_name)
    if not body:
        return
    with st.expander("ℹ️ À propos de cette page — sources et méthode", expanded=False):
        st.markdown(body)


def chart_intro(title: str, purpose: str) -> None:
    """Affiche un mini titre + ligne d'objectif AVANT un graphique ou un tableau."""
    st.markdown(f"**{title}** — {purpose}")


def interpretation(text: str) -> None:
    """Affiche un encart d'interprétation des données présentées."""
    st.info(f"💡 **Lecture** — {text}")


# ---------------------------------------------------------------------------
# Lectures automatiques — produisent un texte d'interprétation à partir des données
# ---------------------------------------------------------------------------
def _label_dc(actual: float, target: float) -> str:
    if actual < target * 0.85:
        return "tendu"
    if actual > target * 1.15:
        return "ample"
    return "équilibré"


def _label_util(actual: float, ideal: float) -> str:
    if actual < ideal - 12:
        return "sous-utilisé"
    if actual > ideal + 12:
        return "saturé"
    return "normal"


def _label_fv(deviation_pct: float) -> str:
    if deviation_pct > 10:
        return "surévalué"
    if deviation_pct < -10:
        return "sous-évalué"
    return "proche de la juste valeur"


def read_dashboard(tpl: "CommodityTemplate", bal: pd.DataFrame, fv: pd.DataFrame,
                   spot: float, fv_now: float) -> str:
    last = bal.iloc[-1]
    dc = float(last["days_cover_model"])
    util = float(last["capacity_pct"])
    dev = (spot - fv_now) / fv_now * 100 if fv_now else 0.0
    parts = [
        f"Avec un prix spot à **{fmt_price(spot, tpl.price_unit)}** et une "
        f"juste valeur modélisée de **{fmt_price(fv_now, tpl.price_unit)}**, "
        f"le marché est **{_label_fv(dev)}** (écart {dev:+.1f} %).",
        f"Les stocks couvrent **{dc:.0f} jours** de demande (cible "
        f"{tpl.days_cover_target:.0f} j) → bilan **{_label_dc(dc, tpl.days_cover_target)}**.",
        f"Le taux d'utilisation du stockage est de **{util:.0f} %** "
        f"(idéal {tpl.ideal_utilization_pct:.0f} %) → régime "
        f"**{_label_util(util, tpl.ideal_utilization_pct)}**.",
    ]
    return " ".join(parts)


def read_balance(bal: pd.DataFrame, tpl: "CommodityTemplate") -> str:
    last = bal.iloc[-1]
    sd = float(last["surplus_deficit"])
    bd12 = float(bal["build_draw"].iloc[-12:].sum())
    avg_demand = float(bal["demand"].mean())
    sd_pct = sd / avg_demand * 100 if avg_demand else 0.0
    sense = "surplus" if sd > 0 else "déficit"
    cumul = "build" if bd12 > 0 else "draw"
    return (
        f"Sur le dernier mois projeté, le marché est en **{sense}** "
        f"de {abs(sd):,.1f} {tpl.unit} (~{abs(sd_pct):.1f} % de la demande). "
        f"Sur 12 mois glissants, l'inventaire affiche un **{cumul}** cumulé "
        f"de {abs(bd12):,.0f} {tpl.inventory_unit}."
    )


def read_inventory(inv: pd.DataFrame, tpl: "CommodityTemplate") -> str:
    last = inv.iloc[-1]
    util = float(last["utilization_pct"])
    floating = float(last["overflow_floating"])
    msg = (
        f"Utilisation des entrepôts à **{util:.0f} %** "
        f"(idéal {tpl.ideal_utilization_pct:.0f} %) → "
        f"**{_label_util(util, tpl.ideal_utilization_pct)}**. "
    )
    if floating > 1:
        msg += (f"Du stockage flottant est mobilisé ({floating:,.0f} "
                f"{tpl.inventory_unit}) — signe que la capacité fixe est saturée.")
    else:
        msg += "Pas de stockage flottant activé : la capacité fixe suffit."
    return msg


def read_scenarios(results: Dict[str, pd.DataFrame], tpl: "CommodityTemplate") -> str:
    fc = lambda b: b.loc[b["is_forecast"], "price"].mean()
    bull = fc(results["Bull"]); base = fc(results["Base"]); bear = fc(results["Bear"])
    spread_pct = (bull - bear) / base * 100 if base else 0.0
    return (
        f"Le scénario haussier projette un prix moyen de "
        f"**{fmt_price(bull, tpl.price_unit)}**, le central "
        f"**{fmt_price(base, tpl.price_unit)}** et le baissier "
        f"**{fmt_price(bear, tpl.price_unit)}**. "
        f"La fourchette bull–bear représente **{spread_pct:.1f} %** du prix "
        f"central : plus elle est large, plus le marché est exposé aux chocs."
    )


def read_regional(rs: pd.DataFrame) -> str:
    exporters = rs[rs["balance"] > 0]
    importers = rs[rs["balance"] < 0]
    if exporters.empty or importers.empty:
        return "Toutes les régions sont à l'équilibre — pas de flux structurels visibles."
    top_exp = exporters.sort_values("balance", ascending=False).iloc[0]
    top_imp = importers.sort_values("balance").iloc[0]
    return (
        f"**{top_exp['region']}** est l'exportateur dominant ({top_exp['balance']:+,.1f}) ; "
        f"**{top_imp['region']}** est l'importateur le plus déficitaire "
        f"({top_imp['balance']:+,.1f}). "
        "Les flèches du Sankey indiquent qui alimente qui, en proportion des déficits."
    )


def read_curve(structure: str, score: float) -> str:
    if score > 0.3:
        market = "tendu (peu de stocks)"
        consistent = "backwardation"
    elif score < -0.3:
        market = "ample (stocks abondants)"
        consistent = "contango"
    else:
        market = "équilibré"
        consistent = "courbe plate"
    cohérent = (
        "cohérent" if (consistent in structure.lower()
                       or (consistent == "courbe plate" and "flat" in structure.lower()))
        else "incohérent"
    )
    return (
        f"Stocks → marché **{market}**, ce qui suggère une structure en **{consistent}**. "
        f"Observation : **{structure}** ({cohérent} avec la fondamentale). "
        "Quand la courbe diverge de la fondamentale, il y a souvent un trade de "
        "convergence à jouer."
    )


def read_macro(corr: pd.DataFrame) -> str:
    if "price" not in corr.columns:
        return "Pas assez de données pour interpréter."
    line = corr["price"].drop("price", errors="ignore")
    top = line.abs().sort_values(ascending=False)
    if top.empty:
        return ""
    name = top.index[0]
    val = line[name]
    direction = "positive" if val > 0 else "négative"
    label = {"gdp_index": "PIB", "pmi": "PMI", "usd_index": "USD",
             "policy_rate": "taux directeur"}.get(name, name)
    return (
        f"Le facteur macro le plus lié au prix est **{label}** "
        f"(corrélation {val:+.2f}, direction **{direction}**). "
        "Un signe positif signifie que le prix monte quand l'indicateur monte ; négatif, "
        "le contraire. Les corrélations roulantes permettent de voir si cette relation "
        "tient dans le temps."
    )


def read_monte_carlo(avg_price: np.ndarray, end_stocks: np.ndarray,
                     tpl: "CommodityTemplate") -> str:
    p5, p50, p95 = np.quantile(avg_price, [0.05, 0.5, 0.95])
    width_pct = (p95 - p5) / p50 * 100 if p50 else 0.0
    return (
        f"Le prix moyen attendu se situe autour de "
        f"**{fmt_price(p50, tpl.price_unit)}** "
        f"(95 % des trajectoires entre {fmt_price(p5, tpl.price_unit)} et "
        f"{fmt_price(p95, tpl.price_unit)}, soit une largeur de **{width_pct:.0f} %** "
        f"du médian). Plus la fourchette est large, plus le marché est risqué."
    )


def read_tornado(tornado_df: pd.DataFrame) -> str:
    if tornado_df.empty:
        return ""
    top = tornado_df.iloc[-1]
    return (
        f"La variable la plus sensible est **{top['variable']}** : un mouvement entre "
        f"{top['low_input']:+.1f} et {top['high_input']:+.1f} fait varier la métrique "
        f"de {top['range']:,.1f} unités. C'est cette hypothèse-là qu'il faut surveiller "
        "en priorité."
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
        st.markdown("### 🛢️ Desk S&D Commodités")

        st.session_state["page"] = st.radio("Navigation", PAGES,
                                            index=PAGES.index(st.session_state["page"]))
        st.divider()

        keys = list(COMMODITY_TEMPLATES.keys())
        st.session_state["commodity_key"] = st.selectbox(
            "Commodité", options=keys,
            format_func=lambda k: COMMODITY_TEMPLATES[k].name,
            index=keys.index(st.session_state["commodity_key"]),
        )
        st.session_state["horizon_months"] = st.slider(
            "Horizon de prévision (mois)", 6, 36,
            st.session_state["horizon_months"], step=3,
        )
        st.session_state["history_start"] = st.text_input(
            "Début de l'historique (AAAA-MM-JJ)", st.session_state["history_start"]
        )
        st.divider()

        with st.expander("Hypothèses", expanded=False):
            a: BalanceAssumptions = st.session_state["assumptions"]
            a.supply_adj_pct = st.slider("Offre Δ %", -10.0, 10.0, a.supply_adj_pct, 0.1)
            a.demand_adj_pct = st.slider("Demande Δ %", -10.0, 10.0, a.demand_adj_pct, 0.1)
            a.weather_pct = st.slider("Météo Δ %", -5.0, 5.0, a.weather_pct, 0.1)
            a.gdp_growth_pct = st.slider("Croissance PIB %", -2.0, 6.0,
                                         a.gdp_growth_pct or 2.5, 0.1)
            a.imports_adj_pct = st.slider("Imports Δ %", -20.0, 20.0, a.imports_adj_pct, 0.5)
            a.exports_adj_pct = st.slider("Exports Δ %", -20.0, 20.0, a.exports_adj_pct, 0.5)
            if st.session_state["commodity_key"] == "crude_oil":
                a.refinery_runs_pct = st.slider("Activité raffineries Δ %", -10.0, 10.0,
                                                a.refinery_runs_pct, 0.1)
            a.forecast_months = st.session_state["horizon_months"]
            st.session_state["assumptions"] = a

        st.divider()
        st.caption("Streamlit · données live (Yahoo) avec repli synthétique")


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
    st.title(f"🏠 {tpl.name} — Tableau de bord")
    st.caption(f"Secteur : **{tpl.sector}** · Contrat à terme : **{tpl.ticker}** · "
               f"Unité de flux : {tpl.unit} · Stocks : {tpl.inventory_unit} · "
               f"Cotation : {tpl.price_unit}")
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
        spot_delta = "référence interne (live indisponible)"

    fv_now = float(fv.loc[fv.index == last_h.name, "fair_value_price"].iloc[0])
    delta_fv_pct = (spot - fv_now) / fv_now * 100.0 if fv_now else 0.0
    yoy_idx = max(-13, -len(bal))
    yoy_pct = (last_h["price"] - bal["price"].iloc[yoy_idx]) / bal["price"].iloc[yoy_idx] * 100

    # --- Bandeau d'indicateurs clés
    chart_intro("Indicateurs clés",
                "Photographie du marché en un coup d'œil — prix, juste valeur, "
                "stocks et utilisation du stockage, comparés à leurs cibles.")
    kpi_row([
        ("Prix spot", spot_label, spot_delta),
        ("Juste valeur", fmt_price(fv_now, tpl.price_unit),
         f"{delta_fv_pct:+.1f} % vs spot"),
        (f"Stocks fin ({tpl.inventory_unit})",
         f"{last_f['stocks_model']:,.0f}",
         f"YoY prix {yoy_pct:+.1f} %"),
        ("Jours de couverture", f"{last_f['days_cover_model']:.1f}",
         f"cible {tpl.days_cover_target:.0f}"),
        ("Utilisation stockage", f"{last_f['capacity_pct']:.1f} %",
         f"idéal {tpl.ideal_utilization_pct:.0f} %"),
    ])

    interpretation(read_dashboard(tpl, bal, fv, spot, fv_now))

    # --- Benchmarks
    chart_intro("📐 Benchmarks idéaux vs lecture actuelle",
                "Pour chaque indicateur on affiche la valeur 'normale' pour ce "
                "produit et la valeur courante. La colonne *Lecture* dit si "
                "on est dans la norme, au-dessus, ou en dessous.")
    yoy_demand = (bal["demand"].iloc[-1] / bal["demand"].iloc[max(-13, -len(bal))] - 1) * 100
    rolling_vol = bal["price"].pct_change().tail(12).std() * 100
    bench_rows = [
        {"Métrique": "Jours de couverture", "Idéal": f"{tpl.days_cover_target:.0f} j",
         "Actuel": f"{last_f['days_cover_model']:.1f} j",
         "Lecture": "Tendu" if last_f["days_cover_model"] < tpl.days_cover_target * 0.85
                    else "Ample" if last_f["days_cover_model"] > tpl.days_cover_target * 1.15
                    else "Équilibré"},
        {"Métrique": "Utilisation stockage", "Idéal": f"{tpl.ideal_utilization_pct:.0f} %",
         "Actuel": f"{last_f['capacity_pct']:.1f} %",
         "Lecture": "Sous-utilisé" if last_f["capacity_pct"] < tpl.ideal_utilization_pct - 10
                    else "Saturé" if last_f["capacity_pct"] > tpl.ideal_utilization_pct + 10
                    else "Sain"},
        {"Métrique": "Volatilité mensuelle (12m)",
         "Idéal": f"{tpl.typical_monthly_vol_pct:.1f} %",
         "Actuel": f"{rolling_vol:.1f} %",
         "Lecture": "Élevée" if rolling_vol > tpl.typical_monthly_vol_pct * 1.3
                    else "Faible" if rolling_vol < tpl.typical_monthly_vol_pct * 0.7
                    else "Normale"},
        {"Métrique": "Croissance demande YoY",
         "Idéal": f"{tpl.normal_yoy_demand_pct:+.1f} %",
         "Actuel": f"{yoy_demand:+.1f} %",
         "Lecture": "Au-dessus tendance" if yoy_demand > tpl.normal_yoy_demand_pct + 1
                    else "Sous tendance" if yoy_demand < tpl.normal_yoy_demand_pct - 1
                    else "Sur tendance"},
    ]
    st.dataframe(pd.DataFrame(bench_rows), hide_index=True, use_container_width=True)

    st.markdown("---")

    # --- Bloc graphiques principaux
    left, right = st.columns([3, 2])
    with left:
        chart_intro("Offre, demande & stocks",
                    "Montre l'évolution mensuelle de l'offre et de la demande "
                    "(échelle gauche) et le niveau de stocks (échelle droite). "
                    "La ligne verticale marque le début de la prévision.")
        st.plotly_chart(supply_demand_chart(bal, unit=tpl.unit), use_container_width=True)

        chart_intro("Prix observé vs juste valeur",
                    "La juste valeur est estimée à partir d'une régression "
                    "stocks↔prix. La bande violette est l'intervalle ±10 % "
                    "considéré comme 'équitablement valorisé'.")
        st.plotly_chart(fair_value_chart(fv), use_container_width=True)
    with right:
        chart_intro("Trajectoire des stocks",
                    "Stocks projetés mois par mois jusqu'à la fin de l'horizon.")
        st.plotly_chart(inventory_chart(bal, unit=tpl.inventory_unit),
                        use_container_width=True)

        chart_intro("Jours de couverture forward",
                    "Combien de jours de demande les stocks couvriraient. La "
                    "ligne pointillée représente la cible normale pour ce produit.")
        st.plotly_chart(days_cover_chart(bal, target=tpl.days_cover_target),
                        use_container_width=True)

    # --- Régional
    st.markdown("### Photo régionale")
    reg = get_regional_dataset(st.session_state["commodity_key"])
    rs = regional_summary(reg)
    c1, c2 = st.columns([3, 2])
    with c1:
        chart_intro("Offre vs demande par région",
                    "Barres groupées. Une barre verte plus grande que rouge "
                    "= région exportatrice nette ; l'inverse = importatrice nette.")
        st.plotly_chart(regional_bar(reg), use_container_width=True)
    with c2:
        chart_intro("Détail par région",
                    "Balance = Offre − Demande. *Exporter* / *Importer* selon le signe.")
        st.dataframe(rs[["region", "supply", "demand", "balance", "status"]].round(2),
                     use_container_width=True, hide_index=True)
    interpretation(read_regional(rs))

    # --- Télémétrie
    st.markdown("### Télémétrie quotidienne")
    chart_intro("Indicateurs haute fréquence",
                "Trois indicateurs proxy pour suivre l'activité réelle au jour "
                "le jour. La variation 7 jours montre la tendance récente.")
    hf = get_high_frequency(st.session_state["commodity_key"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Navires suivis", int(hf["vessels_tracked"].iloc[-1]),
              f"{hf['vessels_tracked'].iloc[-1] - hf['vessels_tracked'].iloc[-8]:+d} sur 7j")
    c2.metric("Utilisation raffineries %", f"{hf['refinery_util_pct'].iloc[-1]:.1f}",
              f"{hf['refinery_util_pct'].iloc[-1] - hf['refinery_util_pct'].iloc[-8]:+.2f}")
    c3.metric("Production satellite", f"{hf['sat_production_est'].iloc[-1]:,.1f}",
              f"{(hf['sat_production_est'].iloc[-1] / hf['sat_production_est'].iloc[-8] - 1) * 100:+.2f} %")
    interpretation(
        "Une utilisation raffineries en hausse ou des navires en transit qui "
        "augmentent signalent une **demande robuste**. Une baisse soutenue est "
        "souvent annonciatrice d'un relâchement du marché."
    )

    # --- Positionnement
    st.markdown("### Positionnement spéculatif")
    chart_intro("Hedge funds & sentiment",
                "Position nette des Managed Money (positions longues − courtes) "
                "et indice de sentiment global. Indicateur 'mou' : peut diverger "
                "longtemps des fondamentaux.")
    pos = get_positioning(st.session_state["commodity_key"])
    c1, c2 = st.columns(2)
    mm_now = float(pos['managed_money_net'].iloc[-1])
    sent_now = float(pos['sentiment_score'].iloc[-1])
    c1.metric("Managed Money net", f"{mm_now:,.0f}")
    c2.metric("Score de sentiment",
              f"{sent_now:.0f}/100 · {sentiment_label(sent_now)}")
    interpretation(
        f"Le sentiment est à **{sent_now:.0f}/100** ({sentiment_label(sent_now)}). "
        f"Un sentiment extrême (>80 ou <20) est souvent un signal contrarian — "
        f"le consensus est déjà pricé, la prochaine surprise va dans l'autre sens."
    )


def page_supply_demand(tpl: CommodityTemplate, df: pd.DataFrame) -> None:
    st.title(f"⚖️ {tpl.name} — Bilan offre/demande")
    render_page_help("⚖️ Supply & Demand")

    with st.expander("Charger un fichier CSV personnalisé (colonnes : date, supply, demand…)"):
        file = st.file_uploader("Fichier CSV", type=["csv"])
        if file is not None:
            df = load_csv(file)

    chart_intro("Fréquence d'agrégation",
                "Mensuelle (par défaut), trimestrielle ou annuelle. "
                "Choisir une cadence plus large lisse la saisonnalité.")
    freq = st.radio("Fréquence", ["Mensuelle", "Trimestrielle", "Annuelle"], horizontal=True)
    freq_map = {"Mensuelle": "M", "Trimestrielle": "Q", "Annuelle": "Y"}
    bal = run_balance(df, st.session_state["commodity_key"],
                      st.session_state["assumptions"], frequency=freq_map[freq])

    last = bal.iloc[-1]
    chart_intro("Indicateurs clés du bilan",
                "État du marché en fin de période projetée.")
    kpi_row([
        (f"Stocks fin ({tpl.inventory_unit})", f"{last['stocks_model']:,.0f}", None),
        ("Surplus / Déficit", f"{last['surplus_deficit']:+,.1f}", None),
        ("Jours de couverture", f"{last['days_cover_model']:.1f}",
         f"cible {tpl.days_cover_target:.0f}"),
        ("Utilisation stockage", f"{last['capacity_pct']:.1f} %",
         f"idéal {tpl.ideal_utilization_pct:.0f} %"),
    ])
    interpretation(read_balance(bal, tpl))

    chart_intro("Offre, demande & stocks",
                "L'écart entre les deux lignes pleines (offre vs demande) "
                "explique la pente des stocks (ligne pointillée).")
    st.plotly_chart(supply_demand_chart(bal, unit=tpl.unit), use_container_width=True)

    chart_intro("Builds & draws mensuels",
                "Barres vertes = stocks qui montent (build), rouges = qui baissent "
                "(draw). Les 24 derniers mois.")
    st.plotly_chart(balance_bars(bal), use_container_width=True)

    st.markdown("### Saisonnalité")
    chart_intro("Profil saisonnier moyen",
                "Pour chaque mois, on affiche la moyenne historique sur 5 ans, "
                "la fourchette min-max, et l'année en cours en pointillé.")
    profile = monthly_profile(df["demand"])
    piv = year_over_year_pivot(df["demand"])
    c1, c2 = st.columns([2, 3])
    with c1:
        st.plotly_chart(seasonal_lines(profile), use_container_width=True)
    with c2:
        chart_intro("Heatmap saisonnier",
                    "Carte de chaleur année × mois pour repérer les anomalies "
                    "structurelles.")
        st.plotly_chart(seasonal_heatmap(piv, "Heatmap demande (Année × Mois)"),
                        use_container_width=True)

    # Lecture saisonnalité
    peak_month = int(profile["mean"].idxmax())
    trough_month = int(profile["mean"].idxmin())
    months_fr = ["janvier", "février", "mars", "avril", "mai", "juin",
                 "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
    interpretation(
        f"Le pic saisonnier de demande tombe en **{months_fr[peak_month - 1]}** "
        f"({profile.loc[peak_month, 'mean']:,.1f}), le creux en "
        f"**{months_fr[trough_month - 1]}** ({profile.loc[trough_month, 'mean']:,.1f}). "
        "Identifier ce cycle aide à anticiper les builds/draws structurels et "
        "à ne pas confondre un mouvement saisonnier avec un vrai signal fondamental."
    )

    with st.expander("Décomposition saisonnière (tendance / saisonnier / résidu)"):
        st.markdown("On sépare la série en 3 composantes additives : "
                    "**tendance** (long terme), **saisonnier** (cycle annuel) et "
                    "**résidu** (le reste, qui doit ressembler à du bruit si le modèle est bon).")
        trend, seasonal, resid = decompose(df["demand"], period=12)
        st.markdown("**Tendance long terme**")
        st.line_chart(trend.dropna(), height=180)
        st.markdown("**Composante saisonnière**")
        st.line_chart(seasonal.dropna(), height=180)
        st.markdown("**Résidu**")
        st.line_chart(resid.dropna(), height=180)
        st.markdown("**Moyenne mobile 12 mois (lissage)**")
        st.line_chart(rolling_seasonal_average(df["demand"]).dropna(), height=180)

    st.markdown("### Élasticité-prix")
    chart_intro("Courbes d'offre et de demande",
                "Plus α (alpha) est grand, plus la demande baisse quand le prix monte. "
                "Plus β (beta) est grand, plus la production réagit positivement au prix. "
                "Le point d'intersection est l'équilibre de marché.")
    c1, c2 = st.columns(2)
    alpha = c1.slider("Élasticité demande α", 0.0, 0.5, tpl.elasticity_alpha, 0.005)
    beta = c2.slider("Élasticité offre β", 0.0, 0.5, tpl.elasticity_beta, 0.005)
    curves = build_curves(st.session_state["commodity_key"], alpha=alpha, beta=beta)
    p = ElasticityParams(alpha=alpha, beta=beta, base_price=tpl.base_price,
                         d0=tpl.base_demand, s0=tpl.base_supply)
    eq_p, eq_q = equilibrium(p)
    st.plotly_chart(elasticity_chart(curves, eq_p, eq_q), use_container_width=True)
    interpretation(
        f"Le prix d'équilibre est de **{fmt_price(eq_p, tpl.price_unit)}** pour une "
        f"quantité de **{eq_q:,.2f} {tpl.unit}**. "
        f"Si α et β sont faibles, la demande et l'offre sont **rigides** : il "
        "faut de gros mouvements de prix pour rééquilibrer le marché. "
        "Quand α et β sont élevés, le marché est **flexible** et absorbe les "
        "chocs avec moins de volatilité."
    )

    st.markdown("### Réponse retardée de l'offre")
    chart_intro("Effet d'un choc de prix sur la production future",
                "Simule combien de temps la production met à réagir à un mouvement "
                "de prix. Pour le shale c'est ~6 mois ; pour le cuivre on parle "
                "d'années (capex minier).")
    c1, c2 = st.columns(2)
    lag = c1.slider("Délai de réaction (mois)", 1, 18, tpl.supply_lag_months)
    shock = c2.slider("Choc de prix %", -50.0, 50.0, 20.0, 5.0)
    try:
        result = fit_lagged_supply(df, lag_months=lag)
        st.caption(f"Qualité de l'ajustement (R²) sur l'historique : {result.r_squared:.3f}")
    except ValueError as exc:
        st.warning(str(exc))
    proj = project_lagged_response(tpl.base_supply, shock, lag)
    st.line_chart(proj, height=220)
    interpretation(
        f"Avec un délai de {lag} mois et un choc de {shock:+.0f} %, la production "
        f"converge progressivement vers un nouveau niveau. La forme en S montre "
        "que la réaction est nulle tant que le délai n'est pas écoulé, puis "
        "s'accélère, puis se stabilise."
    )

    st.markdown("### Tableau du bilan & export")
    chart_intro("Détail mois par mois",
                "Les 36 dernières lignes du bilan complet, exportable.")
    st.dataframe(bal.tail(36).round(2), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.download_button("Télécharger le bilan (CSV)", df_to_csv_bytes(bal),
                       file_name=f"{st.session_state['commodity_key']}_balance.csv",
                       mime="text/csv")
    c2.download_button("Télécharger le classeur Excel",
                       df_to_excel_bytes({"balance": bal, "seasonal_profile": profile}),
                       file_name=f"{st.session_state['commodity_key']}_workbook.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def page_inventories(tpl: CommodityTemplate, df: pd.DataFrame, bal: pd.DataFrame) -> None:
    st.title(f"🛢️ {tpl.name} — Stocks & stockage")
    render_page_help("🛢️ Inventories")

    chart_intro("Paramètres de stockage",
                "Modifiez la capacité de stockage et la part autorisée en "
                "stockage flottant (navires, wagons). Si les stocks dépassent "
                "la capacité fixe, le surplus est absorbé jusqu'à cette limite.")
    c1, c2, c3 = st.columns(3)
    cap = c1.number_input(f"Capacité ({tpl.inventory_unit})",
                          value=float(tpl.storage_capacity),
                          step=10.0, min_value=10.0)
    floating = c2.slider("Stockage flottant (% capacité)", 0.0, 25.0, 5.0, 0.5)
    allow_neg = c3.checkbox("Autoriser stocks négatifs (debug)", value=False)
    inv = project_inventory(bal, st.session_state["commodity_key"],
                            StorageConfig(capacity=cap, floating_buffer_pct=floating,
                                          allow_negative=allow_neg))

    last = inv.iloc[-1]
    chart_intro("Indicateurs clés des stocks",
                "Lecture de la fin de période projetée.")
    kpi_row([
        (f"Stocks courants ({tpl.inventory_unit})",
         f"{last['stocks_capped']:,.0f}", None),
        ("Utilisation stockage", f"{last['utilization_pct']:.1f} %",
         f"idéal {tpl.ideal_utilization_pct:.0f} %"),
        ("Stocks flottants", f"{last['overflow_floating']:,.0f}", None),
        ("Jours de couverture", f"{last['days_cover_model']:.1f}",
         f"cible {tpl.days_cover_target:.0f}"),
    ])
    interpretation(read_inventory(inv, tpl))

    chart_intro("Trajectoire des stocks",
                "Stocks projetés mois par mois, plafonnés par la capacité fixe.")
    st.plotly_chart(inventory_chart(inv, unit=tpl.inventory_unit),
                    use_container_width=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        chart_intro("Waterfall builds/draws (12 mois)",
                    "Pour chaque mois récent, on voit la variation de stocks. "
                    "Une succession de barres vertes = accumulation soutenue.")
        st.plotly_chart(waterfall_chart(draw_build_waterfall(inv, n=12)),
                        use_container_width=True)
    with c2:
        chart_intro("Jauge d'utilisation",
                    "Visualisation rapide de la pression sur la capacité de "
                    "stockage. Au-delà de 80 %, la marge se réduit.")
        st.plotly_chart(utilization_gauge(last["utilization_pct"]),
                        use_container_width=True)

    chart_intro("Jours de couverture forward",
                "Combien de jours de demande les stocks tiendraient sans nouvelle "
                "production. La ligne pointillée représente la cible normale.")
    st.plotly_chart(days_cover_chart(inv, target=tpl.days_cover_target),
                    use_container_width=True)

    chart_intro("Stocks fixes vs flottants",
                "Aire empilée. Quand la couche bleu clair (flottant) apparaît, "
                "c'est que la capacité fixe est dépassée.")
    st.area_chart(inv[["stocks_capped", "overflow_floating"]].tail(48), height=260)

    chart_intro("Détail mensuel",
                "Tableau récapitulatif des 24 derniers mois.")
    st.dataframe(
        inv[["stocks_model", "stocks_capped", "overflow_floating",
             "utilization_pct", "days_cover_model"]].tail(24).round(2),
        use_container_width=True,
    )


def page_scenarios(tpl: CommodityTemplate, df: pd.DataFrame) -> None:
    st.title(f"🌪️ {tpl.name} — Moteur de scénarios")
    render_page_help("🌪️ Scenarios")

    chart_intro("Probabilités des scénarios",
                "Ajustez les probabilités attribuées à chaque scénario. "
                "Le système les renormalise automatiquement à 100 %.")
    c1, c2, c3 = st.columns(3)
    new_probs = {}
    labels = {"Bull": "Bull (haussier)", "Base": "Base (central)", "Bear": "Bear (baissier)"}
    for col, name in zip([c1, c2, c3], ["Bull", "Base", "Bear"]):
        new_probs[name] = col.number_input(
            f"Probabilité {labels[name]}", min_value=0.0, max_value=1.0,
            value=float(SCENARIO_PRESETS[name]["probability"]), step=0.05,
        )
    total = sum(new_probs.values()) or 1.0
    for n in new_probs:
        SCENARIO_PRESETS[n]["probability"] = new_probs[n] / total

    results = run_scenarios(df, st.session_state["commodity_key"],
                            st.session_state["assumptions"])

    chart_intro("Trajectoires de stocks par scénario",
                "Comment les stocks évoluent dans chaque scénario. Bear = stocks "
                "qui montent (surplus) ; Bull = stocks qui baissent (rareté).")
    st.plotly_chart(scenario_paths(results, "stocks_model", "Trajectoires de stocks"),
                    use_container_width=True)

    chart_intro("Trajectoires de prix par scénario",
                "Prix de juste valeur implicite déduit du niveau de stocks de chaque scénario.")
    st.plotly_chart(scenario_paths(results, "fair_value_price",
                                    "Trajectoires de juste valeur"),
                    use_container_width=True)
    interpretation(read_scenarios(results, tpl))

    chart_intro("Tableau récapitulatif",
                "Synthèse par scénario : probabilité, stocks fin, jours de "
                "couverture, build/draw 12m, prix moyen, juste valeur fin.")
    summary = scenario_summary(results, st.session_state["commodity_key"])
    st.dataframe(summary.round(2), use_container_width=True)
    pw = probability_weighted_price(results)
    st.metric("Prix moyen pondéré par les probabilités",
              fmt_price(pw, tpl.price_unit))
    interpretation(
        f"Le prix attendu en moyenne pondérée est de **{fmt_price(pw, tpl.price_unit)}**. "
        "Cette valeur sert de point d'ancrage pour budgéter ou hedger. "
        "Si le prix de marché actuel diverge fortement de ce niveau, il y a soit "
        "une opportunité, soit un risque que nos hypothèses soient mal calibrées."
    )

    with st.expander("Détails des hypothèses par scénario"):
        st.json(SCENARIO_PRESETS)


def page_regional(tpl: CommodityTemplate) -> None:
    st.title(f"🌍 {tpl.name} — Flux régionaux")
    render_page_help("🌍 Regional Flows")
    reg = get_regional_dataset(st.session_state["commodity_key"])
    rs = regional_summary(reg)

    c1, c2 = st.columns([3, 2])
    with c1:
        chart_intro("Offre vs demande par région",
                    "Barres groupées. Compare directement les capacités de "
                    "production et la consommation de chaque grande zone.")
        st.plotly_chart(regional_bar(reg), use_container_width=True)
    with c2:
        chart_intro("Tableau régional",
                    "Balance = Offre − Demande. Positive → exportateur net.")
        st.dataframe(rs.round(2), hide_index=True, use_container_width=True)
    interpretation(read_regional(rs))

    chart_intro("Flux commerciaux implicites (diagramme de Sankey)",
                "Chaque exportateur net envoie son surplus vers les importateurs "
                "nets, au prorata de leur déficit. L'épaisseur des liens représente "
                "le volume de flux. Permet de visualiser la dépendance commerciale.")
    nodes, sources, targets, values = build_trade_flows(reg)
    st.plotly_chart(sankey_chart(nodes, sources, targets, values),
                    use_container_width=True)

    chart_intro("Signaux d'arbitrage",
                "Tableau qui classe chaque région : *Export Arb* = excédent à "
                "écouler à l'export ; *Import Need* = besoin urgent de cargos. "
                "Plus la balance est extrême, plus le différentiel de prix "
                "régional sera large.")
    arb = arbitrage_signals(reg)
    st.dataframe(arb[["region", "balance", "arb_signal"]].round(2),
                 hide_index=True, use_container_width=True)
    interpretation(
        "Quand une région passe en *Import Need*, les prix locaux montent jusqu'à "
        "attirer assez de cargos. C'est ce mécanisme qui explique la prime "
        "Brent-Dubai, le spread TTF-Henry Hub, ou encore l'écart Chicago-Heartland "
        "sur le maïs."
    )


def page_futures_curve(tpl: CommodityTemplate, bal: pd.DataFrame) -> None:
    st.title(f"📈 {tpl.name} — Structure à terme")
    render_page_help("📈 Futures Curve")

    chart_intro("Paramètres de la courbe",
                "Choisissez la forme de courbe à simuler et les coûts de "
                "portage (stockage mensuel + taux de financement annuel).")
    c1, c2, c3 = st.columns(3)
    structure_choice = c1.selectbox("Forme de courbe",
                                     ["contango", "backwardation", "flat"])
    months = c2.slider("Échéances (mois)", 6, 36, 18)
    storage_cost = c3.number_input(f"Coût de stockage mensuel ({tpl.price_unit})",
                                    value=0.20, step=0.05)
    financing_rate = st.slider("Taux de financement %", 0.0, 12.0, 5.0, 0.25)

    curve = get_futures_curve(st.session_state["commodity_key"],
                              structure=structure_choice, months=months)
    struct_label = classify_structure(curve)

    chart_intro("Courbe à terme",
                "Prix par échéance. **Contango** = courbe ascendante (forwards plus "
                "chers que le spot — marché bien approvisionné). "
                "**Backwardation** = descendante (forwards moins chers — prime à la "
                "détention physique aujourd'hui).")
    st.plotly_chart(futures_curve_chart(curve, struct_label), use_container_width=True)

    c1, c2 = st.columns([2, 3])
    with c1:
        chart_intro("Spreads calendaires",
                    "Différence entre l'échéance proche et les échéances plus "
                    "lointaines. Un spread proche-lointain positif = contango.")
        st.dataframe(calendar_spreads(curve).round(3), use_container_width=True)
    with c2:
        chart_intro("Économie du stockage",
                    "Pour chaque échéance, compare la prime de contango au coût "
                    "de portage (stockage + financement). *positive_carry* = il est "
                    "rentable de stocker physiquement et vendre forward.")
        econ = storage_economics(curve, storage_cost, financing_rate)
        st.dataframe(econ[["tenor_month", "price", "contango_premium",
                           "carry", "positive_carry"]].round(3),
                     hide_index=True, use_container_width=True)
    pos_carry_pct = float(econ["positive_carry"].mean()) * 100
    interpretation(
        f"{pos_carry_pct:.0f} % des échéances offrent un **positive carry** : "
        "à ces tenors, un négociant peut acheter le physique aujourd'hui, "
        "stocker, et vendre forward avec un gain mécanique. "
        "Si le carry est positif partout, le marché est sous offre abondante."
    )

    st.subheader("Cohérence stocks ↔ courbe")
    chart_intro("Lecture du marché",
                "On compare le régime de courbe observé au régime théoriquement "
                "induit par les jours de couverture des stocks.")
    last_dc = float(bal["days_cover_model"].iloc[-1])
    label, score = inventory_curve_relationship(curve, last_dc)
    kpi_row([
        ("Jours de couverture", f"{last_dc:.1f}", None),
        ("Indice de tension", f"{score:+.2f}",
         "+1 très tendu, −1 très ample"),
        ("Diagnostic", label, None),
    ])
    interpretation(read_curve(label, score))

    with st.expander("Données brutes de la courbe"):
        st.dataframe(curve.round(3), hide_index=True, use_container_width=True)


def page_macro(tpl: CommodityTemplate, df: pd.DataFrame) -> None:
    st.title(f"🏦 {tpl.name} — Macro overlay")
    render_page_help("🏦 Macro")
    macro = get_macro_panel(months=84)
    joined = align_macro(df, macro)

    cols = ["price", "gdp_index", "pmi", "usd_index", "policy_rate"]
    corr = correlation_matrix(joined, cols)

    chart_intro("Matrice de corrélation",
                "Mesure de la relation linéaire entre le prix et chaque agrégat "
                "macro. Lecture : vert = corrélation positive, rouge = négative. "
                "Plus la valeur absolue est proche de 1, plus le lien est fort.")
    st.plotly_chart(correlation_heatmap(corr), use_container_width=True)
    interpretation(read_macro(corr))

    chart_intro("Diagnostic croisé prix vs facteur macro",
                "Choisissez deux variables macro à confronter au prix. La droite "
                "de régression montre la tendance moyenne. Une dispersion large "
                "= relation faible ; une forme bien alignée = relation robuste.")
    choices = ["gdp_index", "pmi", "usd_index", "policy_rate"]
    label_map = {"gdp_index": "Indice PIB", "pmi": "PMI",
                 "usd_index": "Indice USD", "policy_rate": "Taux directeur"}
    c1, c2 = st.columns(2)
    with c1:
        x1 = st.selectbox("Variable macro A", choices, index=0,
                          format_func=lambda x: label_map[x], key="ma")
        st.plotly_chart(scatter_with_fit(joined[x1], joined["price"],
                                          label_map[x1], f"Prix ({tpl.price_unit})"),
                        use_container_width=True)
    with c2:
        x2 = st.selectbox("Variable macro B", choices, index=2,
                          format_func=lambda x: label_map[x], key="mb")
        st.plotly_chart(scatter_with_fit(joined[x2], joined["price"],
                                          label_map[x2], f"Prix ({tpl.price_unit})"),
                        use_container_width=True)

    st.subheader("Corrélations glissantes vs prix")
    chart_intro("Stabilité de la relation dans le temps",
                "Corrélation calculée sur une fenêtre roulante de 24 mois. "
                "Si la courbe oscille beaucoup autour de zéro, la relation n'est "
                "pas stable ; si elle reste éloignée de zéro, le lien est durable.")
    c1, c2 = st.columns(2)
    rc1 = rolling_correlation(joined["price"], joined["gdp_index"])
    c1.plotly_chart(rolling_corr_chart(rc1, "Prix vs PIB"), use_container_width=True)
    rc2 = rolling_correlation(joined["price"], joined["usd_index"])
    c2.plotly_chart(rolling_corr_chart(rc2, "Prix vs USD"), use_container_width=True)
    last_usd_corr = float(rc2.dropna().iloc[-1]) if not rc2.dropna().empty else 0
    last_gdp_corr = float(rc1.dropna().iloc[-1]) if not rc1.dropna().empty else 0
    interpretation(
        f"Corrélation 24m récente : Prix–PIB = {last_gdp_corr:+.2f}, Prix–USD = "
        f"{last_usd_corr:+.2f}. Pour la plupart des commodités, on attend une "
        "corrélation négative avec l'USD (prix libellés en dollars) et positive "
        "avec le PIB (demande)."
    )

    st.subheader("Régression multivariée : log(prix) expliqué par la macro")
    chart_intro("Modèle linéaire à plusieurs facteurs",
                "On essaie d'expliquer le logarithme du prix avec **tous** les "
                "facteurs macro simultanément. Le R² indique la part de variance "
                "captée par le modèle (plus c'est proche de 1, mieux c'est).")
    joined["log_price"] = np.log(joined["price"])
    try:
        res = regression_summary(joined, "log_price",
                                 ["gdp_index", "pmi", "usd_index", "policy_rate"])
        st.json(res)
        interpretation(
            f"Le modèle explique **{res['r_squared'] * 100:.0f} %** de la "
            "variance du prix. Plus c'est haut, plus la macro est un bon "
            "raccourci pour comprendre le prix. Au-dessus de 50 %, on peut "
            "raisonnablement utiliser les indicateurs macro pour anticiper les "
            "mouvements."
        )
    except ValueError as exc:
        st.warning(str(exc))


def page_monte_carlo(tpl: CommodityTemplate, df: pd.DataFrame) -> None:
    st.title(f"🎲 {tpl.name} — Moteur Monte Carlo")
    render_page_help("🎲 Monte Carlo")

    chart_intro("Paramètres des chocs aléatoires",
                "Volatilités (σ) des chocs d'offre, de demande, de météo, et "
                "fréquence/intensité des pannes. Plus σ est grand, plus les "
                "trajectoires sont dispersées.")
    c1, c2, c3 = st.columns(3)
    n_paths = c1.slider("Nombre de trajectoires", 100, 2000, 500, step=100)
    sigma_supply = c2.slider("Choc offre σ %", 0.5, 5.0, 1.5, 0.1)
    sigma_demand = c3.slider("Choc demande σ %", 0.5, 5.0, 1.2, 0.1)

    c1, c2, c3 = st.columns(3)
    sigma_weather = c1.slider("Choc météo σ %", 0.0, 3.0, 1.0, 0.1)
    outage_prob = c2.slider("Probabilité de panne (par mois)", 0.0, 0.20, 0.05, 0.01)
    outage_size = c3.slider("Taille de panne %", 0.0, 15.0, 4.0, 0.5)

    cfg = MCConfig(n_paths=n_paths, supply_sigma_pct=sigma_supply,
                   demand_sigma_pct=sigma_demand, weather_sigma_pct=sigma_weather,
                   outage_prob=outage_prob, outage_size_pct=outage_size)

    if st.button("Lancer la simulation Monte Carlo", type="primary"):
        with st.spinner(f"Simulation de {n_paths} trajectoires…"):
            out = run_monte_carlo(df, st.session_state["commodity_key"],
                                  st.session_state["assumptions"], cfg)
        end_stocks = out["end_stocks"]
        avg_price = out["avg_price"]
        bd = out["build_draw"]
        base_price = float(np.median(avg_price))
        losses = np.maximum(0, base_price - avg_price)
        var95 = value_at_risk(losses, 0.95)

        chart_intro("Indicateurs probabilistes",
                    "Synthèse des trajectoires simulées : médiane, fourchette P5-P95, "
                    "et Value at Risk 95 % (perte de prix dans le pire 5 % des cas).")
        kpi_row([
            ("Prix médian", fmt_price(float(np.median(avg_price)), tpl.price_unit), None),
            ("Fourchette P5 – P95",
             f"{fmt_price(float(np.quantile(avg_price, 0.05)), tpl.price_unit)} – "
             f"{fmt_price(float(np.quantile(avg_price, 0.95)), tpl.price_unit)}", None),
            (f"Stocks fin médians ({tpl.inventory_unit})",
             f"{np.median(end_stocks):,.0f}", None),
            (f"VaR 95 % (baisse de prix)",
             fmt_price(var95, tpl.price_unit), None),
        ])
        interpretation(read_monte_carlo(avg_price, end_stocks, tpl))

        c1, c2 = st.columns(2)
        with c1:
            chart_intro("Distribution du prix moyen prévisionnel",
                        "Histogramme des prix moyens obtenus sur les N trajectoires. "
                        "Les barres verticales P5/P50/P95 indiquent les seuils.")
            st.plotly_chart(histogram(avg_price, "Distribution du prix moyen",
                                       x_label=f"Prix ({tpl.price_unit})"),
                            use_container_width=True)
        with c2:
            chart_intro("Distribution des stocks de fin d'horizon",
                        "Plus la distribution est étroite, plus la prévision est "
                        "robuste face aux chocs.")
            st.plotly_chart(histogram(end_stocks, "Distribution des stocks fin",
                                       x_label=f"Stocks ({tpl.inventory_unit})"),
                            use_container_width=True)
        chart_intro("Distribution du cumul build/draw",
                    "Variation totale de stocks sur la période prévue, toutes "
                    "trajectoires confondues.")
        st.plotly_chart(histogram(bd, "Distribution build/draw cumulé",
                                   x_label="Δ Stocks"), use_container_width=True)

        st.subheader("Fan charts probabilistes")
        chart_intro("Cône d'incertitude",
                    "La bande colorée représente l'intervalle P5–P95 mois par "
                    "mois. La ligne centrale est la médiane. Plus le cône s'évase, "
                    "plus l'incertitude grandit avec l'horizon.")
        pct = out["percentiles"]
        c1, c2 = st.columns(2)
        c1.plotly_chart(fan_chart(pct, "price"), use_container_width=True)
        c2.plotly_chart(fan_chart(pct, "stocks"), use_container_width=True)
    else:
        st.info("Réglez les paramètres puis cliquez sur **Lancer la simulation** pour démarrer.")


def page_sensitivities(tpl: CommodityTemplate, df: pd.DataFrame) -> None:
    st.title(f"📉 {tpl.name} — Analyse de sensibilité")
    render_page_help("📉 Sensitivities")

    chart_intro("Choix de la métrique cible",
                "Quelle valeur veut-on tester ? Stocks fin, prix moyen "
                "prévisionnel, ou build/draw cumulé sur 12 mois.")
    metric_map = {
        "end_stocks": "Stocks fin de période",
        "avg_fc_price": "Prix moyen prévisionnel",
        "build_draw_sum": "Build/draw cumulé 12m",
    }
    metric = st.selectbox("Métrique",
                          list(metric_map.keys()),
                          format_func=lambda k: metric_map[k], index=0)

    variables = [
        SensitivityVar("Offre Δ %", "supply_adj_pct", -3.0, 3.0),
        SensitivityVar("Demande Δ %", "demand_adj_pct", -3.0, 3.0),
        SensitivityVar("Météo Δ %", "weather_pct", -2.0, 2.0),
        SensitivityVar("PIB %", "gdp_growth_pct", 0.5, 4.0),
        SensitivityVar("Imports Δ %", "imports_adj_pct", -10.0, 10.0),
        SensitivityVar("Exports Δ %", "exports_adj_pct", -10.0, 10.0),
    ]
    torn = tornado(df, st.session_state["commodity_key"],
                   st.session_state["assumptions"], variables, metric=metric)

    chart_intro("Tornado chart",
                "Pour chaque variable, on pousse à sa valeur basse et haute "
                "et on mesure l'impact sur la métrique. Les barres les plus "
                "longues sont les leviers à surveiller en priorité.")
    st.plotly_chart(tornado_chart(torn), use_container_width=True)
    st.dataframe(torn.round(2), hide_index=True, use_container_width=True)
    interpretation(read_tornado(torn))

    st.subheader("Matrice de stress 2D")
    chart_intro("Effets d'interaction entre deux variables",
                "Heatmap où on quadrille deux hypothèses et on lit la métrique "
                "dans chaque cellule. Très utile pour détecter les zones "
                "extrêmes (par ex. choc offre + choc demande simultanés).")
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
        "Les cellules les plus rouges/bleues du coin de la heatmap indiquent les "
        "combinaisons de stress les plus dangereuses. C'est là qu'il faut "
        "construire ses scénarios de risque extrême."
    )


def page_settings() -> None:
    st.title("⚙️ Paramètres")
    render_page_help("⚙️ Settings")

    chart_intro("Hypothèses actives",
                "Ce sont les ajustements actuellement appliqués au bilan, "
                "issus des curseurs de la barre latérale.")
    a: BalanceAssumptions = st.session_state["assumptions"]
    st.json(asdict(a))

    st.subheader("Sauvegarder / Charger")
    chart_intro("Export / import JSON",
                "Permet de figer un jeu d'hypothèses pour le ré-utiliser plus tard "
                "ou le partager avec un collègue.")
    blob = params_to_json(asdict(a))
    st.download_button("Télécharger les paramètres (JSON)", blob.encode("utf-8"),
                       file_name="commodity_sd_params.json", mime="application/json")
    uploaded = st.file_uploader("Charger des paramètres (JSON)", type=["json"])
    if uploaded is not None:
        try:
            loaded = params_from_json(uploaded.read().decode("utf-8"))
            st.session_state["assumptions"] = BalanceAssumptions(**loaded)
            st.success("Paramètres chargés — ils s'appliquent à toutes les pages.")
        except Exception as exc:
            st.error(f"Échec du chargement : {exc}")

    st.subheader("Cache")
    if st.button("Vider le cache et recharger"):
        st.cache_data.clear()
        st.rerun()

    st.subheader("Benchmarks idéaux par commodité")
    chart_intro("Vue comparative",
                "Tableau récapitulatif des paramètres normaux pour chaque "
                "produit. Sert de référence pour la lecture des autres pages.")
    rows = []
    for k, tpl in COMMODITY_TEMPLATES.items():
        rows.append({
            "Commodité": tpl.name, "Secteur": tpl.sector, "Ticker": tpl.ticker,
            "Cotation": tpl.price_unit, "Jours-couverture cible": tpl.days_cover_target,
            "Utilisation idéale %": tpl.ideal_utilization_pct,
            "Vol. mensuelle %": tpl.typical_monthly_vol_pct,
            "Croissance demande YoY %": tpl.normal_yoy_demand_pct,
            "% MM/OI": tpl.ideal_mm_pct_of_oi,
            "Capacité stockage": tpl.storage_capacity,
            "Délai offre (mois)": tpl.supply_lag_months,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Fiches détaillées par commodité")
    for k, tpl in COMMODITY_TEMPLATES.items():
        with st.expander(f"{tpl.name} — fiche complète"):
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
