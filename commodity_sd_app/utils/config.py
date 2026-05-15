"""
Central configuration for the Commodity S&D analytics platform.

Defines commodity templates, default parameters and UI constants in one place
so the rest of the codebase can stay free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class CommodityTemplate:
    """Static template describing the structural characteristics of a commodity."""

    key: str
    name: str
    unit: str                 # e.g. "mb/d", "bcf/d", "kt", "mt"
    inventory_unit: str       # e.g. "mb", "bcf", "kt"
    ticker: str               # Yahoo Finance front-month proxy
    base_supply: float
    base_demand: float
    base_price: float
    price_band: tuple         # (low, high) historical-ish band
    storage_capacity: float
    days_cover_target: float
    seasonal_demand: List[float] = field(default_factory=list)  # 12 monthly factors
    seasonal_supply: List[float] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    elasticity_alpha: float = 0.0   # demand price elasticity
    elasticity_beta: float = 0.0    # supply price elasticity
    supply_lag_months: int = 6      # default lag for supply response


# Seasonality patterns are *multiplicative factors* around 1.0
# (e.g. winter gas demand > 1, summer demand < 1).
_OIL_DEMAND_SEAS = [1.02, 1.00, 0.99, 0.98, 0.99, 1.01, 1.03, 1.04, 1.02, 1.00, 0.99, 1.03]
_OIL_SUPPLY_SEAS = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]

_GAS_DEMAND_SEAS = [1.35, 1.25, 1.10, 0.90, 0.80, 0.85, 0.95, 0.95, 0.85, 0.90, 1.10, 1.30]
_GAS_SUPPLY_SEAS = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]

_COPPER_DEMAND_SEAS = [0.95, 0.97, 1.02, 1.04, 1.05, 1.03, 1.00, 0.99, 1.02, 1.04, 1.00, 0.89]
_COPPER_SUPPLY_SEAS = [0.97, 0.96, 1.00, 1.02, 1.03, 1.03, 1.02, 1.02, 1.01, 1.01, 0.98, 0.95]

_WHEAT_DEMAND_SEAS = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
_WHEAT_SUPPLY_SEAS = [0.60, 0.50, 0.60, 0.80, 1.10, 1.60, 1.90, 1.70, 1.30, 0.90, 0.70, 0.60]


COMMODITY_TEMPLATES: Dict[str, CommodityTemplate] = {
    "crude_oil": CommodityTemplate(
        key="crude_oil",
        name="Crude Oil",
        unit="mb/d",
        inventory_unit="mb",
        ticker="CL=F",
        base_supply=101.0,
        base_demand=100.5,
        base_price=78.0,
        price_band=(40.0, 130.0),
        storage_capacity=4200.0,
        days_cover_target=30.0,
        seasonal_demand=_OIL_DEMAND_SEAS,
        seasonal_supply=_OIL_SUPPLY_SEAS,
        regions=["US", "Europe", "China", "Middle East", "Rest of World"],
        elasticity_alpha=0.06,
        elasticity_beta=0.10,
        supply_lag_months=6,
    ),
    "natural_gas": CommodityTemplate(
        key="natural_gas",
        name="Natural Gas",
        unit="bcf/d",
        inventory_unit="bcf",
        ticker="NG=F",
        base_supply=105.0,
        base_demand=104.0,
        base_price=3.20,
        price_band=(1.50, 9.00),
        storage_capacity=4200.0,
        days_cover_target=35.0,
        seasonal_demand=_GAS_DEMAND_SEAS,
        seasonal_supply=_GAS_SUPPLY_SEAS,
        regions=["US", "Europe", "Asia LNG", "Rest of World"],
        elasticity_alpha=0.18,
        elasticity_beta=0.08,
        supply_lag_months=4,
    ),
    "copper": CommodityTemplate(
        key="copper",
        name="Copper",
        unit="kt/mo",
        inventory_unit="kt",
        ticker="HG=F",
        base_supply=1900.0,
        base_demand=1910.0,
        base_price=9200.0,
        price_band=(5500.0, 11500.0),
        storage_capacity=600.0,
        days_cover_target=20.0,
        seasonal_demand=_COPPER_DEMAND_SEAS,
        seasonal_supply=_COPPER_SUPPLY_SEAS,
        regions=["China", "Europe", "US", "Rest of World"],
        elasticity_alpha=0.04,
        elasticity_beta=0.07,
        supply_lag_months=12,
    ),
    "wheat": CommodityTemplate(
        key="wheat",
        name="Wheat",
        unit="mt/mo",
        inventory_unit="mt",
        ticker="ZW=F",
        base_supply=65.0,
        base_demand=64.5,
        base_price=620.0,
        price_band=(380.0, 1100.0),
        storage_capacity=320.0,
        days_cover_target=70.0,
        seasonal_demand=_WHEAT_DEMAND_SEAS,
        seasonal_supply=_WHEAT_SUPPLY_SEAS,
        regions=["US", "EU", "Black Sea", "China", "Rest of World"],
        elasticity_alpha=0.05,
        elasticity_beta=0.03,
        supply_lag_months=9,
    ),
}


SCENARIO_PRESETS: Dict[str, Dict[str, float]] = {
    "Bull": {
        "supply_shock_pct": -2.0,
        "demand_shock_pct": +1.5,
        "gdp_growth_pct": 3.5,
        "weather_shock_pct": +1.0,
        "fx_usd_pct": -2.0,
        "probability": 0.25,
    },
    "Base": {
        "supply_shock_pct": 0.0,
        "demand_shock_pct": 0.0,
        "gdp_growth_pct": 2.5,
        "weather_shock_pct": 0.0,
        "fx_usd_pct": 0.0,
        "probability": 0.50,
    },
    "Bear": {
        "supply_shock_pct": +2.0,
        "demand_shock_pct": -2.0,
        "gdp_growth_pct": 0.5,
        "weather_shock_pct": -1.0,
        "fx_usd_pct": +3.0,
        "probability": 0.25,
    },
}


# Theming
DARK_BG = "#0e1117"
PANEL_BG = "#161b22"
ACCENT = "#00d4ff"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
GREY = "#9ca3af"

PLOTLY_TEMPLATE = "plotly_dark"
