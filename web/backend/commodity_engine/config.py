"""Commodity templates - same shape as the Streamlit app."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class CommodityTemplate:
    key: str
    name: str
    sector: str
    unit: str
    inventory_unit: str
    ticker: str
    price_unit: str
    base_supply: float
    base_demand: float
    base_price: float
    price_band: Tuple[float, float]
    storage_capacity: float
    days_cover_target: float
    ideal_utilization_pct: float
    typical_monthly_vol_pct: float
    seasonal_demand: List[float] = field(default_factory=list)
    seasonal_supply: List[float] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    region_weights: List[float] = field(default_factory=list)
    supply_weights: List[float] = field(default_factory=list)
    yf_fmt: str = ""
    active_months: str = "FGHJKMNQUVXZ"


_OIL_DEM = [1.02, 1.00, 0.99, 0.98, 0.99, 1.01, 1.03, 1.04, 1.02, 1.00, 0.99, 1.03]
_OIL_SUP = [1.00] * 12
_GAS_DEM = [1.35, 1.25, 1.10, 0.90, 0.80, 0.85, 0.95, 0.95, 0.85, 0.90, 1.10, 1.30]
_FLAT = [1.00] * 12


COMMODITY_TEMPLATES: Dict[str, CommodityTemplate] = {
    "wti_crude": CommodityTemplate(
        key="wti_crude", name="WTI Crude Oil", sector="Energy",
        unit="mb/d", inventory_unit="mb", ticker="CL=F", price_unit="$/bbl",
        base_supply=101.0, base_demand=100.5, base_price=70.0,
        price_band=(40.0, 130.0), storage_capacity=4200.0,
        days_cover_target=30.0, ideal_utilization_pct=70,
        typical_monthly_vol_pct=8, seasonal_demand=_OIL_DEM, seasonal_supply=_OIL_SUP,
        regions=["US", "Middle East", "Russia & CIS", "China", "Europe",
                 "Other Asia", "Rest of World"],
        region_weights=[0.19, 0.09, 0.04, 0.16, 0.13, 0.13, 0.26],
        supply_weights=[0.20, 0.30, 0.11, 0.04, 0.04, 0.02, 0.29],
        yf_fmt="CL{M}{YY}.NYM",
    ),
    "brent_crude": CommodityTemplate(
        key="brent_crude", name="Brent Crude Oil", sector="Energy",
        unit="mb/d", inventory_unit="mb", ticker="BZ=F", price_unit="$/bbl",
        base_supply=101.0, base_demand=100.5, base_price=74.0,
        price_band=(40.0, 130.0), storage_capacity=4200.0,
        days_cover_target=30.0, ideal_utilization_pct=70,
        typical_monthly_vol_pct=8, seasonal_demand=_OIL_DEM, seasonal_supply=_OIL_SUP,
        regions=["US", "Middle East", "Russia & CIS", "China", "Europe",
                 "Other Asia", "Rest of World"],
        region_weights=[0.19, 0.09, 0.04, 0.16, 0.13, 0.13, 0.26],
        supply_weights=[0.20, 0.30, 0.11, 0.04, 0.04, 0.02, 0.29],
        yf_fmt="BZ{M}{YY}.NYM",
    ),
    "henry_hub_gas": CommodityTemplate(
        key="henry_hub_gas", name="Natural Gas (Henry Hub)", sector="Energy",
        unit="bcf/d", inventory_unit="bcf", ticker="NG=F", price_unit="$/MMBtu",
        base_supply=105.0, base_demand=104.0, base_price=3.20,
        price_band=(1.50, 9.00), storage_capacity=4200.0,
        days_cover_target=35.0, ideal_utilization_pct=75,
        typical_monthly_vol_pct=12, seasonal_demand=_GAS_DEM, seasonal_supply=_FLAT,
        regions=["US", "Mexico", "LNG export", "Power", "Industrial"],
        region_weights=[0.40, 0.06, 0.14, 0.25, 0.15],
        supply_weights=[0.92, 0.02, 0.0, 0.0, 0.06],
        yf_fmt="NG{M}{YY}.NYM",
    ),
    "rbob_gasoline": CommodityTemplate(
        key="rbob_gasoline", name="RBOB Gasoline", sector="Energy",
        unit="mb/d", inventory_unit="mb", ticker="RB=F", price_unit="$/gal",
        base_supply=27.0, base_demand=26.5, base_price=2.20,
        price_band=(1.50, 4.50), storage_capacity=280.0,
        days_cover_target=23.0, ideal_utilization_pct=80,
        typical_monthly_vol_pct=9, seasonal_demand=_OIL_DEM, seasonal_supply=_OIL_SUP,
        regions=["US", "Europe", "Asia", "Rest of World"],
        region_weights=[0.34, 0.22, 0.28, 0.16],
        supply_weights=[0.40, 0.20, 0.25, 0.15],
        yf_fmt="RB{M}{YY}.NYM",
    ),
    "ulsd_heating_oil": CommodityTemplate(
        key="ulsd_heating_oil", name="Heating Oil (ULSD)", sector="Energy",
        unit="mb/d", inventory_unit="mb", ticker="HO=F", price_unit="$/gal",
        base_supply=27.0, base_demand=26.0, base_price=2.30,
        price_band=(1.60, 4.80), storage_capacity=180.0,
        days_cover_target=28.0, ideal_utilization_pct=75,
        typical_monthly_vol_pct=9, seasonal_demand=_GAS_DEM, seasonal_supply=_OIL_SUP,
        regions=["US", "Europe", "Asia", "Rest of World"],
        region_weights=[0.30, 0.30, 0.25, 0.15],
        supply_weights=[0.35, 0.25, 0.25, 0.15],
        yf_fmt="HO{M}{YY}.NYM",
    ),
    "gold": CommodityTemplate(
        key="gold", name="Gold", sector="Precious",
        unit="t/mo", inventory_unit="t", ticker="GC=F", price_unit="$/oz",
        base_supply=305.0, base_demand=300.0, base_price=2650.0,
        price_band=(1500.0, 3500.0), storage_capacity=5000.0,
        days_cover_target=90.0, ideal_utilization_pct=60,
        typical_monthly_vol_pct=4,
        seasonal_demand=[1.15, 1.08, 0.95, 0.92, 0.94, 0.92, 0.95, 0.98, 1.00, 1.10, 1.05, 1.10],
        seasonal_supply=_FLAT,
        regions=["China", "Russia", "Australia", "Canada",
                 "India (consumer)", "OECD ETFs", "Central Banks"],
        region_weights=[0.10, 0.06, 0.04, 0.04, 0.22, 0.30, 0.24],
        supply_weights=[0.10, 0.10, 0.10, 0.06, 0.02, 0.0, 0.62],
        yf_fmt="GC{M}{YY}.CMX", active_months="GJMQVZ",
    ),
    "silver": CommodityTemplate(
        key="silver", name="Silver", sector="Precious",
        unit="t/mo", inventory_unit="t", ticker="SI=F", price_unit="$/oz",
        base_supply=2400.0, base_demand=2500.0, base_price=30.0,
        price_band=(15.0, 50.0), storage_capacity=30000.0,
        days_cover_target=90.0, ideal_utilization_pct=60,
        typical_monthly_vol_pct=7, seasonal_demand=_FLAT, seasonal_supply=_FLAT,
        regions=["China", "India", "OECD", "Industrial", "Rest of World"],
        region_weights=[0.28, 0.18, 0.27, 0.18, 0.09],
        supply_weights=[0.20, 0.10, 0.20, 0.0, 0.50],
        yf_fmt="SI{M}{YY}.CMX", active_months="HKNUZ",
    ),
    "comex_copper": CommodityTemplate(
        key="comex_copper", name="Copper (COMEX)", sector="Metals",
        unit="kt/mo", inventory_unit="kt", ticker="HG=F", price_unit="$/lb",
        base_supply=1900.0, base_demand=1910.0, base_price=4.40,
        price_band=(2.50, 6.00), storage_capacity=1400.0,
        days_cover_target=20.0, ideal_utilization_pct=50,
        typical_monthly_vol_pct=6, seasonal_demand=_FLAT, seasonal_supply=_FLAT,
        regions=["Chile & Peru", "China", "DRC & Zambia", "US", "Europe",
                 "Japan/Korea", "Rest of World"],
        region_weights=[0.05, 0.55, 0.04, 0.08, 0.14, 0.07, 0.07],
        supply_weights=[0.38, 0.09, 0.16, 0.06, 0.04, 0.02, 0.25],
        yf_fmt="HG{M}{YY}.CMX", active_months="HKNUZ",
    ),
    "cbot_wheat": CommodityTemplate(
        key="cbot_wheat", name="Wheat (CBOT)", sector="Ags",
        unit="mt/mo", inventory_unit="mt", ticker="ZW=F", price_unit="¢/bu",
        base_supply=65.0, base_demand=64.5, base_price=560.0,
        price_band=(380.0, 1100.0), storage_capacity=320.0,
        days_cover_target=70.0, ideal_utilization_pct=60,
        typical_monthly_vol_pct=7, seasonal_demand=_FLAT,
        seasonal_supply=[0.60, 0.50, 0.60, 0.80, 1.10, 1.60, 1.90, 1.70, 1.30, 0.90, 0.70, 0.60],
        regions=["China", "EU", "India", "Russia", "US", "Canada", "Ukraine",
                 "Rest of World"],
        region_weights=[0.18, 0.13, 0.13, 0.05, 0.04, 0.01, 0.02, 0.44],
        supply_weights=[0.17, 0.17, 0.14, 0.11, 0.06, 0.04, 0.04, 0.27],
        yf_fmt="ZW{M}{YY}.CBT", active_months="HKNUZ",
    ),
    "corn": CommodityTemplate(
        key="corn", name="Corn (CBOT)", sector="Ags",
        unit="mt/mo", inventory_unit="mt", ticker="ZC=F", price_unit="¢/bu",
        base_supply=100.0, base_demand=99.0, base_price=430.0,
        price_band=(330.0, 800.0), storage_capacity=700.0,
        days_cover_target=80.0, ideal_utilization_pct=55,
        typical_monthly_vol_pct=6, seasonal_demand=_FLAT,
        seasonal_supply=[0.50, 0.40, 0.50, 0.70, 1.10, 1.50, 1.40, 1.10, 1.50, 1.80, 1.40, 0.70],
        regions=["US", "China", "Brazil", "EU", "Argentina", "Rest of World"],
        region_weights=[0.31, 0.27, 0.08, 0.06, 0.03, 0.25],
        supply_weights=[0.30, 0.23, 0.11, 0.05, 0.04, 0.27],
        yf_fmt="ZC{M}{YY}.CBT", active_months="HKNUZ",
    ),
}
