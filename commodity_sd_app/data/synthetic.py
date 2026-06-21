"""
Synthetic data generators used when external APIs are unavailable.

All series are reproducible via numpy's default_rng with a configurable seed.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from utils.config import COMMODITY_TEMPLATES, CommodityTemplate


# ---------------------------------------------------------------------------
# Monthly S&D history + forecast horizon
# ---------------------------------------------------------------------------
def generate_sd_history(
    commodity_key: str,
    start: str = "2018-01-01",
    end: Optional[str] = None,
    forecast_months: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a monthly synthetic S&D table for a commodity.

    Columns:
        date, supply, demand, imports, exports, stocks, price, gdp_index,
        weather_index, refinery_runs (oil), is_forecast
    """
    tpl: CommodityTemplate = COMMODITY_TEMPLATES[commodity_key]
    if end is None:
        end = pd.Timestamp(date.today()).strftime("%Y-%m-01")
    idx = pd.date_range(start=start, end=end, freq="MS")
    horizon = pd.date_range(
        start=idx[-1] + pd.offsets.MonthBegin(1),
        periods=forecast_months,
        freq="MS",
    )
    full_idx = idx.append(horizon)

    rng = np.random.default_rng(seed)
    n = len(full_idx)
    months = full_idx.month - 1

    # Seasonal factors
    seas_d = np.array(tpl.seasonal_demand)[months]
    seas_s = np.array(tpl.seasonal_supply)[months]

    # Long-run linear growth - keep supply slightly ahead so synthetic
    # balances don't trivially drain inventory to zero.
    t = np.linspace(0, 1, n)
    demand_trend = tpl.base_demand * (1 + 0.010 * t * 6)        # ~6% over period
    supply_trend = tpl.base_supply * (1 + 0.012 * t * 6)        # ~7%

    # Noise + cycles
    cycle = 0.02 * np.sin(2 * np.pi * np.arange(n) / 48)
    demand = demand_trend * seas_d * (1 + cycle + rng.normal(0, 0.012, n))
    supply = supply_trend * seas_s * (1 + cycle * 0.5 + rng.normal(0, 0.015, n))

    # Trade
    imports = np.abs(rng.normal(tpl.base_demand * 0.20, tpl.base_demand * 0.02, n))
    exports = np.abs(rng.normal(tpl.base_demand * 0.18, tpl.base_demand * 0.02, n))

    # Build inventory series via S&D balance
    days_in_month = full_idx.days_in_month.to_numpy()
    # convert flow units to per-month totals where needed
    if tpl.unit.endswith("/d"):
        supply_monthly = supply * days_in_month
        demand_monthly = demand * days_in_month
    else:
        supply_monthly = supply
        demand_monthly = demand

    if tpl.unit.endswith("/d"):
        starting_stocks = tpl.base_demand * tpl.days_cover_target
    else:
        # monthly-flow commodities - target days_cover converted to months
        starting_stocks = tpl.base_demand * (tpl.days_cover_target / 30.0)
    # never start above 80% capacity
    starting_stocks = min(starting_stocks, 0.8 * tpl.storage_capacity)
    stocks = np.empty(n)
    stocks[0] = starting_stocks
    for i in range(1, n):
        stocks[i] = max(stocks[i - 1] + (supply_monthly[i] - demand_monthly[i]), 0.0)

    # Price - inverse function of inventory days of cover with noise
    avg_daily_demand = demand_monthly / days_in_month
    days_cover = stocks / np.where(avg_daily_demand > 0, avg_daily_demand, 1)
    norm_dc = (days_cover - days_cover.mean()) / max(days_cover.std(), 1e-6)
    lo, hi = tpl.price_band
    mid = (lo + hi) / 2.0
    price = mid * np.exp(-0.18 * norm_dc) * (1 + rng.normal(0, 0.04, n))
    price = np.clip(price, lo * 0.7, hi * 1.3)

    # Macro overlays
    gdp_index = 100 * np.cumprod(1 + rng.normal(0.002, 0.004, n))
    weather_index = 50 + 10 * np.sin(2 * np.pi * np.arange(n) / 12 + 1.1) + rng.normal(0, 2, n)
    refinery_runs = np.clip(85 + 8 * seas_d / seas_d.mean() + rng.normal(0, 1.5, n), 70, 98)

    df = pd.DataFrame(
        {
            "date": full_idx,
            "supply": supply,
            "demand": demand,
            "imports": imports,
            "exports": exports,
            "stocks": stocks,
            "days_cover": days_cover,
            "price": price,
            "gdp_index": gdp_index,
            "weather_index": weather_index,
            "refinery_runs": refinery_runs,
            "is_forecast": [d in horizon for d in full_idx],
        }
    ).set_index("date")
    return df


# ---------------------------------------------------------------------------
# Regional split
# ---------------------------------------------------------------------------
def generate_regional_balances(commodity_key: str, seed: int = 7) -> pd.DataFrame:
    """Return a regional supply/demand/net-trade snapshot."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    rng = np.random.default_rng(seed)
    regions = tpl.regions

    # Demand weights roughly anchored to real-world shares for each commodity
    if tpl.key == "crude_oil":
        weights = np.array([0.21, 0.14, 0.15, 0.09, 0.41])
    elif tpl.key == "natural_gas":
        weights = np.array([0.30, 0.18, 0.20, 0.32])
    elif tpl.key == "copper":
        weights = np.array([0.55, 0.15, 0.10, 0.20])
    else:  # wheat
        weights = np.array([0.10, 0.15, 0.12, 0.18, 0.45])

    weights = weights[: len(regions)] / weights[: len(regions)].sum()
    base_d = tpl.base_demand
    base_s = tpl.base_supply

    demand = base_d * weights * (1 + rng.normal(0, 0.03, len(regions)))
    # Supply skew differs from demand to create natural trade flows
    skew = rng.normal(1.0, 0.20, len(regions))
    supply = base_s * weights * skew
    supply = supply * (base_s / supply.sum())  # rescale

    net_trade = supply - demand  # >0 net exporter, <0 net importer
    df = pd.DataFrame(
        {
            "region": regions,
            "supply": supply,
            "demand": demand,
            "net_trade": net_trade,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Futures curve
# ---------------------------------------------------------------------------
def generate_futures_curve(
    commodity_key: str,
    spot: Optional[float] = None,
    structure: str = "contango",
    months: int = 24,
    seed: int = 5,
) -> pd.DataFrame:
    """
    Build a synthetic futures curve.

    structure: "contango" | "backwardation" | "flat"
    """
    tpl = COMMODITY_TEMPLATES[commodity_key]
    if spot is None:
        spot = tpl.base_price

    rng = np.random.default_rng(seed)
    tenors = np.arange(1, months + 1)
    if structure == "contango":
        slope = 0.005
    elif structure == "backwardation":
        slope = -0.006
    else:
        slope = 0.0

    log_price = np.log(spot) + slope * tenors + rng.normal(0, 0.005, months)
    prices = np.exp(log_price)
    dates = pd.date_range(start=pd.Timestamp.today().normalize() + pd.offsets.MonthBegin(1),
                          periods=months, freq="MS")
    return pd.DataFrame({"tenor_month": tenors, "expiry": dates, "price": prices})


# ---------------------------------------------------------------------------
# High-frequency signals
# ---------------------------------------------------------------------------
def generate_high_frequency(commodity_key: str, days: int = 120, seed: int = 11) -> pd.DataFrame:
    """Daily proxy series - vessel counts, refinery utilisation, weather etc."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    vessels = np.clip(70 + 5 * np.sin(np.arange(days) / 8) + rng.normal(0, 3, days), 40, 110)
    refinery_util = np.clip(88 + 2 * np.sin(np.arange(days) / 14) + rng.normal(0, 1.2, days), 75, 98)
    power_burn = np.clip(40 + 5 * np.sin(np.arange(days) / 7 + 0.4) + rng.normal(0, 1.5, days), 25, 60)
    weather_hdd = np.clip(15 + 10 * np.sin(np.arange(days) / 30) + rng.normal(0, 2, days), 0, 40)
    sat_prod = np.clip(tpl.base_supply * (1 + rng.normal(0, 0.005, days)), 0, None)

    return pd.DataFrame(
        {
            "date": idx,
            "vessels_tracked": vessels.astype(int),
            "refinery_util_pct": refinery_util,
            "power_burn": power_burn,
            "weather_hdd": weather_hdd,
            "sat_production_est": sat_prod,
        }
    ).set_index("date")


# ---------------------------------------------------------------------------
# Macro panel
# ---------------------------------------------------------------------------
def generate_macro_panel(seed: int = 19, months: int = 84) -> pd.DataFrame:
    """Generate a synthetic macro panel (GDP proxy, PMI, USD, rates)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.today().normalize().replace(day=1),
                        periods=months, freq="MS")
    gdp = 100 * np.cumprod(1 + rng.normal(0.0018, 0.003, months))
    pmi = np.clip(50 + 3 * np.sin(np.arange(months) / 8) + rng.normal(0, 1.2, months), 40, 60)
    usd = 100 * np.cumprod(1 + rng.normal(0.0005, 0.005, months))
    rates = np.clip(2 + 1.5 * np.sin(np.arange(months) / 24) + rng.normal(0, 0.2, months), 0, 7)
    return pd.DataFrame(
        {"date": idx, "gdp_index": gdp, "pmi": pmi, "usd_index": usd, "policy_rate": rates}
    ).set_index("date")


# ---------------------------------------------------------------------------
# Speculative positioning
# ---------------------------------------------------------------------------
def generate_positioning(commodity_key: str, weeks: int = 120, seed: int = 23) -> pd.DataFrame:
    """Synthetic CFTC-style positioning + open interest series."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=weeks, freq="W-TUE")

    trend = np.cumsum(rng.normal(0, 1, weeks))
    managed_money_net = 80_000 + 30_000 * np.sin(np.arange(weeks) / 12) + 6_000 * np.sign(trend)
    open_interest = 1_500_000 + 100_000 * np.sin(np.arange(weeks) / 24) + rng.normal(0, 20_000, weeks)
    cta_signal = np.tanh(np.gradient(managed_money_net) / 8_000)
    sentiment = np.clip(50 + 25 * np.tanh(managed_money_net / 80_000), 0, 100)

    return pd.DataFrame(
        {
            "date": idx,
            "managed_money_net": managed_money_net,
            "open_interest": open_interest,
            "cta_signal": cta_signal,
            "sentiment_score": sentiment,
        }
    ).set_index("date")
