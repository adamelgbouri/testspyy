"""
Supply & Demand balance engine.

Implements the fundamental balance identity:

    Ending Stocks = Beginning Stocks + Supply - Demand

with optional adjustments (imports/exports, refinery runs, weather, GDP).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from utils.config import COMMODITY_TEMPLATES, CommodityTemplate


Frequency = Literal["M", "Q", "Y"]


@dataclass
class BalanceAssumptions:
    """User-tunable assumptions for the balance engine."""

    beginning_stocks: Optional[float] = None
    supply_adj_pct: float = 0.0            # +/- % applied to supply
    demand_adj_pct: float = 0.0            # +/- % applied to demand
    imports_adj_pct: float = 0.0
    exports_adj_pct: float = 0.0
    refinery_runs_pct: float = 0.0         # oil-specific demand booster
    weather_pct: float = 0.0               # weather demand booster
    gdp_growth_pct: float = 0.0            # demand booster via GDP
    storage_capacity: Optional[float] = None
    forecast_months: int = 24
    extra: Dict[str, float] = field(default_factory=dict)


def _apply_adjustments(df: pd.DataFrame, tpl: CommodityTemplate,
                       a: BalanceAssumptions) -> pd.DataFrame:
    """Return a new frame with the assumption deltas applied to history+forecast."""
    out = df.copy()
    # Adjustments only impact the forecast portion to preserve historic truth.
    mask = out["is_forecast"]
    out.loc[mask, "supply"] *= 1 + a.supply_adj_pct / 100.0
    base_demand_mult = (
        1
        + a.demand_adj_pct / 100.0
        + a.weather_pct / 100.0
        + (a.gdp_growth_pct - 2.5) / 100.0 * 0.6
    )
    out.loc[mask, "demand"] *= base_demand_mult

    if tpl.key == "crude_oil":
        # Refinery runs proxy for demand-side of the chain
        out.loc[mask, "demand"] *= 1 + a.refinery_runs_pct / 100.0 * 0.4

    out.loc[mask, "imports"] *= 1 + a.imports_adj_pct / 100.0
    out.loc[mask, "exports"] *= 1 + a.exports_adj_pct / 100.0
    return out


def run_balance(
    df: pd.DataFrame,
    commodity_key: str,
    assumptions: Optional[BalanceAssumptions] = None,
    frequency: Frequency = "M",
) -> pd.DataFrame:
    """
    Run the balance engine over a S&D dataframe.

    Returns a frame with: supply, demand, net_trade, build_draw, stocks,
    days_cover, surplus_deficit, capacity_pct.
    """
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

    # Build inventory from balance identity
    start = (
        a.beginning_stocks
        if a.beginning_stocks is not None
        else float(adj["stocks"].iloc[0])
    )
    stocks = np.empty(len(adj))
    stocks[0] = start
    for i in range(1, len(adj)):
        delta = supply_total[i] - demand_total[i] + (net_trade[i] - net_trade.mean()) * 0.05
        stocks[i] = max(stocks[i - 1] + delta, 0.0)

    storage_cap = a.storage_capacity or tpl.storage_capacity
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
    if frequency == "Q":
        return _resample(out, "QE")
    if frequency == "Y":
        return _resample(out, "YE")
    raise ValueError(f"Unknown frequency '{frequency}'")


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate balance frame to quarterly/yearly granularity."""
    agg: Dict[str, str] = {
        "supply": "mean",
        "demand": "mean",
        "imports": "mean",
        "exports": "mean",
        "supply_adj": "mean",
        "demand_adj": "mean",
        "net_trade": "mean",
        "build_draw": "sum",
        "stocks_model": "last",
        "days_cover_model": "mean",
        "surplus_deficit": "sum",
        "capacity_pct": "last",
        "price": "mean",
        "stocks": "last",
        "days_cover": "mean",
        "gdp_index": "last",
        "weather_index": "mean",
        "refinery_runs": "mean",
        "is_forecast": "max",
    }
    cols = [c for c in agg if c in df.columns]
    return df[cols].resample(rule).agg({c: agg[c] for c in cols})
