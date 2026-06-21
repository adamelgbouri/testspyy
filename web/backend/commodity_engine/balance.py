"""S&D balance engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import COMMODITY_TEMPLATES, CommodityTemplate


@dataclass
class BalanceAssumptions:
    """User-tunable assumptions applied to the forecast portion of the dataset."""

    beginning_stocks: Optional[float] = None
    supply_adj_pct: float = 0.0
    demand_adj_pct: float = 0.0
    imports_adj_pct: float = 0.0
    exports_adj_pct: float = 0.0
    weather_pct: float = 0.0
    gdp_growth_pct: float = 2.5
    storage_capacity: Optional[float] = None
    forecast_months: int = 24


def _apply_adjustments(df: pd.DataFrame, tpl: CommodityTemplate,
                       a: BalanceAssumptions) -> pd.DataFrame:
    out = df.copy()
    mask = out["is_forecast"]
    out.loc[mask, "supply"] *= 1 + a.supply_adj_pct / 100.0
    demand_mult = (
        1 + a.demand_adj_pct / 100.0
        + a.weather_pct / 100.0
        + (a.gdp_growth_pct - 2.5) / 100.0 * 0.6
    )
    out.loc[mask, "demand"] *= demand_mult
    out.loc[mask, "imports"] *= 1 + a.imports_adj_pct / 100.0
    out.loc[mask, "exports"] *= 1 + a.exports_adj_pct / 100.0
    return out


def run_balance(df: pd.DataFrame, commodity_key: str,
                assumptions: Optional[BalanceAssumptions] = None) -> pd.DataFrame:
    """Run the S&D balance identity, returning an augmented frame."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    a = assumptions or BalanceAssumptions()
    adj = _apply_adjustments(df, tpl, a)

    days = adj.index.days_in_month.to_numpy()
    mul = days if tpl.unit.endswith("/d") else np.ones_like(days)
    supply_total = adj["supply"].to_numpy() * mul
    demand_total = adj["demand"].to_numpy() * mul
    net_trade = adj["imports"].to_numpy() - adj["exports"].to_numpy()

    start = (a.beginning_stocks if a.beginning_stocks is not None
             else float(adj["stocks"].iloc[0]))
    storage_cap = a.storage_capacity or tpl.storage_capacity
    soft_cap = storage_cap * 1.3

    stocks = np.empty(len(adj))
    stocks[0] = float(np.clip(start, 0.0, soft_cap))
    for i in range(1, len(adj)):
        delta = supply_total[i] - demand_total[i] + (net_trade[i] - net_trade.mean()) * 0.05
        stocks[i] = float(np.clip(stocks[i - 1] + delta, 0.0, soft_cap))

    avg_daily = np.where(demand_total > 0, demand_total / days, 1e-6)
    days_cover = stocks / avg_daily

    out = adj.copy()
    out["build_draw"] = supply_total - demand_total
    out["stocks_model"] = stocks
    out["days_cover_model"] = days_cover
    out["surplus_deficit"] = supply_total - demand_total
    out["capacity_pct"] = (stocks / storage_cap) * 100.0
    return out
