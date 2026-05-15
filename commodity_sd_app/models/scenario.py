"""
Scenario engine - Bull / Base / Bear.

Wraps the balance engine so we can run multiple assumption sets in parallel,
weight them by probability and compute aggregate metrics.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from models.balance import BalanceAssumptions, run_balance
from models.fair_value import estimate_fair_value
from utils.config import COMMODITY_TEMPLATES, SCENARIO_PRESETS


def build_assumptions_from_preset(name: str, base: BalanceAssumptions) -> BalanceAssumptions:
    """Create a BalanceAssumptions object from a SCENARIO_PRESETS entry."""
    preset = SCENARIO_PRESETS[name]
    return BalanceAssumptions(
        beginning_stocks=base.beginning_stocks,
        supply_adj_pct=preset["supply_shock_pct"],
        demand_adj_pct=preset["demand_shock_pct"],
        imports_adj_pct=0.0,
        exports_adj_pct=0.0,
        refinery_runs_pct=0.0,
        weather_pct=preset["weather_shock_pct"],
        gdp_growth_pct=preset["gdp_growth_pct"],
        storage_capacity=base.storage_capacity,
        forecast_months=base.forecast_months,
        extra={"fx_usd_pct": preset["fx_usd_pct"]},
    )


def run_scenarios(
    df: pd.DataFrame,
    commodity_key: str,
    base_assumptions: BalanceAssumptions,
    scenarios: Iterable[str] = ("Bull", "Base", "Bear"),
) -> Dict[str, pd.DataFrame]:
    """Run a balance projection per scenario and return a dict of frames."""
    results: Dict[str, pd.DataFrame] = {}
    for name in scenarios:
        a = build_assumptions_from_preset(name, base_assumptions)
        bal = run_balance(df, commodity_key, a)
        bal["fair_value_price"] = estimate_fair_value(bal, commodity_key)["fair_value_price"]
        results[name] = bal
    return results


def scenario_summary(results: Dict[str, pd.DataFrame], commodity_key: str) -> pd.DataFrame:
    """Compress per-scenario frames into a single comparison summary table."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    rows: List[dict] = []
    for name, bal in results.items():
        last = bal.iloc[-1]
        rows.append(
            {
                "Scenario": name,
                "Probability": SCENARIO_PRESETS[name]["probability"],
                "End Stocks": last["stocks_model"],
                "Days Cover": last["days_cover_model"],
                "Build/Draw (last 12M)": bal["build_draw"].iloc[-12:].sum(),
                "Avg Price (forecast)": bal.loc[bal["is_forecast"], "price"].mean(),
                "Fair Value (end)": last["fair_value_price"],
                "Unit": tpl.inventory_unit,
            }
        )
    return pd.DataFrame(rows).set_index("Scenario")


def probability_weighted_price(results: Dict[str, pd.DataFrame]) -> float:
    """Expected forecast-period price using preset probabilities."""
    pw = 0.0
    for name, bal in results.items():
        prob = SCENARIO_PRESETS[name]["probability"]
        pw += prob * bal.loc[bal["is_forecast"], "price"].mean()
    return float(pw)
