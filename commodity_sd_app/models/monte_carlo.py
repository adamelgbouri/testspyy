"""
Monte Carlo / probabilistic engine.

Wraps the balance engine with randomised supply / demand / weather / outage
shocks and reports the distribution of end-of-horizon inventory, price and
build/draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from models.balance import BalanceAssumptions, run_balance
from models.fair_value import estimate_fair_value
from utils.config import COMMODITY_TEMPLATES


@dataclass
class MCConfig:
    n_paths: int = 500
    supply_sigma_pct: float = 1.5
    demand_sigma_pct: float = 1.2
    weather_sigma_pct: float = 1.0
    outage_prob: float = 0.05      # per month
    outage_size_pct: float = 4.0
    seed: int = 2024


def run_monte_carlo(
    df: pd.DataFrame,
    commodity_key: str,
    base_assumptions: BalanceAssumptions,
    cfg: MCConfig,
) -> Dict[str, pd.DataFrame | np.ndarray]:
    """
    Run n_paths simulations and return distribution metrics.

    Output dict keys:
        end_stocks: ndarray (n_paths,)
        avg_price: ndarray (n_paths,)
        build_draw: ndarray (n_paths,)
        paths_price: DataFrame (date x path) of forecast-horizon price paths
        paths_stocks: DataFrame (date x path) of forecast-horizon stocks
        percentiles: DataFrame with p5/p50/p95 of price and stocks
    """
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
            supply_adj_pct=base_assumptions.supply_adj_pct
            + rng.normal(0.0, cfg.supply_sigma_pct),
            demand_adj_pct=base_assumptions.demand_adj_pct
            + rng.normal(0.0, cfg.demand_sigma_pct),
            imports_adj_pct=base_assumptions.imports_adj_pct,
            exports_adj_pct=base_assumptions.exports_adj_pct,
            refinery_runs_pct=base_assumptions.refinery_runs_pct,
            weather_pct=base_assumptions.weather_pct
            + rng.normal(0.0, cfg.weather_sigma_pct),
            gdp_growth_pct=base_assumptions.gdp_growth_pct,
            storage_capacity=base_assumptions.storage_capacity,
            forecast_months=base_assumptions.forecast_months,
        )
        # apply random outage events to supply
        bal = run_balance(df, commodity_key, a)
        if rng.random() < cfg.outage_prob * n_fc:
            month = rng.integers(0, max(n_fc, 1))
            bal.iloc[-(n_fc - month):, bal.columns.get_loc("supply_adj")] *= (
                1.0 - cfg.outage_size_pct / 100.0
            )
            # rebuild stocks crudely for this path
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

    pct = pd.DataFrame(
        {
            "p5_price": pp.quantile(0.05, axis=1),
            "p50_price": pp.quantile(0.50, axis=1),
            "p95_price": pp.quantile(0.95, axis=1),
            "p5_stocks": ps.quantile(0.05, axis=1),
            "p50_stocks": ps.quantile(0.50, axis=1),
            "p95_stocks": ps.quantile(0.95, axis=1),
        }
    )

    return {
        "end_stocks": end_stocks,
        "avg_price": avg_price,
        "build_draw": build_draw,
        "paths_price": pp,
        "paths_stocks": ps,
        "percentiles": pct,
    }


def value_at_risk(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Simple VaR_α on a vector of P&L-style losses (positive = loss)."""
    return float(np.quantile(losses, alpha))
