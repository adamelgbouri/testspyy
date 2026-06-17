"""Monte Carlo engine for price + stocks under random shocks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .balance import BalanceAssumptions, run_balance
from .config import COMMODITY_TEMPLATES
from .data import get_sd_dataset
from .fair_value import estimate_fair_value


@dataclass
class MCConfig:
    n_paths: int = 500
    supply_sigma_pct: float = 1.5
    demand_sigma_pct: float = 1.2
    weather_sigma_pct: float = 1.0
    outage_prob: float = 0.05
    outage_size_pct: float = 4.0
    forecast_months: int = 18
    seed: int = 2024


def run_monte_carlo(commodity_key: str, cfg: MCConfig) -> Dict:
    """Run cfg.n_paths simulations and return distribution metrics."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    rng = np.random.default_rng(cfg.seed)

    df = get_sd_dataset(commodity_key, forecast_months=cfg.forecast_months)
    base_bal = run_balance(df, commodity_key,
                            BalanceAssumptions(forecast_months=cfg.forecast_months))
    fc_mask = base_bal["is_forecast"]
    n_fc = int(fc_mask.sum())

    end_stocks = np.empty(cfg.n_paths)
    avg_price = np.empty(cfg.n_paths)
    build_draw = np.empty(cfg.n_paths)
    paths_price = np.empty((cfg.n_paths, n_fc))

    for p in range(cfg.n_paths):
        a = BalanceAssumptions(
            supply_adj_pct=rng.normal(0.0, cfg.supply_sigma_pct),
            demand_adj_pct=rng.normal(0.0, cfg.demand_sigma_pct),
            weather_pct=rng.normal(0.0, cfg.weather_sigma_pct),
            forecast_months=cfg.forecast_months,
        )
        bal = run_balance(df, commodity_key, a)
        # Outage event
        if rng.random() < cfg.outage_prob * n_fc:
            month = int(rng.integers(0, max(n_fc, 1)))
            bal.iloc[-(n_fc - month):, bal.columns.get_loc("supply")] *= (
                1.0 - cfg.outage_size_pct / 100.0
            )
        bal = estimate_fair_value(bal)
        end_stocks[p] = float(bal["stocks_model"].iloc[-1])
        avg_price[p] = float(bal.loc[fc_mask, "fair_value_price"].mean())
        build_draw[p] = float(bal.loc[fc_mask, "build_draw"].sum())
        paths_price[p, :] = bal.loc[fc_mask, "fair_value_price"].to_numpy()

    p5_price = np.quantile(paths_price, 0.05, axis=0)
    p50_price = np.quantile(paths_price, 0.50, axis=0)
    p95_price = np.quantile(paths_price, 0.95, axis=0)
    fc_dates = [str(d.date()) for d in base_bal.index[fc_mask]]

    return {
        "key": commodity_key, "name": tpl.name,
        "price_unit": tpl.price_unit, "inventory_unit": tpl.inventory_unit,
        "n_paths": cfg.n_paths,
        "median_price": float(np.median(avg_price)),
        "p5_price_avg": float(np.quantile(avg_price, 0.05)),
        "p95_price_avg": float(np.quantile(avg_price, 0.95)),
        "median_end_stocks": float(np.median(end_stocks)),
        "var_95_price_drop": float(np.median(avg_price) - np.quantile(avg_price, 0.05)),
        "histogram_price": _binned(avg_price, 30),
        "histogram_stocks": _binned(end_stocks, 30),
        "fan_chart": [
            {"date": d, "p5": float(p5), "p50": float(p50), "p95": float(p95)}
            for d, p5, p50, p95 in zip(fc_dates, p5_price, p50_price, p95_price)
        ],
    }


def _binned(arr: np.ndarray, n_bins: int = 30) -> List[Dict]:
    counts, edges = np.histogram(arr, bins=n_bins)
    return [
        {"x": float((edges[i] + edges[i + 1]) / 2), "count": int(counts[i])}
        for i in range(len(counts))
    ]
