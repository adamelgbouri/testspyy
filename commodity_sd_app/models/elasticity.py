"""
Price elasticity model.

Implements simple linear supply / demand schedules of the form:

    Demand(P) = D0 * (1 - alpha * (P - P0) / P0)
    Supply(P) = S0 * (1 + beta  * (P - P0) / P0)

with optional lag effects handled via shift in the calling code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.config import COMMODITY_TEMPLATES


@dataclass
class ElasticityParams:
    alpha: float = 0.06   # demand elasticity (positive: higher price -> lower demand)
    beta: float = 0.10    # supply elasticity (positive: higher price -> more supply)
    base_price: float = 0.0
    d0: float = 0.0
    s0: float = 0.0


def demand_curve(prices: np.ndarray, p: ElasticityParams) -> np.ndarray:
    return p.d0 * (1.0 - p.alpha * (prices - p.base_price) / p.base_price)


def supply_curve(prices: np.ndarray, p: ElasticityParams) -> np.ndarray:
    return p.s0 * (1.0 + p.beta * (prices - p.base_price) / p.base_price)


def equilibrium(p: ElasticityParams) -> tuple[float, float]:
    """
    Analytical equilibrium price/quantity for the linear schedule.

    Solving Demand(P) = Supply(P):
        d0 (1 - alpha (P-P0)/P0) = s0 (1 + beta (P-P0)/P0)
    -> P = P0 * (1 + (d0 - s0) / (d0*alpha + s0*beta))
    """
    denom = p.d0 * p.alpha + p.s0 * p.beta
    if abs(denom) < 1e-12:
        return p.base_price, p.d0
    p_eq = p.base_price * (1.0 + (p.d0 - p.s0) / denom)
    q_eq = demand_curve(np.array([p_eq]), p)[0]
    return float(p_eq), float(q_eq)


def build_curves(commodity_key: str, alpha: float, beta: float,
                 price_band_pct: float = 0.5, n: int = 80) -> pd.DataFrame:
    """Return a DataFrame of price, demand, supply for plotting."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    p = ElasticityParams(alpha=alpha, beta=beta, base_price=tpl.base_price,
                         d0=tpl.base_demand, s0=tpl.base_supply)
    p_min = tpl.base_price * (1 - price_band_pct)
    p_max = tpl.base_price * (1 + price_band_pct)
    prices = np.linspace(p_min, p_max, n)
    df = pd.DataFrame(
        {
            "price": prices,
            "demand": demand_curve(prices, p),
            "supply": supply_curve(prices, p),
        }
    )
    df["params"] = "alpha=%.3f, beta=%.3f" % (alpha, beta)
    return df
