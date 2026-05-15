"""
Lagged response model.

Captures the delay between price signals and supply reaction, e.g. shale
drilling response, mining capex cycles, agricultural planting decisions.

Implements a simple distributed-lag regression on first-differences:

    supply_t = b0 + sum_{i=1..L} b_i * price_{t-i} + eps_t

Plus a convenience helper that *projects* a delayed supply response to a
price shock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class LaggedRegressionResult:
    coefficients: np.ndarray
    intercept: float
    r_squared: float
    lag: int


def fit_lagged_supply(
    df: pd.DataFrame,
    lag_months: int = 6,
    price_col: str = "price",
    supply_col: str = "supply",
) -> LaggedRegressionResult:
    """Fit distributed-lag regression of supply on lagged prices."""
    s = df[[price_col, supply_col]].dropna()
    if len(s) <= lag_months + 5:
        raise ValueError("Not enough observations for the requested lag")
    X = pd.concat([s[price_col].shift(i) for i in range(1, lag_months + 1)], axis=1)
    X.columns = [f"price_lag_{i}" for i in range(1, lag_months + 1)]
    y = s[supply_col]
    data = pd.concat([X, y], axis=1).dropna()
    model = LinearRegression()
    model.fit(data[X.columns], data[supply_col])
    r2 = model.score(data[X.columns], data[supply_col])
    return LaggedRegressionResult(
        coefficients=model.coef_,
        intercept=float(model.intercept_),
        r_squared=float(r2),
        lag=lag_months,
    )


def project_lagged_response(
    base_supply: float,
    price_shock_pct: float,
    lag_months: int,
    horizon: int = 24,
    response_strength: float = 0.08,
) -> pd.Series:
    """
    Return a synthetic supply response (% change vs base) over `horizon` months
    following a one-off price shock.

    The response is zero until the lag is hit, then approaches the long-run
    elasticity using a logistic curve.
    """
    months = np.arange(horizon)
    delayed = 1.0 / (1.0 + np.exp(-(months - lag_months) / 2.0))
    response_pct = response_strength * (price_shock_pct / 100.0) * delayed
    series = base_supply * (1.0 + response_pct)
    idx = pd.date_range(start=pd.Timestamp.today().normalize().replace(day=1),
                        periods=horizon, freq="MS")
    return pd.Series(series, index=idx, name="lagged_supply")
