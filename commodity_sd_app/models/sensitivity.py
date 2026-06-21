"""
Sensitivity analysis.

One-variable tornado charts and multi-variable stress matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from models.balance import BalanceAssumptions, run_balance


@dataclass
class SensitivityVar:
    name: str             # human label
    attr: str             # attribute of BalanceAssumptions to perturb
    low: float            # low value (absolute)
    high: float           # high value (absolute)


def _eval(df: pd.DataFrame, commodity_key: str, a: BalanceAssumptions, metric: str) -> float:
    bal = run_balance(df, commodity_key, a)
    if metric == "end_stocks":
        return float(bal["stocks_model"].iloc[-1])
    if metric == "avg_fc_price":
        return float(bal.loc[bal["is_forecast"], "price"].mean())
    if metric == "build_draw_sum":
        return float(bal["build_draw"].iloc[-12:].sum())
    raise ValueError(f"Unknown metric: {metric}")


def tornado(
    df: pd.DataFrame,
    commodity_key: str,
    base: BalanceAssumptions,
    variables: Iterable[SensitivityVar],
    metric: str = "end_stocks",
) -> pd.DataFrame:
    """Build a tornado dataframe with low/base/high values per variable."""
    base_val = _eval(df, commodity_key, base, metric)
    rows: List[dict] = []
    for v in variables:
        a_low = replace(base, **{v.attr: v.low})
        a_high = replace(base, **{v.attr: v.high})
        low_val = _eval(df, commodity_key, a_low, metric)
        high_val = _eval(df, commodity_key, a_high, metric)
        rows.append(
            {
                "variable": v.name,
                "low": low_val - base_val,
                "high": high_val - base_val,
                "low_input": v.low,
                "high_input": v.high,
            }
        )
    out = pd.DataFrame(rows)
    out["range"] = out["high"].abs() + out["low"].abs()
    return out.sort_values("range", ascending=True)


def stress_matrix(
    df: pd.DataFrame,
    commodity_key: str,
    base: BalanceAssumptions,
    var_a: SensitivityVar,
    var_b: SensitivityVar,
    grid: int = 5,
    metric: str = "end_stocks",
) -> pd.DataFrame:
    """2D stress matrix - useful for spider/heatmap visualisations."""
    a_grid = np.linspace(var_a.low, var_a.high, grid)
    b_grid = np.linspace(var_b.low, var_b.high, grid)
    out = pd.DataFrame(index=a_grid, columns=b_grid, dtype=float)
    out.index.name = var_a.name
    out.columns.name = var_b.name
    for av in a_grid:
        for bv in b_grid:
            a = replace(base, **{var_a.attr: av, var_b.attr: bv})
            out.loc[av, bv] = _eval(df, commodity_key, a, metric)
    return out
