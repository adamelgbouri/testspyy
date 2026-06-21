"""
Inventory / storage model.

Builds inventory trajectories with optional capacity constraints and a
simple floating-storage spill-over rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils.config import COMMODITY_TEMPLATES


@dataclass
class StorageConfig:
    capacity: Optional[float] = None
    floating_buffer_pct: float = 5.0    # % of capacity available as floating storage
    allow_negative: bool = False


def project_inventory(
    df: pd.DataFrame,
    commodity_key: str,
    config: Optional[StorageConfig] = None,
) -> pd.DataFrame:
    """
    Given a balance frame with build_draw column, project inventory with caps.

    Returns the same frame with extra columns:
        stocks_capped, overflow_floating, utilization_pct.
    """
    tpl = COMMODITY_TEMPLATES[commodity_key]
    cfg = config or StorageConfig()
    cap = cfg.capacity or tpl.storage_capacity
    floating_cap = cap * (cfg.floating_buffer_pct / 100.0)

    out = df.copy()
    stocks = np.empty(len(out))
    floating = np.zeros(len(out))
    stocks[0] = float(out["stocks_model"].iloc[0])

    bd = out["build_draw"].to_numpy()
    for i in range(1, len(out)):
        candidate = stocks[i - 1] + bd[i]
        if not cfg.allow_negative:
            candidate = max(candidate, 0.0)
        if candidate > cap:
            spillover = candidate - cap
            floating[i] = min(spillover, floating_cap)
            stocks[i] = cap
        else:
            stocks[i] = candidate

    out["stocks_capped"] = stocks
    out["overflow_floating"] = floating
    out["utilization_pct"] = (stocks / cap) * 100.0
    return out


def draw_build_waterfall(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Return the last `n` monthly inventory deltas, signed for waterfall plot."""
    s = df["build_draw"].iloc[-n:]
    return pd.DataFrame({"period": s.index.strftime("%b-%y"), "delta": s.values})


def days_of_forward_cover(df: pd.DataFrame, horizon_days: int = 30) -> pd.Series:
    """Forward looking days-of-cover using the next-month demand projection."""
    forward = df["demand"].rolling(window=2, min_periods=1).mean()
    days = df.index.days_in_month
    daily = forward / days
    cover = df["stocks_model"] / daily.replace(0, np.nan)
    cover = cover.ffill()
    return cover.clip(lower=0)
