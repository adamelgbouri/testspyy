"""
Futures curve / term structure module.

Includes:
- spread analysis (m1-m2, m1-m12)
- storage economics check (carry vs contango slope)
- inventory-to-curve relationship
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def classify_structure(curve: pd.DataFrame) -> str:
    """Return 'Contango', 'Backwardation' or 'Mixed' based on slope."""
    if len(curve) < 2:
        return "Mixed"
    prices = curve["price"].to_numpy()
    diffs = np.diff(prices)
    if np.all(diffs > 0):
        return "Contango"
    if np.all(diffs < 0):
        return "Backwardation"
    avg_slope = diffs.mean()
    if avg_slope > 0:
        return "Contango (Mixed)"
    if avg_slope < 0:
        return "Backwardation (Mixed)"
    return "Flat"


def calendar_spreads(curve: pd.DataFrame) -> pd.DataFrame:
    """Return key calendar spreads: m1-m2, m1-m6, m1-m12, m6-m12."""
    p = curve.set_index("tenor_month")["price"]

    def diff(a: int, b: int) -> float:
        if a in p.index and b in p.index:
            return float(p.loc[a] - p.loc[b])
        return float("nan")

    rows = [
        ("m1 - m2", diff(1, 2)),
        ("m1 - m6", diff(1, 6)),
        ("m1 - m12", diff(1, 12)),
        ("m6 - m12", diff(6, 12)),
    ]
    return pd.DataFrame(rows, columns=["Spread", "Value"]).set_index("Spread")


def storage_economics(curve: pd.DataFrame, storage_cost_per_month: float,
                      financing_rate_pct: float) -> pd.DataFrame:
    """
    Check whether contango covers carry.

    For each tenor t > 1 compare (price_t - price_1) against the cumulative
    cost of storing one unit from m1 to t (storage + financing).
    """
    p = curve.set_index("tenor_month")["price"]
    base = float(p.iloc[0])
    out = curve.copy().set_index("tenor_month")
    out["carry"] = (storage_cost_per_month
                    + (financing_rate_pct / 100.0 / 12.0) * base) * out.index
    out["contango_premium"] = out["price"] - base
    out["positive_carry"] = out["contango_premium"] > out["carry"]
    return out.reset_index()


def inventory_curve_relationship(curve: pd.DataFrame, days_cover: float) -> Tuple[str, float]:
    """
    Heuristic inventory ↔ curve relationship.

    Very low inventory -> backwardation.  Very high -> contango.
    Returns a label and a "tightness score" in [-1, 1] (positive = tight).
    """
    structure = classify_structure(curve)
    # tightness score from days_cover relative to typical band
    z = (35 - days_cover) / 20.0
    score = float(np.clip(z, -1.0, 1.0))
    if score > 0.3:
        label = "Tight (supports backwardation)"
    elif score < -0.3:
        label = "Loose (supports contango)"
    else:
        label = "Balanced"
    return f"{label} - observed: {structure}", score
