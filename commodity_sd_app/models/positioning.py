"""
Financial market overlay - speculative positioning and CTA flows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def positioning_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return rolling means + z-scores for the positioning panel."""
    out = df.copy()
    out["mm_net_4w_avg"] = out["managed_money_net"].rolling(4).mean()
    out["oi_4w_avg"] = out["open_interest"].rolling(4).mean()
    z = (out["managed_money_net"] - out["managed_money_net"].rolling(52).mean()) / \
        out["managed_money_net"].rolling(52).std()
    out["mm_z_score"] = z
    return out


def sentiment_label(score: float) -> str:
    """Map sentiment score in [0,100] to a label."""
    if score >= 75:
        return "Bullish"
    if score >= 55:
        return "Mildly Bullish"
    if score >= 45:
        return "Neutral"
    if score >= 25:
        return "Mildly Bearish"
    return "Bearish"


def positioning_heatmap(df: pd.DataFrame, n_weeks: int = 52) -> pd.DataFrame:
    """Pivot recent positioning + sentiment into a small heatmap matrix."""
    recent = df.tail(n_weeks).copy()
    recent["week"] = recent.index.isocalendar().week
    recent["year"] = recent.index.isocalendar().year
    pivot = recent.pivot_table(
        index="year", columns="week", values="sentiment_score", aggfunc="last"
    )
    return pivot
