"""
Seasonality module.

Provides month-of-year profiles, rolling seasonal averages and a simple
classical decomposition into trend / seasonal / residual.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def monthly_profile(series: pd.Series, years: int = 5) -> pd.DataFrame:
    """
    Return per-month statistics over the last `years` of `series`.

    Columns: mean, std, min, max, current
    """
    s = series.dropna()
    cutoff = s.index.max() - pd.DateOffset(years=years)
    recent = s[s.index >= cutoff]
    grouped = recent.groupby(recent.index.month)
    df = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "std": grouped.std(),
            "min": grouped.min(),
            "max": grouped.max(),
        }
    )
    current_year = s.index.max().year
    current = s[s.index.year == current_year].groupby(s[s.index.year == current_year].index.month).mean()
    df["current"] = current
    df.index.name = "month"
    return df


def rolling_seasonal_average(series: pd.Series, window: int = 12) -> pd.Series:
    """Rolling mean over a 12-month default window."""
    return series.rolling(window=window, min_periods=max(3, window // 3)).mean()


def normalize_seasonal(series: pd.Series) -> pd.Series:
    """
    Strip seasonality by dividing each point by its month-of-year mean.

    Useful when comparing trend or running regression on price/demand.
    """
    monthly_mean = series.groupby(series.index.month).transform("mean")
    return series / monthly_mean


def decompose(series: pd.Series, period: int = 12) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Light-weight additive decomposition: trend / seasonal / residual.

    Uses statsmodels if available, otherwise falls back to a moving-average
    based implementation so the rest of the app keeps working without it.
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        s = series.dropna()
        if len(s) < 2 * period:
            raise ValueError("series too short")
        res = seasonal_decompose(s, model="additive", period=period, extrapolate_trend="freq")
        return res.trend, res.seasonal, res.resid
    except Exception:
        trend = series.rolling(period, min_periods=period // 2, center=True).mean()
        detrended = series - trend
        seasonal = detrended.groupby(detrended.index.month).transform("mean")
        residual = series - trend - seasonal
        return trend, seasonal, residual


def year_over_year_pivot(series: pd.Series) -> pd.DataFrame:
    """Pivot a monthly series into a year x month heatmap-ready frame."""
    s = series.dropna()
    pivot = pd.DataFrame({"year": s.index.year, "month": s.index.month, "value": s.values})
    return pivot.pivot_table(index="year", columns="month", values="value", aggfunc="mean")
