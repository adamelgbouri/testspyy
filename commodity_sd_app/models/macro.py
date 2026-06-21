"""
Macroeconomic overlay - GDP, PMI, USD, rates.

Provides regression / correlation utilities aligning macro series with
commodity prices.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def align_macro(sd: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Inner-join macro panel onto monthly S&D frame on date index.

    Drops any S&D columns that are also in the macro panel (e.g. gdp_index)
    so macro values take precedence and the join doesn't error on duplicates.
    """
    macro = macro.copy()
    macro.index = pd.to_datetime(macro.index)
    sd = sd.drop(columns=[c for c in sd.columns if c in macro.columns], errors="ignore")
    return sd.join(macro, how="inner")


def correlation_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Compute a Pearson correlation matrix on the supplied columns."""
    available = [c for c in cols if c in df.columns]
    return df[available].corr().round(3)


def rolling_correlation(s1: pd.Series, s2: pd.Series, window: int = 24) -> pd.Series:
    """Rolling Pearson correlation between two aligned series."""
    return s1.rolling(window=window, min_periods=window // 2).corr(s2)


def regression_summary(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> Dict:
    """OLS-style summary using sklearn (no statsmodels dependency required)."""
    data = df[[y_col] + x_cols].dropna()
    if len(data) < len(x_cols) + 3:
        raise ValueError("Not enough data to fit regression")
    model = LinearRegression()
    model.fit(data[x_cols], data[y_col])
    pred = model.predict(data[x_cols])
    resid = data[y_col].to_numpy() - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((data[y_col] - data[y_col].mean()) ** 2))
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    return {
        "coefficients": dict(zip(x_cols, model.coef_)),
        "intercept": float(model.intercept_),
        "r_squared": float(r2),
        "n_obs": int(len(data)),
    }
