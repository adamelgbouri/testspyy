"""Inventory-to-price fair-value regression."""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_fair_value(df: pd.DataFrame) -> Dict:
    """Fit log(price) ~ days_cover on history only."""
    dc_col = "days_cover_model" if "days_cover_model" in df.columns else "days_cover"
    hist = df[~df["is_forecast"]].dropna(subset=["price", dc_col])
    X = hist[[dc_col]].to_numpy()
    y = np.log(hist["price"].to_numpy())
    model = LinearRegression().fit(X, y)
    return {
        "intercept": float(model.intercept_),
        "slope": float(model.coef_[0]),
        "r_squared": float(model.score(X, y)),
    }


def estimate_fair_value(df: pd.DataFrame) -> pd.DataFrame:
    """Augment frame with fair_value_price + signal columns."""
    dc_col = "days_cover_model" if "days_cover_model" in df.columns else "days_cover"
    try:
        fit = fit_fair_value(df)
        fv = np.exp(fit["intercept"] + fit["slope"] * df[dc_col])
    except Exception:
        fv = df["price"]
    out = df.copy()
    out["fair_value_price"] = fv
    out["fv_residual_pct"] = (out["price"] - fv) / fv * 100.0
    out["fv_signal"] = np.where(
        out["fv_residual_pct"] > 10, "overvalued",
        np.where(out["fv_residual_pct"] < -10, "undervalued", "fair"),
    )
    return out
