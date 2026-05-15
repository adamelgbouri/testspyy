"""
Fair-value price model.

Maps inventory level (specifically days-of-cover) to a fair-value price using
a log-linear regression on the historical sample.  Provides over/undervalued
signals against observed price.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.config import COMMODITY_TEMPLATES


def _dc_col(df: pd.DataFrame) -> str:
    """Prefer the model's recomputed days_cover when available."""
    return "days_cover_model" if "days_cover_model" in df.columns else "days_cover"


def fit_fair_value(df: pd.DataFrame) -> Dict:
    """Fit log(price) ~ days_cover regression on history only."""
    dc = _dc_col(df)
    hist = df[~df["is_forecast"]].dropna(subset=["price", dc])
    X = hist[[dc]].to_numpy()
    y = np.log(hist["price"].to_numpy())
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    return {
        "intercept": float(model.intercept_),
        "slope": float(model.coef_[0]),
        "r_squared": float(r2),
        "mean_dc": float(hist[dc].mean()),
        "mean_price": float(hist["price"].mean()),
    }


def estimate_fair_value(df: pd.DataFrame, commodity_key: str) -> pd.DataFrame:
    """
    Return original frame augmented with fair_value_price, residual, signal.

    `signal` is 'Overvalued' / 'Undervalued' / 'Fair' based on +/-10% bands.
    """
    tpl = COMMODITY_TEMPLATES[commodity_key]
    dc = _dc_col(df)
    try:
        fit = fit_fair_value(df)
        log_fv = fit["intercept"] + fit["slope"] * df[dc]
        fv = np.exp(log_fv)
    except Exception:
        # fallback: price band midpoint scaled by inverse of days_cover
        lo, hi = tpl.price_band
        mid = (lo + hi) / 2
        fv = mid * (1 - 0.15 * (df[dc] - df[dc].mean())
                    / max(df[dc].std(), 1e-6))

    out = df.copy()
    out["fair_value_price"] = fv
    out["fv_residual_pct"] = (out["price"] - fv) / fv * 100.0
    out["fv_signal"] = np.where(
        out["fv_residual_pct"] > 10, "Overvalued",
        np.where(out["fv_residual_pct"] < -10, "Undervalued", "Fair"),
    )
    return out


def marginal_cost_curve(commodity_key: str, n_quantiles: int = 10) -> pd.DataFrame:
    """
    Synthetic marginal cost curve (cost stack) using a logistic shape.

    Top quantile of producers has costs near the lower price band, bottom
    quantile near the upper band.
    """
    tpl = COMMODITY_TEMPLATES[commodity_key]
    lo, hi = tpl.price_band
    q = np.linspace(0.02, 0.98, n_quantiles)
    costs = lo + (hi - lo) * q ** 1.4
    cum_share = q * 100.0
    return pd.DataFrame({"cum_share_pct": cum_share, "marginal_cost": costs})
