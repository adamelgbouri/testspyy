"""Risk metrics: parametric / historical VaR, CVaR, stress tests."""
from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import COMMODITY_TEMPLATES
from .data import get_sd_dataset


def _price_series(commodity_key: str, lookback: int = 60) -> pd.Series:
    df = get_sd_dataset(commodity_key, forecast_months=6)
    return df["price"].tail(lookback).reset_index(drop=True)


def parametric_var(prices: pd.Series, qty: float, confidence: float = 0.95,
                   horizon_days: int = 1) -> Dict[str, float]:
    rets = prices.pct_change().dropna()
    if rets.empty:
        return {"var": float("nan"), "cvar": float("nan"), "vol": float("nan")}
    mu = float(rets.mean()) * horizon_days
    sigma = float(rets.std()) * np.sqrt(horizon_days)
    z = float(norm.ppf(confidence))
    var_pct = -(mu - z * sigma)
    cvar_pct = -(mu - sigma * norm.pdf(z) / (1 - confidence))
    notional = qty * float(prices.iloc[-1])
    return {
        "var": var_pct * notional, "cvar": cvar_pct * notional,
        "vol_pct": sigma * 100,
        "var_pct": var_pct * 100, "cvar_pct": cvar_pct * 100,
    }


def historical_var(prices: pd.Series, qty: float,
                   confidence: float = 0.95) -> Dict[str, float]:
    rets = prices.pct_change().dropna()
    if rets.empty:
        return {"var": float("nan"), "cvar": float("nan")}
    q = float(rets.quantile(1 - confidence))
    tail = rets[rets <= q]
    cvar_pct = float(-tail.mean()) if not tail.empty else -q
    notional = qty * float(prices.iloc[-1])
    return {
        "var": -q * notional, "cvar": cvar_pct * notional,
        "var_pct": -q * 100, "cvar_pct": cvar_pct * 100,
    }


def stress_scenarios(price: float, qty: float, direction: str = "Long"
                     ) -> List[Dict]:
    sign = 1 if direction == "Long" else -1
    scenarios = [
        ("Mild correction (−5%)", -0.05),
        ("Sharp drop (−10%)", -0.10),
        ("Crash (−20%)", -0.20),
        ("Black swan (−35%)", -0.35),
        ("Rally (+10%)", +0.10),
        ("Squeeze (+25%)", +0.25),
    ]
    return [
        {"scenario": label, "shock_pct": shock * 100,
         "new_price": price * (1 + shock),
         "pnl_impact": sign * (price * (1 + shock) - price) * qty}
        for label, shock in scenarios
    ]


def portfolio_var(positions: List[Dict], confidence: float = 0.95,
                  horizon_days: int = 1) -> Dict:
    """
    Sum-of-individual-VaR approximation (conservative, ignores correlation).
    Each position dict needs: commodity_key, quantity, direction.
    """
    rows = []
    total_var = 0.0
    total_cvar = 0.0
    for p in positions:
        ck = p["commodity_key"]
        if ck not in COMMODITY_TEMPLATES:
            continue
        tpl = COMMODITY_TEMPLATES[ck]
        prices = _price_series(ck)
        par = parametric_var(prices, p["quantity"], confidence, horizon_days)
        rows.append({
            "commodity": tpl.name, "sector": tpl.sector,
            "direction": p["direction"], "quantity": p["quantity"],
            "vol_pct": par["vol_pct"],
            "var": par["var"], "cvar": par["cvar"],
        })
        total_var += par["var"]
        total_cvar += par["cvar"]
    return {
        "rows": rows, "total_var": total_var, "total_cvar": total_cvar,
        "confidence": confidence, "horizon_days": horizon_days,
    }
