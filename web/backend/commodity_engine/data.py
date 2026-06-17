"""Data layer - synthetic generators + live Yahoo fetch with fallback."""
from __future__ import annotations
import logging
import os
from datetime import date, datetime
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import COMMODITY_TEMPLATES

logger = logging.getLogger(__name__)
_MONTH_CODES = list("FGHJKMNQUVXZ")
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@lru_cache(maxsize=64)
def get_sd_dataset(commodity_key: str, start: str = "2018-01-01",
                   forecast_months: int = 24, seed: int = 42) -> pd.DataFrame:
    """Monthly synthetic S&D series, deterministic in (key, start, horizon, seed)."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    end = pd.Timestamp(date.today()).strftime("%Y-%m-01")
    idx = pd.date_range(start=start, end=end, freq="MS")
    horizon = pd.date_range(
        start=idx[-1] + pd.offsets.MonthBegin(1),
        periods=forecast_months, freq="MS",
    )
    full_idx = idx.append(horizon)

    rng = np.random.default_rng(seed)
    n = len(full_idx)
    months = full_idx.month - 1
    seas_d = np.array(tpl.seasonal_demand)[months]
    seas_s = np.array(tpl.seasonal_supply)[months]

    t = np.linspace(0, 1, n)
    demand_trend = tpl.base_demand * (1 + 0.010 * t * 6)
    supply_trend = tpl.base_supply * (1 + 0.012 * t * 6)
    cycle = 0.02 * np.sin(2 * np.pi * np.arange(n) / 48)
    demand = demand_trend * seas_d * (1 + cycle + rng.normal(0, 0.012, n))
    supply = supply_trend * seas_s * (1 + cycle * 0.5 + rng.normal(0, 0.015, n))

    imports = np.abs(rng.normal(tpl.base_demand * 0.20, tpl.base_demand * 0.02, n))
    exports = np.abs(rng.normal(tpl.base_demand * 0.18, tpl.base_demand * 0.02, n))

    days_in_month = full_idx.days_in_month.to_numpy()
    if tpl.unit.endswith("/d"):
        supply_monthly = supply * days_in_month
        demand_monthly = demand * days_in_month
    else:
        supply_monthly = supply
        demand_monthly = demand

    # Rescale supply to keep stocks in a healthy band
    n_hist = len(idx)
    s_mean = float(np.mean(supply_monthly[:n_hist]))
    d_mean = float(np.mean(demand_monthly[:n_hist]))
    if s_mean > 0:
        scale = (d_mean * 1.002) / s_mean
        supply = supply * scale
        supply_monthly = supply_monthly * scale

    starting_stocks = (tpl.base_demand * tpl.days_cover_target
                       if tpl.unit.endswith("/d")
                       else tpl.base_demand * (tpl.days_cover_target / 30.0))
    starting_stocks = min(max(starting_stocks, 0.2 * tpl.storage_capacity),
                          0.7 * tpl.storage_capacity)
    cap_soft = tpl.storage_capacity * 1.2
    stocks = np.empty(n)
    stocks[0] = starting_stocks
    for i in range(1, n):
        candidate = stocks[i - 1] + (supply_monthly[i] - demand_monthly[i])
        stocks[i] = float(np.clip(candidate, 0.0, cap_soft))

    avg_daily_demand = demand_monthly / days_in_month
    days_cover = stocks / np.where(avg_daily_demand > 0, avg_daily_demand, 1)
    norm_dc = (days_cover - days_cover.mean()) / max(days_cover.std(), 1e-6)
    lo, hi = tpl.price_band
    mid = (lo + hi) / 2
    price = mid * np.exp(-0.18 * norm_dc) * (1 + rng.normal(0, 0.04, n))
    price = np.clip(price, lo * 0.7, hi * 1.3)

    df = pd.DataFrame({
        "date": full_idx, "supply": supply, "demand": demand,
        "imports": imports, "exports": exports, "stocks": stocks,
        "days_cover": days_cover, "price": price,
        "is_forecast": [d in horizon for d in full_idx],
    }).set_index("date")
    return df


@lru_cache(maxsize=64)
def get_regional_dataset(commodity_key: str, seed: int = 7) -> pd.DataFrame:
    """Regional supply/demand split using template weights."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    rng = np.random.default_rng(seed)
    regions = tpl.regions
    n = len(regions)
    d_w = np.array(tpl.region_weights, dtype=float) if tpl.region_weights else np.ones(n)
    d_w = d_w / d_w.sum()
    if tpl.supply_weights and len(tpl.supply_weights) == n:
        s_w = np.array(tpl.supply_weights, dtype=float)
        s_w = s_w / s_w.sum()
        supply = tpl.base_supply * s_w * (1 + rng.normal(0, 0.01, n))
        demand = tpl.base_demand * d_w * (1 + rng.normal(0, 0.01, n))
    else:
        demand = tpl.base_demand * d_w * (1 + rng.normal(0, 0.03, n))
        skew = rng.normal(1.0, 0.20, n)
        supply = tpl.base_supply * d_w * skew
        supply = supply * (tpl.base_supply / supply.sum())
    return pd.DataFrame({
        "region": regions, "supply": supply, "demand": demand,
        "net_trade": supply - demand,
        "supply_share_pct": supply / supply.sum() * 100,
        "demand_share_pct": demand / demand.sum() * 100,
    })


def get_live_spot(commodity_key: str) -> Optional[Dict]:
    """Latest live Yahoo close + 1-day change, or None if unreachable."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    if os.environ.get("DISABLE_YF") == "1":
        return None
    try:
        import yfinance as yf
        hist = yf.Ticker(tpl.ticker).history(period="5d", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        close = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else close
        change_pct = (close - prev) / prev * 100 if prev else 0.0
        return {
            "price": close, "change_pct": change_pct,
            "asof": str(hist.index[-1].date()),
            "source": "yahoo",
        }
    except Exception as exc:
        logger.warning("Yahoo fetch failed for %s: %s", commodity_key, exc)
        return None


def _build_contract_tickers(tpl, n_max: int) -> List[Dict]:
    if not tpl.yf_fmt:
        return []
    now = datetime.now()
    out, offset = [], 0
    while len(out) < n_max and offset < n_max * 4:
        m = (now.month - 1 + offset) % 12
        year = now.year + (now.month - 1 + offset) // 12
        offset += 1
        if _MONTH_CODES[m] not in tpl.active_months:
            continue
        exp_m = m - 1 if m > 0 else 11
        exp_y = year if m > 0 else year - 1
        if now > datetime(exp_y, exp_m + 1, 20):
            continue
        yr2 = f"{year}"[-2:]
        ticker = tpl.yf_fmt.replace("{M}", _MONTH_CODES[m]).replace("{YY}", yr2)
        out.append({
            "ticker": ticker,
            "label": f"{_MONTH_NAMES[m]}-{year}",
            "tenor_month": len(out) + 1,
        })
    return out


def get_futures_curve(commodity_key: str, n_max: int = 12) -> pd.DataFrame:
    """
    Try live curve from Yahoo, fall back to a synthetic shape on failure.
    Returns a DataFrame with tenor_month, label, price, source.
    """
    tpl = COMMODITY_TEMPLATES[commodity_key]
    contracts = _build_contract_tickers(tpl, n_max)
    if contracts and os.environ.get("DISABLE_YF") != "1":
        try:
            import yfinance as yf
            tickers = [c["ticker"] for c in contracts]
            raw = yf.download(tickers, period="5d", auto_adjust=True,
                              progress=False, group_by="ticker")
            if raw is not None and not raw.empty:
                results = []
                for c in contracts:
                    try:
                        if isinstance(raw.columns, pd.MultiIndex):
                            close = raw[c["ticker"]]["Close"].dropna()
                        else:
                            close = raw["Close"].dropna()
                        if not close.empty:
                            results.append({**c, "price": float(close.iloc[-1]),
                                            "source": "yahoo"})
                    except Exception:
                        continue
                if len(results) >= 2:
                    df = pd.DataFrame(results).sort_values("tenor_month").reset_index(drop=True)
                    df["tenor_month"] = range(1, len(df) + 1)
                    return df
        except Exception as exc:
            logger.warning("Live curve failed for %s: %s", commodity_key, exc)

    # Synthetic fallback
    spot = tpl.base_price
    rng = np.random.default_rng(5)
    if not contracts:
        contracts = [{"tenor_month": i + 1, "label": f"M{i+1}"} for i in range(n_max)]
    slope = 0.005   # default contango
    log_p = np.log(spot) + slope * np.arange(1, len(contracts) + 1) + \
            rng.normal(0, 0.005, len(contracts))
    rows = []
    for c, p in zip(contracts, np.exp(log_p)):
        rows.append({**c, "price": float(p), "source": "synthetic"})
    return pd.DataFrame(rows)
