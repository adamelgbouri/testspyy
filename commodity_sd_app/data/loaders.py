"""
Data loaders.

Tries optional sources (yfinance, tvdatafeed) when available, otherwise falls
back to the synthetic generators - the app keeps working with zero network.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from data.synthetic import (
    generate_high_frequency,
    generate_macro_panel,
    generate_positioning,
    generate_regional_balances,
    generate_sd_history,
    generate_futures_curve,
)
from utils.cache import cache_data
from utils.config import COMMODITY_TEMPLATES
from utils.logging_setup import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# CSV import / sample loader
# ---------------------------------------------------------------------------
def load_csv(file) -> pd.DataFrame:
    """Read an uploaded CSV (UploadedFile or path) into a DataFrame."""
    df = pd.read_csv(file)
    # standard normalisation
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    return df


def load_sample_csv(commodity_key: str) -> Optional[pd.DataFrame]:
    """Load the bundled sample CSV for a commodity, if present."""
    path = DATA_DIR / f"sample_{commodity_key}.csv"
    if not path.exists():
        return None
    return load_csv(path)


# ---------------------------------------------------------------------------
# Cached synthetic series
# ---------------------------------------------------------------------------
@cache_data(ttl_seconds=600)
def get_sd_dataset(commodity_key: str, start: str = "2018-01-01",
                   forecast_months: int = 24, seed: int = 42) -> pd.DataFrame:
    return generate_sd_history(commodity_key, start=start,
                               forecast_months=forecast_months, seed=seed)


@cache_data(ttl_seconds=600)
def get_regional_dataset(commodity_key: str) -> pd.DataFrame:
    return generate_regional_balances(commodity_key)


@cache_data(ttl_seconds=600)
def get_high_frequency(commodity_key: str, days: int = 120) -> pd.DataFrame:
    return generate_high_frequency(commodity_key, days=days)


@cache_data(ttl_seconds=600)
def get_macro_panel(months: int = 84) -> pd.DataFrame:
    return generate_macro_panel(months=months)


@cache_data(ttl_seconds=600)
def get_positioning(commodity_key: str, weeks: int = 120) -> pd.DataFrame:
    return generate_positioning(commodity_key, weeks=weeks)


@cache_data(ttl_seconds=600)
def get_futures_curve(commodity_key: str, structure: str = "contango",
                      months: int = 24) -> pd.DataFrame:
    return generate_futures_curve(commodity_key, structure=structure, months=months)


# ---------------------------------------------------------------------------
# Optional live price - yfinance
# ---------------------------------------------------------------------------
@cache_data(ttl_seconds=300)
def get_yahoo_history(commodity_key: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Fetch a Yahoo Finance series for the commodity, or None if unavailable."""
    tpl = COMMODITY_TEMPLATES[commodity_key]
    if os.environ.get("COMMODITY_SD_DISABLE_YF") == "1":
        return None
    try:
        import yfinance as yf

        hist = yf.Ticker(tpl.ticker).history(period=period, auto_adjust=False)
        if hist is None or hist.empty:
            logger.info("Yahoo returned empty history for %s", tpl.ticker)
            return None
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as exc:  # network / library errors
        logger.warning("Yahoo Finance fetch failed for %s: %s", tpl.ticker, exc)
        return None
