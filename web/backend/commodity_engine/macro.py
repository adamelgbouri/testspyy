"""Per-country macro panels for the Macro overlay."""
from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd


# Calibrated to typical 2020-2024 IMF / OECD / FRED ranges.
COUNTRY_PROFILES: Dict[str, Dict[str, float]] = {
    "🇺🇸 United States":   dict(g_mu=0.0018, g_sig=0.0030, pmi=52, fx_mu=0.0,
                                fx_sig=0.0040, rate=4.5, rate_sig=0.30, cpi=3.0),
    "🇪🇺 Euro Area":       dict(g_mu=0.0010, g_sig=0.0030, pmi=49, fx_mu=0.0002,
                                fx_sig=0.0050, rate=3.5, rate_sig=0.30, cpi=2.5),
    "🇨🇳 China":           dict(g_mu=0.0040, g_sig=0.0035, pmi=50, fx_mu=-0.0003,
                                fx_sig=0.0030, rate=2.8, rate_sig=0.15, cpi=0.5),
    "🇯🇵 Japan":           dict(g_mu=0.0007, g_sig=0.0020, pmi=49, fx_mu=-0.0010,
                                fx_sig=0.0060, rate=0.3, rate_sig=0.10, cpi=2.5),
    "🇬🇧 United Kingdom":  dict(g_mu=0.0011, g_sig=0.0030, pmi=51, fx_mu=-0.0005,
                                fx_sig=0.0055, rate=4.5, rate_sig=0.30, cpi=2.8),
    "🇮🇳 India":           dict(g_mu=0.0055, g_sig=0.0040, pmi=55, fx_mu=-0.0005,
                                fx_sig=0.0040, rate=6.5, rate_sig=0.30, cpi=5.5),
    "🇧🇷 Brazil":          dict(g_mu=0.0020, g_sig=0.0035, pmi=51, fx_mu=-0.0010,
                                fx_sig=0.0080, rate=10.0, rate_sig=0.50, cpi=4.5),
    "🇨🇦 Canada":          dict(g_mu=0.0015, g_sig=0.0025, pmi=50, fx_mu=-0.0003,
                                fx_sig=0.0045, rate=4.0, rate_sig=0.25, cpi=2.8),
}


def get_country_macro(country: str, months: int = 84) -> pd.DataFrame:
    """Generate a stable monthly macro panel for the given country."""
    cfg = COUNTRY_PROFILES.get(country, COUNTRY_PROFILES["🇺🇸 United States"])
    seed = abs(hash(country)) % 10000
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.today().normalize().replace(day=1),
                        periods=months, freq="MS")
    gdp = 100 * np.cumprod(1 + rng.normal(cfg["g_mu"], cfg["g_sig"], months))
    pmi = np.clip(cfg["pmi"] + 3 * np.sin(np.arange(months) / 8)
                  + rng.normal(0, 1.5, months), 40, 60)
    fx = 100 * np.cumprod(1 + rng.normal(cfg["fx_mu"], cfg["fx_sig"], months))
    rate = np.clip(cfg["rate"] + 1.0 * np.sin(np.arange(months) / 24)
                   + rng.normal(0, cfg["rate_sig"], months), 0.0, 15.0)
    cpi = np.clip(cfg["cpi"] + 1.5 * np.sin(np.arange(months) / 18 + 1)
                  + rng.normal(0, 0.4, months), -1.0, 12.0)
    return pd.DataFrame({
        "date": idx, "gdp_index": gdp, "pmi": pmi,
        "fx_vs_usd": fx, "policy_rate": rate, "cpi_yoy": cpi,
    }).set_index("date")


def list_countries() -> List[str]:
    return list(COUNTRY_PROFILES.keys())
