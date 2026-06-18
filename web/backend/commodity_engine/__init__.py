"""Commodity analytics engine — pure-Python, framework-agnostic."""
from .balance import BalanceAssumptions, run_balance
from .config import COMMODITY_TEMPLATES, CommodityTemplate
from .data import (
    get_futures_curve,
    get_live_spot,
    get_regional_dataset,
    get_sd_dataset,
)
from .events import get_market_events
from .fair_value import estimate_fair_value
from .macro import get_country_macro, list_countries
from .monte_carlo import MCConfig, run_monte_carlo
from .options import Black76
from .risk import parametric_var, portfolio_var, stress_scenarios
from .spreads import CRACK_SPREADS, crack_margin

__all__ = [
    "BalanceAssumptions", "Black76", "COMMODITY_TEMPLATES",
    "CommodityTemplate", "CRACK_SPREADS", "crack_margin",
    "estimate_fair_value", "get_country_macro", "get_futures_curve",
    "get_live_spot", "get_market_events", "get_regional_dataset",
    "get_sd_dataset", "list_countries", "MCConfig", "parametric_var",
    "portfolio_var", "run_balance", "run_monte_carlo", "stress_scenarios",
]
