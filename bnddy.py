"""
Portfolio Lab - build a portfolio, test it honestly, and see where it might take you.

    streamlit run portfolio_lab.py
    python portfolio_lab.py --selftest

Educational simulation. Not investment advice.
"""
from __future__ import annotations

import hashlib
import re
import json
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import streamlit as st
    HAS_ST = True
except Exception:
    st, HAS_ST = None, False
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    go, HAS_PLOTLY = None, False
try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    yf, HAS_YF = None, False


def cache_data(**kw) -> Callable:
    def deco(fn):
        return st.cache_data(**kw)(fn) if HAS_ST else fn
    return deco


class PortfolioLabError(Exception): ...
class DataError(PortfolioLabError): ...
class FXError(DataError): ...
class InfeasibleConstraints(PortfolioLabError): ...
class OptimizationError(PortfolioLabError): ...


# ── Constants ───────────────────────────────────────────────────────────────
PERIODS, MONTHS, SEED, N_MC = 252, 12, 42, 12_000
PERIODS_PER = {"D": 252, "W": 52, "M": 12}
FREQ_RULE = {"D": None, "W": "W-FRI", "M": "ME"}

BG, PANEL, GRID = "#05080D", "#0A1220", "rgba(255,255,255,0.05)"
TEXT, MUTED, GOLD = "#EEF2F7", "#64748B", "#D4AF37"
GREEN, RED, BLUE = "#10B981", "#EF4444", "#3B82F6"
ORANGE, PURPLE, CYAN, PINK = "#F59E0B", "#8B5CF6", "#06B6D4", "#EC4899"
PALETTE = [GOLD, BLUE, GREEN, RED, PURPLE, ORANGE, CYAN, PINK, "#84CC16", "#F97316"]
PORT_COLORS = {"Max Sharpe": GOLD, "Min Variance": BLUE, "Risk Parity": PURPLE,
               "Min CVaR": CYAN, "Resampled": PINK,
               "Equal weight (1/N)": MUTED, "Benchmark 60/40": ORANGE}

CURRENCIES = {
    "EUR": {"symbol": "€", "flag": "🇪🇺", "rf": 0.0200, "proxy": None},
    "USD": {"symbol": "$", "flag": "🇺🇸", "rf": 0.0425, "proxy": "^IRX"},
    "GBP": {"symbol": "£", "flag": "🇬🇧", "rf": 0.0400, "proxy": None},
    "CHF": {"symbol": "CHF", "flag": "🇨🇭", "rf": 0.0050, "proxy": None},
    "JPY": {"symbol": "¥", "flag": "🇯🇵", "rf": 0.0050, "proxy": None},
    "CAD": {"symbol": "C$", "flag": "🇨🇦", "rf": 0.0300, "proxy": None},
    "AUD": {"symbol": "A$", "flag": "🇦🇺", "rf": 0.0385, "proxy": None},
}
FX_FALLBACK = {"EUR": 1.08, "GBP": 1.27, "CHF": 1.12, "JPY": 0.0067,
               "CAD": 0.74, "AUD": 0.65, "USD": 1.00}

# What you buy.
INSTRUMENT = {"EQUITY": "Stock", "ETF": "ETF", "MUTUALFUND": "Fund",
              "CRYPTOCURRENCY": "Crypto", "INDEX": "Index", "CURRENCY": "Forex",
              "FUTURE": "Future", "OPTION": "Option"}

# What you are exposed to.
CLASS_HINTS = {
    "Bonds": ("AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "GOVT",
                    "AGGH", "IEAG", "EUNA", "VGEA", "IBTA", "SEGA", "BNDX", "EMB"),
    "Commodities": ("GLD", "IAU", "SLV", "DBC", "PDBC", "USO", "GC=F",
                           "SI=F", "CL=F", "SGLN", "PHAU"),
    "Real estate": ("VNQ", "IYR", "SCHH", "REET", "RWO", "IPRP", "EPRA", "IWDP"),
    "Cash": ("BIL", "SHV", "SGOV", "ERNA", "XEON", "CSH2", "PARO"),
}
CLASSES = ["Stocks", "Bonds", "Commodities", "Real estate", "Cash", "Crypto", "Other"]

DEFAULT_BROKER_BPS, DEFAULT_SPREAD_BPS, DEFAULT_TER_BPS = 10.0, 5.0, 20.0

TAX_REGIMES = {
    "None (gross)": (0.000, "Returns before any tax."),
    "Flat 15%": (0.150, "Long-term capital gains, many jurisdictions."),
    "Flat 20%": (0.200, "Higher bracket capital gains."),
    "France - CTO 30%": (0.300, "12.8% income tax + 17.2% social charges."),
    "France - PEA 17.2%": (0.172, "Social charges only, after 5 years, EU assets."),
    "UK - 20% CGT": (0.200, "Above annual allowance."),
}

RISK_PROFILES = {
    "Cautious": (0.25, 0.40, "Max 25% per holding, risky assets capped at 40%."),
    "Balanced": (0.35, 0.70, "Max 35% per holding, risky assets capped at 70%."),
    "Growth": (0.45, 0.90, "Max 45% per holding, risky assets capped at 90%."),
    "Aggressive": (1.00, 1.00, "No automatic guardrails."),
}

SCENARIOS = {
    "2008 - Financial crisis": ("2008-09-01", "2009-03-31", RED,
                                "Lehman collapses. S&P 500 down 57% peak to trough."),
    "2020 - Covid crash": ("2020-02-19", "2020-03-23", ORANGE,
                           "Down 34% in 33 days, the fastest bear market ever."),
    "2022 - Rate shock": ("2022-01-01", "2022-12-31", PURPLE,
                          "Fed hiked 425bp. Stocks and bonds fell together."),
    "2000-2002 - Dot-com bust": ("2000-03-10", "2002-10-09", PINK,
                                 "NASDAQ lost 78% over two and a half years."),
    "2011 - Euro debt crisis": ("2011-07-01", "2011-10-03", CYAN,
                                "Greece, Italy and Spain under pressure."),
    "2015 - Yuan devaluation": ("2015-08-10", "2016-02-11", "#84CC16",
                                "China slowdown, oil collapse."),
}

FACTOR_PROXIES = {"Market": "SPY", "Size": "IWM", "Value": "VLUE",
                  "Momentum": "MTUM", "Quality": "QUAL", "Low volatility": "USMV"}

REBALANCE_CHOICES = {"Never (buy & hold)": "none", "Monthly": "M",
                     "Quarterly": "Q", "Yearly": "A",
                     "Only when 5pts off target": "T5", "Daily (theoretical)": "D"}


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class CostModel:
    broker_bps: float = DEFAULT_BROKER_BPS
    spread_bps: float = DEFAULT_SPREAD_BPS
    ter_bps: float = DEFAULT_TER_BPS

    @property
    def one_way(self) -> float:
        return (self.broker_bps + self.spread_bps) / 1e4

    def ter_vector(self, n: int) -> np.ndarray:
        if np.isscalar(self.ter_bps):
            return np.full(n, float(self.ter_bps) / 1e4)
        v = np.asarray(self.ter_bps, float) / 1e4
        if v.size != n:
            raise ValueError("TER: length mismatch")
        return v


@dataclass
class Constraints:
    """Per-asset bounds plus group constraints (instrument type, asset class)."""
    n: int
    min_w: np.ndarray = None
    max_w: np.ndarray = None
    groups: dict[str, tuple[list[int], float, float]] = field(default_factory=dict)
    max_assets: Optional[int] = None

    def __post_init__(self):
        self.min_w = np.zeros(self.n) if self.min_w is None else np.asarray(self.min_w, float)
        self.max_w = np.ones(self.n) if self.max_w is None else np.asarray(self.max_w, float)
        if self.min_w.size != self.n or self.max_w.size != self.n:
            raise ValueError("bounds: length mismatch")

    def bounds(self, floor: float = 0.0) -> list[tuple[float, float]]:
        return [(max(float(self.min_w[i]), floor), float(self.max_w[i]))
                for i in range(self.n)]

    def group_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """A_ub @ w <= b_ub, encoding gmin <= sum(w) <= gmax for each group."""
        rows, rhs = [], []
        for idx, gmin, gmax in self.groups.values():
            sel = np.zeros(self.n)
            sel[idx] = 1.0
            if gmax < 1.0 - 1e-12:
                rows.append(sel.copy()); rhs.append(float(gmax))
            if gmin > 1e-12:
                rows.append(-sel.copy()); rhs.append(-float(gmin))
        if not rows:
            return np.zeros((0, self.n)), np.zeros(0)
        return np.vstack(rows), np.array(rhs)

    def scipy_constraints(self) -> list[dict]:
        cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
        A, b = self.group_matrix()
        for i in range(A.shape[0]):
            cons.append({"type": "ineq",
                         "fun": lambda w, a=A[i].copy(), bb=float(b[i]): bb - float(a @ w)})
        return cons

    def subset(self, idx: Sequence[int]) -> "Constraints":
        idx = list(idx)
        pos = {o: n for n, o in enumerate(idx)}
        groups = {}
        for name, (gidx, _, gmax) in self.groups.items():
            kept = [pos[i] for i in gidx if i in pos]
            if kept:
                groups[name] = (kept, 0.0, gmax)
        return Constraints(len(idx), self.min_w[idx], self.max_w[idx], groups,
                           self.max_assets)


@dataclass
class ModelSpec:
    mu_method: str = "historical"      # historical | black_litterman | none
    cov_method: str = "ledoit_wolf"    # sample | ledoit_wolf
    est_freq: str = "D"                # D | W | M
    bl_delta: float = 2.5
    bl_tau: float = 0.05
    bl_views: Optional[np.ndarray] = None
    bl_confidence: Optional[np.ndarray] = None
    bl_prior_w: Optional[np.ndarray] = None


# ══ Moments ═════════════════════════════════════════════════════════════════

def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def resample_returns(returns: pd.DataFrame, freq: str) -> pd.DataFrame:
    rule = FREQ_RULE.get(freq)
    if rule is None:
        return returns
    return ((1.0 + returns).resample(rule).prod() - 1.0).dropna(how="all")


def annualized_mu(returns, periods: int = PERIODS) -> np.ndarray:
    return returns.mean().values * periods


def sample_cov(returns, periods: int = PERIODS) -> np.ndarray:
    return returns.cov().values * periods


def geometric_to_arithmetic(g, sigma):
    return g + 0.5 * np.asarray(sigma) ** 2


def arithmetic_to_geometric(mu, sigma):
    return mu - 0.5 * sigma ** 2


def ledoit_wolf_cov(returns, periods: int = PERIODS) -> tuple[np.ndarray, float]:
    """Shrinkage toward a constant-correlation target (Ledoit & Wolf 2003)."""
    X = returns.values
    X = X[~np.isnan(X).any(axis=1)]
    t, n = X.shape
    if t < 10 or n < 2:
        return sample_cov(returns, periods), 0.0
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / t
    var = np.diag(S)
    sd = np.sqrt(np.maximum(var, 1e-18))
    off = ~np.eye(n, dtype=bool)
    r_bar = (S / np.outer(sd, sd))[off].mean()
    F = r_bar * np.outer(sd, sd)
    np.fill_diagonal(F, var)

    Xc2 = Xc ** 2
    pi_mat = (Xc2.T @ Xc2) / t - S ** 2
    term = ((Xc ** 3).T @ Xc) / t - var[:, None] * S
    inv = 1.0 / np.maximum(var, 1e-18)
    rho = np.diag(pi_mat).sum() + r_bar * (
        (np.sqrt(np.outer(var, inv)) * term
         + np.sqrt(np.outer(inv, var)) * term.T) / 2.0)[off].sum()
    gamma = float(((F - S) ** 2).sum())
    delta = 0.0 if gamma <= 1e-18 else float(np.clip((pi_mat.sum() - rho) / gamma / t, 0, 1))
    shrunk = delta * F + (1 - delta) * S
    return 0.5 * (shrunk + shrunk.T) * periods, delta


def nearest_psd(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    return vecs @ np.diag(np.maximum(vals, eps)) @ vecs.T


def black_litterman_mu(cov: np.ndarray, spec: ModelSpec, rf: float) -> np.ndarray:
    """Equilibrium returns pi = delta*Cov*w_mkt, updated by the user's views."""
    n = cov.shape[0]
    w = np.ones(n) / n if spec.bl_prior_w is None else np.asarray(spec.bl_prior_w, float)
    w = w / w.sum()
    pi = spec.bl_delta * (cov @ w) + rf
    if spec.bl_views is None:
        return pi
    views = np.asarray(spec.bl_views, float)
    conf = np.asarray(spec.bl_confidence if spec.bl_confidence is not None
                      else np.full(n, 0.5), float)
    active = ~np.isnan(views) & (conf > 1e-6)
    if not active.any():
        return pi
    P, Q = np.eye(n)[active], views[active]
    tau_cov = spec.bl_tau * cov
    omega = np.diag(np.diag(P @ tau_cov @ P.T) / np.clip(conf[active], 1e-3, 1.0))
    it, io = np.linalg.pinv(tau_cov), np.linalg.pinv(omega)
    return np.linalg.pinv(it + P.T @ io @ P) @ (it @ pi + P.T @ io @ Q)


def estimate_moments(returns: pd.DataFrame, spec: ModelSpec,
                     rf: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict]:
    freq = spec.est_freq if spec.est_freq in PERIODS_PER else "D"
    r = resample_returns(returns, freq)
    if len(r) < 30 and freq != "D":
        freq, r = "D", returns
    per = PERIODS_PER[freq]
    t = len(r)

    if spec.cov_method == "ledoit_wolf":
        cov, delta = ledoit_wolf_cov(r, per)
    else:
        cov, delta = sample_cov(r, per), 0.0
    cov = nearest_psd(cov)
    vol = np.sqrt(np.diag(cov))
    mu_hist = annualized_mu(r, per)
    se_mu = vol / np.sqrt(max(t / per, 1e-9))

    if spec.mu_method == "none":
        mu = np.zeros(returns.shape[1])
    elif spec.mu_method == "black_litterman":
        mu = black_litterman_mu(cov, spec, rf)
    else:
        mu = mu_hist
    return mu, cov, {"est_freq": freq, "periods_per_year": per, "t_obs": t,
                     "shrinkage_delta": delta, "se_mu": se_mu, "mu_hist": mu_hist,
                     "noise_ratio": np.abs(se_mu) / np.maximum(np.abs(mu_hist), 1e-9)}


# ══ Mesures de risque ═══════════════════════════════════════════════════════

def port_return(w, mu) -> float:
    return float(np.asarray(w) @ np.asarray(mu))


def port_vol(w, cov) -> float:
    w = np.asarray(w)
    return float(np.sqrt(max(float(w @ cov @ w), 0.0)))


def sharpe(w, mu, cov, rf) -> float:
    v = port_vol(w, cov)
    return (port_return(w, mu) - rf) / v if v > 1e-9 else 0.0


def pmetrics(w, mu, cov, rf):
    r, v = port_return(w, mu), port_vol(w, cov)
    return r, v, ((r - rf) / v if v > 1e-9 else 0.0)


def drawdown_series(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def max_drawdown(nav: pd.Series) -> float:
    return float(drawdown_series(nav).min())


def underwater_days(nav: pd.Series) -> int:
    dd = drawdown_series(nav).values
    longest = run = 0
    for x in dd:
        run = run + 1 if x < -1e-12 else 0
        longest = max(longest, run)
    return int(longest)


def ulcer_index(nav: pd.Series) -> float:
    return float(np.sqrt(np.mean(drawdown_series(nav).values ** 2)))


def cagr(nav: pd.Series, periods: int = PERIODS) -> float:
    y = len(nav) / periods
    if y <= 0.1 or nav.iloc[0] <= 0:
        return 0.0
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / y) - 1.0)


def ann_vol_from_series(r: pd.Series, periods: int = PERIODS) -> float:
    return float(r.std(ddof=1) * np.sqrt(periods))


def downside_deviation(r: pd.Series, mar: float = 0.0, periods: int = PERIODS) -> float:
    d = np.minimum(r.values - mar / periods, 0.0)
    return float(np.sqrt(np.mean(d ** 2)) * np.sqrt(periods))


def sortino(r: pd.Series, rf: float = 0.0, periods: int = PERIODS) -> float:
    dd = downside_deviation(r, rf, periods)
    return float((r.mean() * periods - rf) / dd) if dd > 1e-12 else 0.0


def calmar(nav: pd.Series, periods: int = PERIODS) -> float:
    mdd = abs(max_drawdown(nav))
    return float(cagr(nav, periods) / mdd) if mdd > 1e-9 else 0.0


def historical_var(r: pd.Series, alpha: float = 0.95) -> float:
    return float("nan") if len(r) < 20 else float(-np.quantile(r.values, 1 - alpha))


def historical_cvar(r: pd.Series, alpha: float = 0.95) -> float:
    if len(r) < 20:
        return float("nan")
    tail = r.values[r.values <= np.quantile(r.values, 1 - alpha)]
    return float(-tail.mean()) if tail.size else float("nan")


def benchmark_stats(r: pd.Series, rb: pd.Series, rf=0.0, periods=PERIODS) -> dict:
    df = pd.concat([r, rb], axis=1).dropna()
    keys = ("beta", "alpha", "tracking_error", "information_ratio", "r2", "corr")
    if len(df) < 30:
        return {k: float("nan") for k in keys}
    x = df.iloc[:, 1].values - rf / periods
    y = df.iloc[:, 0].values - rf / periods
    vx = x.var(ddof=1)
    beta = float(np.cov(y, x, ddof=1)[0, 1] / vx) if vx > 1e-18 else float("nan")
    active = df.iloc[:, 0].values - df.iloc[:, 1].values
    te = float(active.std(ddof=1) * np.sqrt(periods))
    corr = float(np.corrcoef(y, x)[0, 1])
    return {"beta": beta, "alpha": float((y.mean() - beta * x.mean()) * periods),
            "tracking_error": te,
            "information_ratio": float(active.mean() * periods / te) if te > 1e-12 else float("nan"),
            "r2": corr ** 2, "corr": corr}


def full_stats(nav: pd.Series, rets: pd.Series, rf: float,
               bench: Optional[pd.Series] = None) -> dict:
    vol = ann_vol_from_series(rets)
    out = {"cagr": cagr(nav), "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
           "vol": vol,
           "sharpe": (rets.mean() * PERIODS - rf) / vol if vol > 1e-12 else 0.0,
           "sortino": sortino(rets, rf), "calmar": calmar(nav),
           "max_drawdown": max_drawdown(nav), "underwater_days": underwater_days(nav),
           "ulcer": ulcer_index(nav),
           "var95": historical_var(rets), "cvar95": historical_cvar(rets),
           "var99": historical_var(rets, 0.99), "cvar99": historical_cvar(rets, 0.99),
           "skew": float(rets.skew()), "kurtosis": float(rets.kurtosis())}
    if bench is not None:
        out.update(benchmark_stats(rets, bench, rf))
    return out


def risk_contributions(w, cov) -> np.ndarray:
    w = np.asarray(w, float)
    pv = np.sqrt(max(float(w @ cov @ w), 0.0))
    if pv < 1e-12:
        return np.zeros(len(w))
    rc = w * ((cov @ w) / pv)
    tot = rc.sum()
    return rc / tot * 100.0 if abs(tot) > 1e-12 else rc * 0.0


def diversification_ratio(w, cov) -> float:
    pv = port_vol(w, cov)
    return float((np.asarray(w) @ np.sqrt(np.diag(cov))) / pv) if pv > 1e-12 else 1.0


def effective_n(w) -> float:
    h = float((np.asarray(w, float) ** 2).sum())
    return 1.0 / h if h > 1e-12 else 0.0


# ══ Optimisation ════════════════════════════════════════════════════════════

def check_feasibility(cons: Constraints) -> None:
    """LP feasibility check: fail loudly, never fall back to a silent 1/n."""
    if np.any(cons.min_w > cons.max_w + 1e-12):
        bad = np.where(cons.min_w > cons.max_w)[0].tolist()
        raise InfeasibleConstraints(f"Minimum weight exceeds maximum for holding(s) {bad}.")
    if cons.min_w.sum() > 1 + 1e-9:
        raise InfeasibleConstraints(
            f"Minimum weights add up to {cons.min_w.sum():.1%}, which is over 100%.")
    if cons.max_w.sum() < 1 - 1e-9:
        raise InfeasibleConstraints(
            f"Maximum weights only add up to {cons.max_w.sum():.1%}, under 100%.")
    for name, (idx, gmin, gmax) in cons.groups.items():
        if gmin > gmax + 1e-12:
            raise InfeasibleConstraints(f"Group '{name}': minimum is above maximum.")
        if cons.max_w[idx].sum() < gmin - 1e-9:
            raise InfeasibleConstraints(
                f"Group '{name}': the {gmin:.0%} minimum cannot be reached - "
                f"individual caps inside this group only add up to "
                f"{cons.max_w[idx].sum():.0%}.")
        if cons.min_w[idx].sum() > gmax + 1e-9:
            raise InfeasibleConstraints(
                f"Group '{name}': the {gmax:.0%} cap is already exceeded by the "
                f"individual minimums inside it ({cons.min_w[idx].sum():.0%}).")
        outside = [i for i in range(cons.n) if i not in set(idx)]
        # Whatever the group is not allowed to hold has to go somewhere else.
        if cons.max_w[outside].sum() < 1 - gmax - 1e-9:
            room = f"{cons.max_w[outside].sum():.0%}" if outside else "nothing"
            raise InfeasibleConstraints(
                f"Group '{name}' is capped at {gmax:.0%}, so {1-gmax:.0%} has to sit "
                f"outside it - but the rest of your portfolio can only hold {room}. "
                + ("Every holding belongs to this group, so the cap cannot bind. "
                   if not outside else "")
                + "Either raise this cap or add a holding outside the group.")
        if cons.min_w[outside].sum() > 1 - gmin + 1e-9:
            raise InfeasibleConstraints(
                f"Group '{name}' needs at least {gmin:.0%}, but the minimums you set "
                f"outside it already claim {cons.min_w[outside].sum():.0%}.")
    A, b = cons.group_matrix()
    r = linprog(np.zeros(cons.n), A_ub=A if A.shape[0] else None,
                b_ub=b if A.shape[0] else None,
                A_eq=np.ones((1, cons.n)), b_eq=[1.0], bounds=cons.bounds(),
                method="highs")
    if not r.success:
        raise InfeasibleConstraints(
            "Your limits contradict each other - no portfolio can satisfy them "
            "all. Groups in play: " + (", ".join(cons.groups) or "none") + ". The "
            "usual cause is group minimums adding up to more than 100%, or two "
            "overlapping groups (a type limit and a class limit) squeezing the "
            "same holdings from both sides.")


def feasible_return_range(mu, cons: Constraints) -> tuple[float, float]:
    A, b = cons.group_matrix()
    kw = dict(A_ub=A if A.shape[0] else None, b_ub=b if A.shape[0] else None,
              A_eq=np.ones((1, cons.n)), b_eq=[1.0], bounds=cons.bounds(),
              method="highs")
    lo, hi = linprog(mu, **kw), linprog(-mu, **kw)
    if not (lo.success and hi.success):
        raise InfeasibleConstraints("Return range could not be computed.")
    return float(mu @ lo.x), float(mu @ hi.x)


def _clean(w, cons: Constraints, tol: float = 1e-6) -> np.ndarray:
    w = np.asarray(w, float).copy()
    w[np.abs(w) < tol] = 0.0
    w = np.clip(w, cons.min_w, cons.max_w)
    s = w.sum()
    if s <= 1e-9:
        raise OptimizationError("Degenerate weights (they sum to zero).")
    return w / s


def _starts(cons: Constraints, k: int, rng) -> list[np.ndarray]:
    base = np.clip(np.ones(cons.n) / cons.n, cons.min_w, cons.max_w)
    pts = [base / base.sum()]
    for _ in range(max(k - 1, 0)):
        w = np.clip(rng.dirichlet(np.ones(cons.n)), cons.min_w, cons.max_w)
        pts.append(w / w.sum() if w.sum() > 1e-9 else pts[0])
    return pts


def _solve(fun, w0, cons: Constraints, floor: float = 0.0):
    return minimize(fun, w0, method="SLSQP", bounds=cons.bounds(floor),
                    constraints=cons.scipy_constraints(),
                    options={"ftol": 1e-12, "maxiter": 800})


def solve_min_variance(cov, cons: Constraints) -> np.ndarray:
    check_feasibility(cons)
    rng = np.random.default_rng(SEED)
    best, bw = np.inf, None
    for w0 in _starts(cons, 5, rng):
        r = _solve(lambda w: float(w @ cov @ w), w0, cons)
        if r.success and r.fun < best:
            best, bw = r.fun, r.x
    if bw is None:
        raise OptimizationError("Min variance did not converge - the covariance "
                                "matrix may be degenerate, or your limits too tight.")
    return _clean(bw, cons)


def solve_max_sharpe(mu, cov, cons: Constraints, rf: float, n_starts=40) -> np.ndarray:
    check_feasibility(cons)
    if np.allclose(mu, 0.0):
        raise OptimizationError(
            "Sharpe cannot be maximised without expected returns. Pick a return "
            "model, or use Min Variance / Risk Parity instead.")
    rng = np.random.default_rng(SEED)

    def neg(w):
        return -(float(w @ mu) - rf) / np.sqrt(max(float(w @ cov @ w), 1e-18))

    best, bw = -np.inf, None
    for w0 in _starts(cons, n_starts, rng):
        r = _solve(neg, w0, cons)
        if r.success and -r.fun > best:
            best, bw = -r.fun, r.x
    if bw is None:
        raise OptimizationError("Max Sharpe did not converge.")
    return _clean(bw, cons)


def solve_risk_parity(cov, cons: Constraints, n_starts=20) -> np.ndarray:
    check_feasibility(cons)
    rng = np.random.default_rng(SEED)

    def obj(w):
        pv = np.sqrt(max(float(w @ cov @ w), 1e-18))
        rc = w * ((cov @ w) / pv)
        return float(((rc - rc.mean()) ** 2).sum()) * 1e4

    best, bw = np.inf, None
    for w0 in _starts(cons, n_starts, rng):
        r = _solve(obj, w0, cons, floor=1e-4)
        if r.success and r.fun < best:
            best, bw = r.fun, r.x
    if bw is None:
        raise OptimizationError("Risk parity did not converge.")
    return _clean(bw, cons)


def solve_target_return(mu, cov, cons: Constraints, target: float):
    extra = cons.scipy_constraints() + [
        {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t}]
    rng = np.random.default_rng(SEED)
    for w0 in _starts(cons, 3, rng):
        r = minimize(lambda w: float(w @ cov @ w), w0, method="SLSQP",
                     bounds=cons.bounds(), constraints=extra,
                     options={"ftol": 1e-12, "maxiter": 500})
        if r.success:
            return _clean(r.x, cons)
    return None


def solve_min_cvar(returns: pd.DataFrame, cons: Constraints, alpha: float = 0.95,
                   target_return: Optional[float] = None, mu=None) -> np.ndarray:
    """Rockafellar-Uryasev linear program for minimum conditional value at risk."""
    from scipy import sparse
    check_feasibility(cons)
    R = returns.values
    T, n = R.shape
    if T < 60:
        raise DataError("Not enough history for CVaR optimisation (under 60 days).")
    c = np.concatenate([np.zeros(n), [1.0], np.full(T, 1.0 / (T * (1 - alpha)))])
    A_ub = [sparse.hstack([sparse.csr_matrix(-R), sparse.csr_matrix(-np.ones((T, 1))),
                           -sparse.identity(T, format="csr")], format="csr")]
    b_ub = [np.zeros(T)]
    Ag, bg = cons.group_matrix()
    if Ag.shape[0]:
        A_ub.append(sparse.hstack([sparse.csr_matrix(Ag),
                                   sparse.csr_matrix((Ag.shape[0], 1 + T))], format="csr"))
        b_ub.append(bg)
    if target_return is not None and mu is not None:
        A_ub.append(sparse.hstack([sparse.csr_matrix(-np.asarray(mu).reshape(1, -1)),
                                   sparse.csr_matrix((1, 1 + T))], format="csr"))
        b_ub.append(np.array([-target_return]))
    res = linprog(c, A_ub=sparse.vstack(A_ub, format="csr"), b_ub=np.concatenate(b_ub),
                  A_eq=sparse.hstack([sparse.csr_matrix(np.ones((1, n))),
                                      sparse.csr_matrix((1, 1 + T))], format="csr"),
                  b_eq=[1.0], bounds=cons.bounds() + [(None, None)] + [(0, None)] * T,
                  method="highs")
    if not res.success:
        raise OptimizationError(f"CVaR optimisation failed: {res.message}")
    return _clean(res.x[:n], cons)


def apply_cardinality(w, k: int, resolve: Callable[[list[int]], np.ndarray],
                      cons: Constraints) -> np.ndarray:
    """Keep the k largest weights, then re-optimise on that subset."""
    if k is None or k >= cons.n or (w > 1e-6).sum() <= k:
        return w
    keep = sorted(np.argsort(-w)[:k].tolist())
    if np.any(cons.min_w[[i for i in range(cons.n) if i not in keep]] > 1e-9):
        raise InfeasibleConstraints(
            "Holding limit conflicts with a minimum weight on an excluded asset.")
    for _, (idx, gmin, _) in cons.groups.items():
        if gmin > 1e-9 and not set(idx) & set(keep):
            raise InfeasibleConstraints(
                "Holding limit conflicts with a group minimum: every asset in "
                "that group would be dropped.")
    full = np.zeros(cons.n)
    full[keep] = resolve(keep)
    return full


def efficient_frontier(mu, cov, cons: Constraints, n_points: int = 60):
    lo, hi = feasible_return_range(mu, cons)
    w_mvp = solve_min_variance(cov, cons)
    lo = max(lo, float(w_mvp @ mu))
    if hi <= lo + 1e-9:
        return np.array([port_vol(w_mvp, cov)]), np.array([float(w_mvp @ mu)]), 0
    vols, rets, failed = [], [], 0
    for t in np.linspace(lo, hi, n_points):
        w = solve_target_return(mu, cov, cons, float(t))
        if w is None:
            failed += 1
            continue
        vols.append(port_vol(w, cov)); rets.append(float(w @ mu))
    return np.array(vols), np.array(rets), failed


def resampled_weights(returns, cons: Constraints, rf: float, spec: ModelSpec,
                      objective="max_sharpe", n_boot=60, seed=SEED):
    """Michaud resampling: average the optimal weights over bootstrap draws."""
    rng = np.random.default_rng(seed)
    T = len(returns)
    W = []
    for _ in range(n_boot):
        sample = returns.iloc[rng.integers(0, T, T)]
        try:
            mu_b, cov_b, _ = estimate_moments(sample, spec, rf)
            if objective == "max_sharpe":
                W.append(solve_max_sharpe(mu_b, cov_b, cons, rf, n_starts=8))
            elif objective == "min_variance":
                W.append(solve_min_variance(cov_b, cons))
            else:
                W.append(solve_risk_parity(cov_b, cons, n_starts=5))
        except PortfolioLabError:
            continue
    if not W:
        raise OptimizationError("Resampling failed: no bootstrap draw converged.")
    W = np.array(W)
    return _clean(W.mean(axis=0), cons), W


def random_cloud(mu, cov, cons: Constraints, rf=0.0, n=N_MC, seed=SEED):
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(cons.n), size=n)
    ok = np.all((W >= cons.min_w - 1e-9) & (W <= cons.max_w + 1e-9), axis=1)
    if ok.sum() < 50:
        W = np.clip(W, cons.min_w, cons.max_w)
        W = W / W.sum(axis=1, keepdims=True)
    else:
        W = W[ok]
    rets = W @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    with np.errstate(divide="ignore", invalid="ignore"):
        return rets, vols, np.where(vols > 1e-9, (rets - rf) / vols, 0.0)


# ══ Trajectoire, frais, backtest ════════════════════════════════════════════

def rebalance_flags(index: pd.DatetimeIndex, policy: str) -> np.ndarray:
    n = len(index)
    if policy in ("none", "T5"):
        return np.zeros(n, dtype=bool)
    if policy == "D":
        return np.ones(n, dtype=bool)
    per = pd.PeriodIndex(index, freq={"M": "M", "Q": "Q", "A": "Y"}[policy])
    flags = np.zeros(n, dtype=bool)
    flags[:-1] = per[:-1].asi8 != per[1:].asi8
    return flags


def portfolio_path(returns: pd.DataFrame, w_target, *, rebalance="M",
                   costs: Optional[CostModel] = None, charge_initial=True,
                   band: float = 0.05) -> dict:
    """Net-of-fee portfolio value, anchored at 1.0 before the first session.

    rebalance: none | M | Q | A | D | T5 (rebalance once drift exceeds `band`).
    """
    costs = costs or CostModel()
    R, idx = returns.values, returns.index
    n = R.shape[1]
    w_target = np.asarray(w_target, float)
    if w_target.size != n:
        raise ValueError("weights: length mismatch")
    c, ter = costs.one_way, costs.ter_vector(n)
    flags = rebalance_flags(idx, rebalance)
    threshold = (rebalance == "T5")

    nav = 1.0
    fees = 0.0
    if charge_initial:
        fees = float(np.abs(w_target).sum() * c)
        nav *= (1 - fees)
    w = w_target.copy()
    navs, turn = np.empty(len(idx)), np.zeros(len(idx))
    w_hist = np.empty((len(idx), n))

    for t in range(len(idx)):
        r = np.nan_to_num(R[t])
        gross = max(1.0 + float(w @ r), 1e-12)
        nav *= gross
        w = w * (1 + r) / gross
        drag = float(w @ ter) / PERIODS
        nav *= (1 - drag)
        fees += drag
        drift = float(np.abs(w_target - w).sum())
        if flags[t] or (threshold and drift > 2 * band):
            nav *= (1 - drift * c)
            fees += drift * c
            turn[t] = drift / 2
            w = w_target.copy()
        navs[t], w_hist[t] = nav, w

    anchor = idx[0] - pd.Timedelta(days=1)
    nav_s = pd.Series(np.concatenate([[1.0], navs]),
                      index=pd.DatetimeIndex([anchor]).append(idx), name="nav")
    years = max(len(idx) / PERIODS, 1e-9)
    return {"nav": nav_s, "returns": nav_s.pct_change().dropna(),
            "turnover": pd.Series(turn, index=idx),
            "annual_turnover": float(turn.sum() / years), "total_fees": fees,
            "weights": pd.DataFrame(w_hist, index=idx, columns=returns.columns),
            "final_weights": w.copy()}


def portfolio_stats(returns, w, rf, *, rebalance="M", costs=None, bench=None) -> dict:
    p = portfolio_path(returns, w, rebalance=rebalance, costs=costs)
    s = full_stats(p["nav"], p["returns"], rf, bench)
    s.update(annual_turnover=p["annual_turnover"], total_fees=p["total_fees"],
             nav=p["nav"], net_returns=p["returns"])
    return s


def _period_ends(index: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    per = pd.PeriodIndex(index, freq={"M": "M", "Q": "Q", "A": "Y"}.get(freq, "Q"))
    return [index[i] for i in range(len(index) - 1) if per[i] != per[i + 1]]


def walk_forward(returns: pd.DataFrame, *, method: str, cons: Constraints, rf: float,
                 spec: ModelSpec, lookback_years=3.0, reb_freq="Q",
                 costs: Optional[CostModel] = None,
                 progress: Optional[Callable] = None) -> dict:
    """Re-optimise on a rolling window, then measure on the period that follows."""
    costs = costs or CostModel()
    lookback = int(lookback_years * PERIODS)
    idx = returns.index
    if len(idx) < lookback + 60:
        raise DataError(
            f"Not enough history: {len(idx)} sessions available, {lookback + 60} "
            f"needed ({lookback_years:.1f}y training window plus a test period). "
            f"Extend the date range or shorten the training window.")
    reb = [d for d in _period_ends(idx, reb_freq) if idx.get_loc(d) >= lookback]
    if not reb:
        raise DataError("No usable rebalancing date in this range.")

    c, ter, R = costs.one_way, costs.ter_vector(returns.shape[1]), returns.values
    nav, w = 1.0, np.zeros(returns.shape[1])
    navs, dates, w_rows, w_dates = [], [], [], []
    turnover, n_failed, first = 0.0, 0, True
    pos = [idx.get_loc(d) for d in reb] + [len(idx) - 1]

    for k in range(len(reb)):
        d_pos, end_pos = pos[k], pos[k + 1]
        window = returns.iloc[max(0, d_pos - lookback):d_pos]
        try:
            if method == "equal_weight":
                w_new = _clean(np.ones(cons.n) / cons.n, cons)
            else:
                mu_w, cov_w, _ = estimate_moments(window, spec, rf)
                w_new = {"max_sharpe": lambda: solve_max_sharpe(mu_w, cov_w, cons, rf, 12),
                         "min_variance": lambda: solve_min_variance(cov_w, cons),
                         "risk_parity": lambda: solve_risk_parity(cov_w, cons, 8),
                         "min_cvar": lambda: solve_min_cvar(window, cons, 0.95)}[method]()
        except PortfolioLabError:
            n_failed += 1
            w_new = w.copy() if not first else _clean(np.ones(cons.n) / cons.n, cons)

        traded = float(np.abs(w_new - w).sum())
        nav *= (1 - traded * c)
        turnover += traded / 2
        if first:
            navs.append(1.0); dates.append(idx[d_pos]); first = False
        w = w_new.copy()
        w_rows.append(w.copy()); w_dates.append(idx[d_pos])

        for t in range(d_pos + 1, end_pos + 1):
            r = np.nan_to_num(R[t])
            gross = max(1.0 + float(w @ r), 1e-12)
            nav *= gross
            w = w * (1 + r) / gross
            nav *= (1 - float(w @ ter) / PERIODS)
            navs.append(nav); dates.append(idx[t])
        if progress:
            progress((k + 1) / len(reb), f"{k + 1}/{len(reb)}")

    nav_s = pd.Series(navs, index=pd.DatetimeIndex(dates), name=method)
    nav_s = nav_s[~nav_s.index.duplicated(keep="last")]
    return {"nav": nav_s, "returns": nav_s.pct_change().dropna(),
            "weights": pd.DataFrame(w_rows, index=pd.DatetimeIndex(w_dates),
                                    columns=returns.columns),
            "annual_turnover": turnover / max(len(nav_s) / PERIODS, 1e-9),
            "n_rebalances": len(reb), "n_failed": n_failed, "method": method}


def oos_verdict(strategy: dict, naive: dict, rf: float) -> dict:
    """Out-of-sample Sharpe gap plus a Memmel test on excess returns."""
    s = full_stats(strategy["nav"], strategy["returns"], rf)
    nv = full_stats(naive["nav"], naive["returns"], rf)
    df = pd.concat([strategy["returns"], naive["returns"]], axis=1).dropna()
    p_value = float("nan")
    if len(df) > 60:
        T = len(df)
        r1 = df.iloc[:, 0].values - rf / PERIODS
        r2 = df.iloc[:, 1].values - rf / PERIODS
        s1 = r1.mean() / r1.std(ddof=1) if r1.std(ddof=1) > 0 else 0.0
        s2 = r2.mean() / r2.std(ddof=1) if r2.std(ddof=1) > 0 else 0.0
        rho = float(np.corrcoef(r1, r2)[0, 1])
        var = (2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / T
        if var > 0:
            from scipy.stats import norm
            p_value = float(2 * (1 - norm.cdf(abs((s1 - s2) / np.sqrt(var)))))
    gap = s["sharpe"] - nv["sharpe"]
    return {"strategy": s, "naive": nv, "sharpe_gap": gap, "p_value": p_value,
            "beats_naive": bool(gap > 0),
            "significant": bool(p_value == p_value and p_value < 0.05)}


def sixty_forty(assets: Sequence[str], classes: dict) -> Optional[np.ndarray]:
    eq = [i for i, a in enumerate(assets) if classes.get(a) == "Stocks"]
    bd = [i for i, a in enumerate(assets) if classes.get(a) == "Bonds"]
    if not eq or not bd:
        return None
    w = np.zeros(len(assets))
    w[eq], w[bd] = 0.60 / len(eq), 0.40 / len(bd)
    return w


# ══ Projection, orders, factors, scenarios ══════════════════════════════

def to_monthly(r: pd.Series, min_days: int = 15) -> np.ndarray:
    """Compounded monthly returns; partial first and last months are dropped."""
    g = (1.0 + r).resample("ME")
    m = (g.prod() - 1.0)[g.count() >= min_days]
    return m.dropna().values


def block_bootstrap(monthly: np.ndarray, horizon: int, n_sims=5000, block=6,
                    seed=SEED) -> np.ndarray:
    r = np.asarray(monthly, float)
    r = r[~np.isnan(r)]
    if len(r) < 24:
        raise DataError(f"Not enough history to project: {len(r)} months "
                        f"available, 24 needed.")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(r), size=(n_sims, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % len(r)
    return r[idx.reshape(n_sims, -1)[:, :horizon]]


def money_weighted_return(cashflows: np.ndarray, terminal: float) -> float:
    """Annualised money-weighted return; final value lands on the last period."""
    cf = np.asarray(cashflows, float).copy()
    cf[-1] += terminal
    t = np.arange(len(cf))

    def npv(rate):
        return float(np.sum(cf / (1 + rate) ** t))

    lo, hi = -0.99 / MONTHS, 1.0
    if npv(lo) * npv(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if npv(lo) * npv(mid) <= 0 else (mid, hi)
    return float((1 + 0.5 * (lo + hi)) ** MONTHS - 1)


def simulate_wealth(monthly: np.ndarray, *, initial: float, monthly_contribution: float,
                    horizon_years: float, n_sims=5000, block=6, inflation=0.02,
                    tax_rate=0.0, seed=SEED) -> dict:
    """Final wealth with regular contributions, after tax and inflation."""
    H = int(round(horizon_years * MONTHS))
    if H < 1:
        raise ValueError("Horizon is too short.")
    paths = block_bootstrap(monthly, H, n_sims, block, seed)
    n = paths.shape[0]
    wealth = np.full(n, float(initial))
    peak, mdd = wealth.copy(), np.zeros(n)
    traj = np.empty((n, H + 1)); traj[:, 0] = wealth
    invested = float(initial)
    for m in range(H):
        wealth = wealth * (1 + paths[:, m]) + monthly_contribution
        invested += monthly_contribution
        peak = np.maximum(peak, wealth)
        mdd = np.minimum(mdd, wealth / np.maximum(peak, 1e-9) - 1)
        traj[:, m + 1] = wealth
    after_tax = wealth - np.maximum(wealth - invested, 0.0) * tax_rate
    qs = [0.05, 0.25, 0.50, 0.75, 0.95]
    cash = np.concatenate([[-initial], np.full(H, -monthly_contribution)])
    return {"terminal": wealth, "after_tax": after_tax,
            "real": after_tax / (1 + inflation) ** horizon_years,
            "trajectories": traj, "invested": invested, "max_drawdown": mdd,
            "quantiles": {q: float(np.quantile(wealth, q)) for q in qs},
            "quantiles_real": {q: float(np.quantile(
                after_tax / (1 + inflation) ** horizon_years, q)) for q in qs},
            "prob_loss": float((wealth < invested).mean()),
            "median_irr": money_weighted_return(cash, float(np.median(wealth))),
            "horizon_months": H}


def prob_reach_goal(terminal, goal) -> float:
    return float((np.asarray(terminal) >= goal).mean())


def build_execution_plan(assets, w_target, budget: float, prices: dict, *,
                         lot_size: int = 1, allow_fractional=False,
                         costs: Optional[CostModel] = None) -> pd.DataFrame:
    """Order quantities a broker would accept, leftover cash and drift from target."""
    costs = costs or CostModel()
    rows, invested, missing = [], 0.0, []
    for i, a in enumerate(assets):
        target = budget * float(w_target[i])
        px = prices.get(a)
        if not px or px <= 0 or not np.isfinite(px):
            missing.append(a)
            rows.append({"Asset": a, "Target weight": float(w_target[i]),
                         "Target amount": target, "Price": np.nan,
                         "Quantity": np.nan, "Amount invested": 0.0})
            continue
        qty = target / px if allow_fractional else np.floor(target / (px * lot_size)) * lot_size
        invested += qty * px
        rows.append({"Asset": a, "Target weight": float(w_target[i]),
                     "Target amount": target, "Price": float(px),
                     "Quantity": float(qty), "Amount invested": float(qty * px)})
    df = pd.DataFrame(rows)
    df["Actual weight"] = df["Amount invested"] / budget
    df["Drift (pts)"] = (df["Actual weight"] - df["Target weight"]) * 100
    fees = invested * costs.one_way
    df.attrs.update(cash=float(budget - invested - fees), fees=float(fees),
                    invested_ratio=float(invested / budget) if budget else 0.0,
                    max_gap=float(df["Drift (pts)"].abs().max()) if len(df) else 0.0,
                    missing=missing)
    return df


def factor_regression(port: pd.Series, factors: pd.DataFrame, rf=0.0) -> pd.DataFrame:
    df = pd.concat([port.rename("p"), factors], axis=1).dropna()
    if len(df) < 60:
        raise DataError("Not enough overlapping history for the factor regression.")
    y = df["p"].values - rf / PERIODS
    X = df.drop(columns=["p"]).values - rf / PERIODS
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    s2 = float(resid @ resid) / max(len(y) - X1.shape[1], 1)
    se = np.sqrt(np.maximum(np.diag(s2 * np.linalg.pinv(X1.T @ X1)), 0.0))
    t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    ss = float(((y - y.mean()) ** 2).sum())
    out = pd.DataFrame({"Factor": ["Alpha (annual)"] + list(df.columns[1:]),
                        "Coefficient": [beta[0] * PERIODS] + list(beta[1:]),
                        "t-stat": t, "Significant": np.abs(t) > 1.96})
    out.attrs["r2"] = 1 - float(resid @ resid) / ss if ss > 0 else float("nan")
    return out


def run_scenario(prices_sc: pd.DataFrame, weights: dict, assets: Sequence[str],
                 name: str, meta: tuple, *, rebalance="none", costs=None) -> dict:
    """Crisis statistics; weights are rescaled over the assets that existed then."""
    if prices_sc is None or prices_sc.empty:
        return {}
    avail = [a for a in assets
             if a in prices_sc.columns and prices_sc[a].notna().sum() >= 5]
    if not avail:
        return {}
    p = prices_sc[avail].ffill().dropna(how="any")
    if len(p) < 5:
        return {}
    rets = simple_returns(p)
    if rets.empty:
        return {}
    idx = [list(assets).index(a) for a in avail]
    out = {}
    for pname, wf in weights.items():
        w = np.asarray(wf, float)[idx]
        if w.sum() < 1e-9:
            continue
        path = portfolio_path(rets, w / w.sum(), rebalance=rebalance, costs=costs,
                              charge_initial=False)
        out[pname] = {"total_return": float(path["nav"].iloc[-1] - 1),
                      "max_drawdown": max_drawdown(path["nav"]),
                      "volatility": ann_vol_from_series(path["returns"]),
                      "cvar95": historical_cvar(path["returns"]),
                      "cum_series": path["nav"] * 100}
    astats = {a: {"total_return": float(p[a].iloc[-1] / p[a].iloc[0] - 1),
                  "max_drawdown": max_drawdown(p[a])}
              for a in avail if p[a].notna().sum() >= 2}
    return {"portfolios": out, "assets": astats, "prices": p, "available": avail,
            "missing": [a for a in assets if a not in avail], "name": name, "meta": meta}


# ══ Data ═════════════════════════════════════════════════════════════════
# yfinance est une source non contractuelle : pour un usage professionnel,
# remplacer fetch_prices / fetch_fx par un fournisseur sous contrat.

def _require_yf():
    if not HAS_YF:
        raise DataError("yfinance requis : pip install yfinance")


@cache_data(show_spinner=False, ttl=3600)
def fetch_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    _require_yf()
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True,
                      progress=False, threads=True)
    if raw is None or raw.empty:
        raise DataError(f"No data for {', '.join(tickers)} between {start} and {end}.")
    if isinstance(raw.columns, pd.MultiIndex):
        lvl = raw.columns.get_level_values(0)
        raw = raw["Close"] if "Close" in lvl else raw[lvl[0]]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    raw = raw.ffill(limit=5)
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    return raw.dropna(how="all").sort_index()


@cache_data(show_spinner=False, ttl=3600)
def fetch_fx(currencies: tuple, base: str, start: str, end: str) -> pd.DataFrame:
    """rate[X] = units of `base` per 1 unit of X. Fails loudly if unavailable."""
    _require_yf()
    needed = {c for c in currencies if c}
    usd_per = {}
    for c in needed | {base}:
        if c == "USD":
            continue
        s = None
        for sym, inv in ((f"{c}USD=X", False), (f"USD{c}=X", True)):
            try:
                d = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
                if d is not None and not d.empty:
                    col = d["Close"]
                    col = col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col
                    col = col.dropna()
                    if len(col) > 5:
                        s = 1.0 / col if inv else col
                        break
            except Exception:
                continue
        if s is None:
            raise FXError(
                f"No {c}/USD exchange rate for this period, so returns cannot be "
                f"converted to {base}. Remove that asset, change your base "
                f"currency, or switch on the static fallback rate.")
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        usd_per[c] = s
    if not usd_per:
        return pd.DataFrame()
    base_usd = (pd.Series(1.0, index=usd_per[next(iter(usd_per))].index)
                if base == "USD" else usd_per[base])
    out = {}
    for c in needed:
        cs = pd.Series(1.0, index=base_usd.index) if c == "USD" else usd_per[c]
        df = pd.concat([cs.rename("x"), base_usd.rename("b")], axis=1).ffill().dropna()
        out[c] = df["x"] / df["b"]
    return pd.DataFrame(out).sort_index()


def convert_to_base(prices: pd.DataFrame, ccy_map: dict, base: str,
                    fx: Optional[pd.DataFrame], allow_static=False) -> tuple[pd.DataFrame, dict]:
    """Convert BEFORE computing returns, otherwise the covariance mixes currencies."""
    foreign = {a: c for a, c in ccy_map.items() if c and c != base}
    info = {"converted": list(foreign), "method": "none", "base": base}
    if not foreign:
        return prices.copy(), info
    out = prices.copy()
    if fx is not None and not fx.empty:
        al = fx.reindex(out.index).ffill().bfill()
        missing = [c for c in set(foreign.values())
                   if c not in al.columns or al[c].isna().all()]
        if missing and not allow_static:
            raise FXError(f"Missing exchange rate series for: {', '.join(sorted(missing))}.")
        for a, c in foreign.items():
            if a not in out.columns:
                continue
            if c in al.columns and not al[c].isna().all():
                out[a] = out[a] * al[c]
            elif allow_static:
                out[a] = out[a] * (FX_FALLBACK.get(c, 1.0) / FX_FALLBACK.get(base, 1.0))
        info["method"] = "series"
        return out, info
    if not allow_static:
        raise FXError("Multi-currency portfolio with no exchange rate data. Switch "
                      "on the static fallback to continue with approximate amounts.")
    for a, c in foreign.items():
        out[a] = out[a] * (FX_FALLBACK.get(c, 1.0) / FX_FALLBACK.get(base, 1.0))
    info["method"] = "static"
    return out, info


def classify_asset(ticker: str, quote_type: str = "") -> str:
    """Asset class - what the money is exposed to."""
    t, q = ticker.upper(), (quote_type or "").upper()
    if q == "CRYPTOCURRENCY" or t.endswith(("-USD", "-EUR")):
        return "Crypto"
    for cls, hints in CLASS_HINTS.items():
        if any(t == h or t.startswith(h + ".") for h in hints):
            return cls
    if q in ("CURRENCY", "FOREX", "CASH", "MONEYMARKET"):
        return "Cash"
    # Accepts both the yfinance vocabulary (EQUITY, MUTUALFUND) and the
    # catalogue's (Stock, Fund).
    if q in ("EQUITY", "STOCK", "ETF", "MUTUALFUND", "FUND", "INDEX", "DR", ""):
        return "Stocks"
    return "Other"


def instrument_type(ticker: str, quote_type: str = "") -> str:
    """Instrument type - what you actually buy."""
    q = (quote_type or "").upper()
    if q in INSTRUMENT:
        return INSTRUMENT[q]
    if q.title() in ("Stock", "Etf", "Fund", "Index", "Crypto", "Forex", "Future", "Cash"):
        return "ETF" if q.upper() == "ETF" else q.title()
    t = ticker.upper()
    if t.endswith(("-USD", "-EUR")):
        return "Crypto"
    if t.startswith("^"):
        return "Index"
    if t.endswith("=X"):
        return "Forex"
    if t.endswith("=F"):
        return "Future"
    return "Action"


def pea_eligible(country: str = "", exchange: str = "", asset_type: str = "Stock") -> bool:
    """French PEA eligibility depends on where the issuer is based, not only on
    where it lists. Country comes from the catalogue when available."""
    if asset_type not in ("Stock", "ETF", "Fund"):
        return False
    if country:
        return country in EU_COUNTRIES
    eu_ex = {"PAR", "AMS", "BRU", "LIS", "MIL", "MCE", "GER", "FRA", "XETRA", "EBS",
             "STO", "CPH", "HEL", "OSL", "DUB", "VIE", "WSE"}
    return (exchange or "").upper() in eu_ex


@cache_data(ttl=1800, show_spinner=False)
def validate_ticker(ticker: str, deep: bool = False) -> dict:
    """fast_info by default (quick); the slower .info only when `deep`."""
    _require_yf()
    try:
        tk = yf.Ticker(ticker)
        fi = tk.fast_info
        price = getattr(fi, "last_price", None)
        if price is None or not np.isfinite(float(price)) or float(price) <= 0:
            return {"valid": False, "ticker": ticker, "name": ticker,
                    "error": "No quote returned"}
        out = {"valid": True, "ticker": ticker, "name": ticker,
               "currency": (getattr(fi, "currency", None) or "USD").upper(),
               "exchange": getattr(fi, "exchange", "") or "",
               "quote_type": (getattr(fi, "quote_type", None) or "").upper(),
               "price": float(price)}
        if deep or not out["quote_type"]:
            try:
                info = tk.info
                out["name"] = info.get("longName") or info.get("shortName") or ticker
                out["quote_type"] = (info.get("quoteType") or out["quote_type"]).upper()
                out["exchange"] = info.get("exchange", out["exchange"])
                out["currency"] = (info.get("currency") or out["currency"]).upper()
                out["market_cap"] = info.get("marketCap")
            except Exception:
                pass
        return out
    except Exception as e:
        return {"valid": False, "ticker": ticker, "name": ticker, "error": str(e)[:120]}


@cache_data(ttl=1800, show_spinner=False)
def asset_profile(ticker: str) -> tuple[str, str]:
    """(instrument type, asset class) - used to build the constraint groups."""
    q = ""
    if HAS_YF:
        try:
            q = validate_ticker(ticker).get("quote_type", "")
        except Exception:
            q = ""
    return instrument_type(ticker, q), classify_asset(ticker, q)


@cache_data(ttl=3600, show_spinner=False)
def fetch_market_caps(tickers: tuple) -> dict:
    out = {}
    for t in tickers:
        mc = validate_ticker(t, deep=True).get("market_cap")
        if mc and mc > 0:
            out[t] = float(mc)
    return out


@cache_data(ttl=300, show_spinner=False)
def fetch_risk_free(currency: str) -> tuple[float, str]:
    cfg = CURRENCIES.get(currency, {})
    proxy = cfg.get("proxy")
    if proxy and HAS_YF:
        try:
            d = yf.download(proxy, period="1mo", progress=False, auto_adjust=True)
            if d is not None and not d.empty:
                v = float(d["Close"].dropna().iloc[-1]) / 100
                if 0 <= v < 0.25:
                    return v, f"live market ({proxy})"
        except Exception:
            pass
    return float(cfg.get("rf", 0.02)), "default assumption"


@cache_data(ttl=60, show_spinner=False)
def fetch_last_prices(tickers: tuple) -> dict:
    _require_yf()
    try:
        raw = yf.download(list(tickers), period="5d", progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return {}
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        close = close.ffill().iloc[-1]
        return {str(t): float(close[t]) for t in tickers
                if t in close.index and pd.notna(close[t])}
    except Exception:
        return {}


def data_quality_report(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change()
    rows = []
    for c in prices.columns:
        s, r = prices[c], rets[c].dropna()
        gaps, stale = int(s.isna().sum()), int((r.abs() < 1e-12).sum())
        jumps = int((r.abs() > 0.35).sum())
        flags = []
        if len(s.dropna()) < 250:
            flags.append("under 1 year of history")
        if gaps > len(s) * 0.05:
            flags.append("more than 5% missing")
        if stale > len(r) * 0.08:
            flags.append(f"{stale/max(len(r),1):.0%} of sessions unchanged")
        if jumps:
            flags.append(f"{jumps} jump(s) over 35%")
        if r.std() < 1e-9:
            flags.append("zero variance")
        rows.append({"Asset": c, "Sessions": int(s.notna().sum()),
                     "Starts": str(s.dropna().index.min().date()) if s.notna().any() else "-",
                     "Gaps": gaps, "Flat days": stale, "Jumps > 35%": jumps,
                     "Warnings": " · ".join(flags) or "✅ clean"})
    return pd.DataFrame(rows)


def align_common_history(prices: pd.DataFrame, min_obs=252) -> tuple[pd.DataFrame, list]:
    kept = [c for c in prices.columns if prices[c].notna().sum() >= min_obs]
    dropped = [c for c in prices.columns if c not in kept]
    if len(kept) < 2:
        raise DataError(f"Fewer than two assets have {min_obs} sessions of history. "
                        f"Dropped: {', '.join(dropped) or '-'}.")
    sub = prices[kept].dropna(how="any")
    if len(sub) < min_obs:
        starts = {c: prices[c].dropna().index.min() for c in kept}
        late = max(starts, key=starts.get)
        raise DataError(f"Only {len(sub)} sessions are common to every asset. The "
                        f"most recent one is {late}, which starts "
                        f"{starts[late].date()}. Remove it or start later.")
    return sub, dropped


def data_fingerprint(prices: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(",".join(map(str, prices.columns)).encode())
    h.update(f"{prices.index.min()}{prices.index.max()}".encode())
    h.update(np.ascontiguousarray(prices.values, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def build_audit(params: dict, prices: pd.DataFrame, weights: dict, stats: dict) -> dict:
    return {
        "run_id": hashlib.sha256((str(datetime.now(timezone.utc))
                                  + json.dumps(params, default=str)).encode()).hexdigest()[:12],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_fingerprint": data_fingerprint(prices),
        "period": {"start": str(prices.index.min().date()),
                   "end": str(prices.index.max().date()), "observations": len(prices)},
        "parameters": params,
        "weights": {k: [round(float(x), 6) for x in v] for k, v in weights.items()},
        "statistics": {k: {kk: round(float(vv), 6) for kk, vv in v.items()
                           if isinstance(vv, (int, float, np.floating))}
                       for k, v in stats.items()},
        "disclaimer": "Educational simulation, not investment advice.",
    }


# ══ Instrument catalogue (financedatabase) ══════════════════════════════════
# Offline catalogue of ~300k instruments. It carries sector, country and ISIN,
# which neither Yahoo nor TradingView expose cleanly, and its symbols are in
# Yahoo format - so prices still come from yfinance with dividends reinvested.

try:
    import financedatabase as fd
    HAS_FD = True
except Exception:
    fd, HAS_FD = None, False

CAT_COLS = ["symbol", "name", "type", "exchange", "currency", "sector",
            "country", "market_cap", "isin"]
MAJOR_EXCHANGES = {"NMS", "NAS", "NASDAQ", "NYQ", "NYSE", "NGM", "PCX", "ASE", "BTS",
                   "PAR", "AMS", "BRU", "LIS", "MIL", "MCE", "GER", "FRA", "STU",
                   "LSE", "IOB", "EBS", "STO", "CPH", "HEL", "OSL", "VIE", "DUB",
                   "TOR", "VAN", "ASX", "JPX", "TYO", "HKG", "SHH", "SHZ", "CCC",
                   "CCY", "SNP", "WCB"}
EU_COUNTRIES = {"France", "Germany", "Netherlands", "Belgium", "Spain", "Italy",
                "Portugal", "Ireland", "Austria", "Finland", "Sweden", "Denmark",
                "Luxembourg", "Poland", "Greece", "Czech Republic", "Hungary",
                "Norway", "Iceland", "Estonia", "Latvia", "Lithuania", "Slovakia",
                "Slovenia", "Croatia", "Bulgaria", "Romania", "Malta", "Cyprus"}


@cache_data(show_spinner=False, ttl=86400)
def load_catalogue(major_only: bool = True) -> pd.DataFrame:
    """Flatten financedatabase into one searchable frame."""
    if not HAS_FD:
        raise DataError("Catalogue needs financedatabase: pip install financedatabase")
    loaders = {"Stock": "Equities", "ETF": "ETFs", "Fund": "Funds",
               "Index": "Indices", "Crypto": "Cryptos", "Forex": "Currencies",
               "Cash": "Moneymarkets"}
    frames = []
    for typ, cls_name in loaders.items():
        cls = getattr(fd, cls_name, None)
        if cls is None:
            continue
        try:
            df = cls().select()
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "symbol"})
        out = pd.DataFrame({"symbol": df["symbol"].astype(str).str.upper()})
        out["name"] = df.get("name", pd.Series("", index=df.index)).fillna("").astype(str)
        out["type"] = typ
        for col, src in (("exchange", "exchange"), ("currency", "currency"),
                         ("country", "country"), ("market_cap", "market_cap"),
                         ("isin", "isin")):
            out[col] = df.get(src, pd.Series("", index=df.index)).fillna("").astype(str)
        # Equities carry a sector; funds and ETFs carry a category instead.
        sector = df.get("sector")
        if sector is None:
            sector = df.get("category_group", df.get("category"))
        out["sector"] = (sector.fillna("").astype(str) if sector is not None
                         else pd.Series("", index=df.index))
        frames.append(out)
    if not frames:
        raise DataError("financedatabase returned nothing.")
    cat = pd.concat(frames, ignore_index=True)
    cat = cat[cat["symbol"].str.len().between(1, 24)]
    if major_only:
        cat = cat[cat["exchange"].str.upper().isin(MAJOR_EXCHANGES)
                  | cat["type"].isin(["Crypto", "Forex", "Index"])]
    cat["currency"] = cat["currency"].str.upper().replace("", "USD")
    cat["haystack"] = (cat["symbol"] + " " + cat["name"]).str.lower()
    cat["label"] = (cat["symbol"] + " · " + cat["name"].str.slice(0, 40)
                    + " · " + cat["exchange"] + " · " + cat["type"])
    return cat.drop_duplicates("symbol").reset_index(drop=True)


def search_catalogue(cat: pd.DataFrame, query: str, types=None, limit=40) -> list[dict]:
    """Substring search, exact symbol matches first."""
    if cat is None or cat.empty or not query or len(query) < 2:
        return []
    df = cat if not types else cat[cat["type"].isin(types)]
    q = query.strip().lower()
    hit = df[df["haystack"].str.contains(re.escape(q), regex=True, na=False)]
    if hit.empty:
        return []
    exact = hit["symbol"].str.lower() == q
    hit = pd.concat([hit[exact], hit[~exact]]).head(limit)
    return hit[CAT_COLS + ["label"]].to_dict("records")


def catalogue_holding(row: dict) -> dict:
    """Catalogue row -> holding. Symbols are Yahoo-format, so prices via yfinance."""
    return {"symbol": row["symbol"], "exchange": row.get("exchange", ""),
            "description": row.get("name", row["symbol"]), "type": row.get("type", "Stock"),
            "currency": row.get("currency", "USD") or "USD", "source": "yahoo",
            "sector": row.get("sector", "") or "Unclassified",
            "country": row.get("country", "") or "Unknown",
            "isin": row.get("isin", ""),
            "asset_class": classify_asset(row["symbol"], row.get("type", "").upper())}


# ══ TradingView symbol search ═══════════════════════════════════════════════
# tvDatafeed gives a real searchable universe (stocks, ETFs, funds, crypto,
# forex, futures) instead of asking people to guess ticker strings.

try:
    from tvDatafeed import TvDatafeed, Interval
    HAS_TV = True
except Exception:
    try:
        from tvdatafeed import TvDatafeed, Interval
        HAS_TV = True
    except Exception:
        TvDatafeed, Interval, HAS_TV = None, None, False

TV_TYPE_MAP = {"stock": "Stock", "dr": "Stock", "fund": "ETF", "etf": "ETF",
               "index": "Index", "crypto": "Crypto", "forex": "Forex",
               "futures": "Future", "bond": "Bond", "economic": "Other"}


def _tv_client():
    if not HAS_TV:
        raise DataError("TradingView search needs tvDatafeed: pip install tvdatafeed")
    if HAS_ST:
        if "_tv" not in st.session_state:
            st.session_state._tv = TvDatafeed()
        return st.session_state._tv
    return TvDatafeed()


@cache_data(show_spinner=False, ttl=900)
def tv_search(query: str, exchange: str = "") -> list[dict]:
    """Search the TradingView universe. Returns display-ready rows."""
    if not query or len(query) < 2:
        return []
    try:
        raw = _tv_client().search_symbol(query, exchange=exchange or "")
    except Exception as e:
        raise DataError(f"TradingView search failed: {str(e)[:120]}")
    out, seen = [], set()
    for r in raw or []:
        sym, exch = (r.get("symbol") or "").upper(), (r.get("exchange") or "").upper()
        if not sym or not exch or (exch, sym) in seen:
            continue
        seen.add((exch, sym))
        desc = re.sub(r"<[^>]+>", "", r.get("description") or "")
        typ = TV_TYPE_MAP.get((r.get("type") or "").lower(), "Other")
        ccy = (r.get("currency_code") or "USD").upper()
        out.append({"key": f"{exch}:{sym}", "symbol": sym, "exchange": exch,
                    "description": desc, "type": typ, "currency": ccy,
                    "source": "tv",
                    "label": f"{sym} · {desc[:44]} · {exch} · {typ}"})
    return out[:40]


@cache_data(show_spinner=False, ttl=3600)
def tv_history(symbol: str, exchange: str, start: str, end: str) -> pd.Series:
    n = int((pd.Timestamp(end) - pd.Timestamp(start)).days * 0.72) + 60
    df = _tv_client().get_hist(symbol=symbol, exchange=exchange,
                               interval=Interval.in_daily, n_bars=min(max(n, 120), 5000))
    if df is None or df.empty:
        raise DataError(f"TradingView returned no history for {exchange}:{symbol}.")
    s = df["close"].copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s.loc[str(start):str(end)].dropna()


@cache_data(show_spinner=False, ttl=3600)
def fetch_universe(spec: tuple, start: str, end: str) -> pd.DataFrame:
    """spec = ((key, source, symbol, exchange), ...) -> daily close, one column per key."""
    cols, failed = {}, []
    yahoo = [t for t in spec if t[1] == "yahoo"]
    if yahoo:
        try:
            df = fetch_prices(tuple(t[2] for t in yahoo), start, end)
            for key, _, sym, _ in yahoo:
                if sym in df.columns:
                    cols[key] = df[sym]
                else:
                    failed.append(key)
        except DataError:
            failed += [t[0] for t in yahoo]
    for key, src, sym, exch in spec:
        if src != "tv":
            continue
        try:
            cols[key] = tv_history(sym, exch, start, end)
        except Exception:
            failed.append(key)
    if len(cols) < 2:
        raise DataError(
            f"Could not download usable history for at least two holdings. "
            f"Failed: {', '.join(failed) or 'all'}. Try a different date range "
            f"or another data source.")
    out = pd.DataFrame(cols).sort_index()
    out.attrs["failed"] = failed
    return out


# ══ Charts ══════════════════════════════════════════════════════════════════

def layout(title="", x="", y="", **kw) -> dict:
    ax = dict(gridcolor=GRID, zerolinecolor=GRID, color=MUTED, linecolor=GRID,
              title_font=dict(color=MUTED, size=11), tickfont=dict(color=MUTED, size=10))
    return dict(paper_bgcolor=BG, plot_bgcolor=PANEL,
                font=dict(color=TEXT, family="Inter, sans-serif"),
                title=dict(text=title, font=dict(size=14, color=TEXT)),
                xaxis=dict(title=x, **ax), yaxis=dict(title=y, **ax),
                legend=dict(bgcolor="rgba(10,18,32,.85)", bordercolor=GRID,
                            borderwidth=1, font=dict(size=11, color=TEXT)),
                margin=dict(l=58, r=25, t=56, b=50), **kw)


def chart_pie(w, assets, title, sub="") -> "go.Figure":
    mask = np.asarray(w) > 0.005
    fig = go.Figure()
    if not mask.any():
        fig.add_annotation(text="No meaningful weights", x=.5, y=.5, xref="paper",
                           yref="paper", showarrow=False, font=dict(color=TEXT))
    else:
        fig.add_trace(go.Pie(labels=[assets[i] for i in range(len(assets)) if mask[i]],
                             values=np.asarray(w)[mask], hole=.5,
                             marker=dict(colors=PALETTE, line=dict(color=BG, width=2.5)),
                             textinfo="label+percent", textfont=dict(size=13),
                             hovertemplate="%{label}: %{percent}<extra></extra>"))
    fig.update_layout(**layout(f"{title}{'  ·  ' + sub if sub else ''}", height=400))
    return fig


def chart_lines(series: dict, title, y="Value", height=430, base100=False) -> "go.Figure":
    fig = go.Figure()
    for name, s in series.items():
        v = s.values * 100 if base100 else s.values
        fig.add_trace(go.Scatter(x=s.index, y=v, mode="lines", name=name,
                                 line=dict(color=PORT_COLORS.get(name, GOLD),
                                           width=2.4,
                                           dash="dot" if "1/N" in name or "60/40" in name
                                           else "solid")))
    fig.update_layout(**layout(title, "Date", y, height=height))
    return fig


def chart_fan(traj: np.ndarray, contributed: np.ndarray, sym: str) -> "go.Figure":
    m = np.arange(traj.shape[1])
    q = {k: np.quantile(traj, k, axis=0) for k in (.05, .25, .5, .75, .95)}
    fig = go.Figure()
    for lo, hi, op in ((.05, .95, .12), (.25, .75, .22)):
        fig.add_trace(go.Scatter(x=np.concatenate([m, m[::-1]]),
                                 y=np.concatenate([q[hi], q[lo][::-1]]),
                                 fill="toself", fillcolor=f"rgba(212,175,55,{op})",
                                 line=dict(width=0), hoverinfo="skip",
                                 name=f"{int(lo*100)}-{int(hi*100)}th percentile"))
    fig.add_trace(go.Scatter(x=m, y=q[.5], mode="lines", name="Median outcome",
                             line=dict(color=GOLD, width=2.6)))
    fig.add_trace(go.Scatter(x=m, y=contributed, mode="lines", name="Money you put in",
                             line=dict(color=MUTED, width=1.6, dash="dash")))
    fig.update_layout(**layout("Where your money could end up", "Months from now",
                               f"Portfolio value ({sym})", height=440))
    return fig


def chart_hist(terminal, invested, goal, sym) -> "go.Figure":
    fig = go.Figure(go.Histogram(x=terminal, nbinsx=60,
                                 marker=dict(color=GOLD, opacity=.75), name="Outcomes"))
    fig.add_vline(x=invested, line=dict(color=MUTED, width=1.6, dash="dash"),
                  annotation_text="Money in", annotation_font_color=MUTED)
    fig.add_vline(x=float(np.median(terminal)), line=dict(color=GREEN, width=1.8),
                  annotation_text="Median", annotation_font_color=GREEN)
    if goal:
        fig.add_vline(x=goal, line=dict(color=BLUE, width=1.8, dash="dot"),
                      annotation_text="Your goal", annotation_font_color=BLUE)
    fig.update_layout(**layout("All simulated outcomes", f"Final value ({sym})",
                               "Simulations", height=360))
    return fig


def chart_drawdown(navs: dict) -> "go.Figure":
    fig = go.Figure()
    for name, nav in navs.items():
        dd = drawdown_series(nav) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=name,
                                 fill="tozeroy", opacity=.6,
                                 line=dict(color=PORT_COLORS.get(name, GOLD), width=1.4)))
    fig.update_layout(**layout("How far below the previous peak you would have been",
                               "Date", "Drawdown (%)"))
    return fig


def chart_bars(labels, values, title, x="", horizontal=True, fmt="{:+.1f}%",
               colors=None) -> "go.Figure":
    colors = colors or [GREEN if v >= 0 else RED for v in values]
    kw = dict(marker=dict(color=colors, opacity=.88, line=dict(color=BG, width=1)),
              text=[fmt.format(v) for v in values], textposition="outside",
              textfont=dict(color=TEXT, size=11))
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", **kw) if horizontal
                    else go.Bar(x=labels, y=values, **kw))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1)) if horizontal else None
    fig.update_layout(**layout(title, x, "", height=max(300, len(labels) * 46 + 130)))
    return fig


def chart_uncertainty(assets, mu, se) -> "go.Figure":
    o = np.argsort(mu)
    fig = go.Figure(go.Scatter(
        x=mu[o] * 100, y=[assets[i] for i in o], mode="markers",
        marker=dict(size=11, color=GOLD, line=dict(color=BG, width=1.5)),
        error_x=dict(type="data", array=1.96 * se[o] * 100, thickness=1.6, width=6,
                     color="rgba(212,175,55,.45)"), name="Estimate ± 95% range"))
    fig.add_vline(x=0, line=dict(color=RED, width=1.2, dash="dash"))
    fig.update_layout(**layout("Expected return, and how unsure we are about it",
                               "Annual return (%)", "",
                               height=max(300, len(assets) * 44 + 130)))
    return fig


def chart_corr(returns) -> "go.Figure":
    c = returns.corr()
    z = np.round(c.values, 2)
    fig = go.Figure(go.Heatmap(z=z, x=c.columns.tolist(), y=c.index.tolist(),
                               colorscale=[[0, RED], [.5, "#1A2540"], [1, GOLD]],
                               zmid=0, zmin=-1, zmax=1, text=z,
                               texttemplate="%{text:.2f}", textfont=dict(size=11),
                               colorbar=dict(tickfont=dict(color=MUTED))))
    fig.update_layout(**layout("Do these holdings move together?",
                               height=max(360, len(c.columns) * 60 + 90)))
    return fig


def chart_frontier(mu, cov, rf, assets, weights, cloud, frontier) -> "go.Figure":
    mc_r, mc_v, mc_s = cloud
    fv, fr = frontier
    vols = np.sqrt(np.diag(cov))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=mc_v * 100, y=mc_r * 100, mode="markers", opacity=.3,
                               marker=dict(size=3, color=mc_s, colorscale="RdYlGn",
                                           showscale=True,
                                           colorbar=dict(title="Sharpe", x=.99,
                                                         thickness=12)),
                               name="Random portfolios"))
    if len(fv):
        fig.add_trace(go.Scatter(x=fv * 100, y=fr * 100, mode="lines", name="Best possible mixes",
                                 line=dict(color=GOLD, width=2.5)))
    shapes = {"Max Sharpe": ("star", 24), "Min Variance": ("diamond", 18),
              "Risk Parity": ("pentagon", 18), "Min CVaR": ("hexagon", 17),
              "Resampled": ("cross", 17)}
    for name, w in weights.items():
        r, v, sh = pmetrics(w, mu, cov, rf)
        sym, size = shapes.get(name, ("circle", 15))
        fig.add_trace(go.Scatter(x=[v * 100], y=[r * 100], mode="markers",
                                 marker=dict(size=size, symbol=sym,
                                             color=PORT_COLORS.get(name, GOLD),
                                             line=dict(color=BG, width=2)),
                                 name=f"{name} · Sharpe {sh:.2f}"))
    for i, a in enumerate(assets):
        fig.add_trace(go.Scatter(x=[vols[i] * 100], y=[mu[i] * 100], mode="markers+text",
                                 marker=dict(size=10, symbol="circle-open",
                                             color=PALETTE[i % len(PALETTE)],
                                             line=dict(width=2.2)),
                                 text=[a], textposition="top right", showlegend=False,
                                 textfont=dict(color=TEXT, size=10), name=a))
    fig.update_layout(**layout("Risk versus reward for every possible mix",
                               "Annual volatility (%)", "Annual return (%)", height=540))
    return fig


def chart_factors(fac: pd.DataFrame) -> "go.Figure":
    body = fac[fac["Factor"] != "Alpha (annual)"]
    fig = go.Figure(go.Bar(
        x=body["Coefficient"], y=body["Factor"], orientation="h",
        marker=dict(color=[GOLD if sig else MUTED for sig in body["Significant"]],
                    opacity=.88, line=dict(color=BG, width=1)),
        text=[f"{b:.2f} (t={t:.1f})" for b, t in zip(body["Coefficient"], body["t-stat"])],
        textposition="outside", textfont=dict(color=TEXT, size=11)))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(**layout(
        f"Factor exposure  ·  R² = {fac.attrs.get('r2', float('nan')):.2f}",
        "Beta", "", height=340))
    return fig


def chart_rolling(returns: dict, rf: float, window: int) -> "go.Figure":
    fig = go.Figure()
    for name, r in returns.items():
        rs = ((r.rolling(window).mean() * PERIODS - rf)
              / (r.rolling(window).std(ddof=1) * np.sqrt(PERIODS)))
        fig.add_trace(go.Scatter(x=rs.index, y=rs.values, mode="lines", name=name,
                                 line=dict(color=PORT_COLORS.get(name, GOLD), width=1.7)))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_layout(**layout(f"Rolling {window}-day Sharpe ratio", "Date", "Sharpe",
                               height=380))
    return fig


def chart_box(assets, W) -> "go.Figure":
    fig = go.Figure()
    for i, a in enumerate(assets):
        fig.add_trace(go.Box(y=W[:, i] * 100, name=a, boxpoints=False,
                             marker_color=PALETTE[i % len(PALETTE)]))
    fig.update_layout(**layout("How much the 'optimal' weights move when the data changes",
                               "", "Weight (%)", height=400, showlegend=False))
    return fig


# ══ Analysis pipeline ═══════════════════════════════════════════════════════

def build_constraints(assets, types: dict, classes: dict, p: dict) -> Constraints:
    """Per-asset bounds + guardrails + min/max limits per instrument type and class."""
    n = len(assets)
    min_w, max_w = np.zeros(n), np.ones(n)
    for i, a in enumerate(assets):
        if a in p.get("per_asset", {}):
            min_w[i], max_w[i] = p["per_asset"][a]
    groups: dict[str, tuple[list[int], float, float]] = {}
    relaxed = []
    if p.get("apply_profile", True):
        cap, risky_cap = RISK_PROFILES[p["profile"]][:2]
        risky = [i for i, a in enumerate(assets)
                 if classes.get(a) in ("Stocks", "Crypto", "Real estate")]
        n_safe = n - len(risky)
        # A guardrail must never make the portfolio impossible: with few holdings,
        # a 25% cap on 4 lines plus a 40% risky cap cannot reach 100%. Widen the
        # caps to the smallest values that still add up, and say so.
        if n_safe == 0 and risky_cap < 1.0:
            risky_cap, _ = 1.0, relaxed.append("risky-asset cap lifted (nothing to hold instead)")
        need = max(1.0 / n, (1.0 - risky_cap) / n_safe if n_safe else 0.0)
        if need > cap + 1e-9:
            relaxed.append(f"per-holding cap widened to {need:.0%} so the mix can add up")
            cap = need
        max_w = np.minimum(max_w, cap)
        if risky and risky_cap < 1.0:
            groups["Risky assets"] = (risky, 0.0, float(risky_cap))

    for label, mapping, limits in (("Type", types, p.get("type_limits", {})),
                                   ("Class", classes, p.get("class_limits", {})),
                                   ("Sector", p.get("sectors", {}), p.get("sector_limits", {})),
                                   ("Country", p.get("countries", {}), p.get("country_limits", {}))):
        for g, (gmin, gmax) in limits.items():
            idx = [i for i, a in enumerate(assets) if mapping.get(a) == g]
            if idx and (gmin > 0 or gmax < 1):
                groups[f"{label}: {g}"] = (idx, float(gmin), float(gmax))

    cons = Constraints(n, min_w, max_w, groups, p.get("max_assets"))
    check_feasibility(cons)
    cons.relaxed = relaxed
    return cons


def recommended_portfolio(profile: str, available: list[str]) -> str:
    """One clear answer instead of five competing ones."""
    order = {"Cautious": ["Min Variance", "Risk Parity", "Resampled", "Max Sharpe"],
             "Balanced": ["Risk Parity", "Resampled", "Min Variance", "Max Sharpe"],
             "Growth": ["Resampled", "Risk Parity", "Max Sharpe", "Min Variance"],
             "Aggressive": ["Resampled", "Max Sharpe", "Risk Parity", "Min Variance"]}
    for name in order.get(profile, order["Balanced"]):
        if name in available:
            return name
    return available[0]


def run_analysis(p: dict, log: Callable[[str], None]) -> dict:
    holdings = p["holdings"]
    if len(holdings) < 2:
        raise PortfolioLabError("Add at least two holdings before running.")
    if p["start"] >= p["end"]:
        raise PortfolioLabError("The start date must come before the end date.")

    log("Downloading price history…")
    spec = tuple((k, h["source"], h["symbol"], h.get("exchange", ""))
                 for k, h in holdings.items())
    raw = fetch_universe(spec, str(p["start"]), str(p["end"]))
    failed_dl = list(raw.attrs.get("failed", []))
    ccy_map = {k: holdings[k].get("currency", "USD") for k in raw.columns}

    fx = None
    foreign = sorted({c for c in ccy_map.values() if c != p["base_ccy"]})
    if foreign:
        log(f"Converting to {p['base_ccy']} ({', '.join(foreign)})…")
        try:
            fx = fetch_fx(tuple(sorted(set(ccy_map.values()))), p["base_ccy"],
                          str(p["start"]), str(p["end"]))
        except FXError:
            if not p["allow_static_fx"]:
                raise
    prices, fx_info = convert_to_base(raw, ccy_map, p["base_ccy"], fx, p["allow_static_fx"])

    prices, dropped = align_common_history(prices)
    assets = list(prices.columns)
    types = {a: holdings[a].get("type", "Stock") for a in assets}
    sectors = {a: holdings[a].get("sector", "Unclassified") for a in assets}
    countries = {a: holdings[a].get("country", "Unknown") for a in assets}
    p = {**p, "sectors": sectors, "countries": countries}
    classes = {a: holdings[a].get("asset_class")
               or classify_asset(holdings[a]["symbol"], holdings[a].get("quote_type", ""))
               for a in assets}
    quality = data_quality_report(prices)

    log("Estimating risk and return…")
    rets = simple_returns(prices)
    freq = p.get("est_freq", "auto")
    if freq == "auto":
        multi = len({holdings[a].get("exchange", "") for a in assets}) > 1 or len(foreign) > 0
        freq = "W" if multi else "D"
    spec_m = ModelSpec(mu_method=p["mu_method"], cov_method=p["cov_method"], est_freq=freq)

    bl_has_views = False
    if p["mu_method"] == "black_litterman":
        conf = np.array([float(p.get("bl_conf", {}).get(a, 0.0)) for a in assets])
        views = np.array([float(p.get("bl_views", {}).get(a, np.nan)) if conf[i] > 0
                          else np.nan for i, a in enumerate(assets)])
        bl_has_views = bool(np.any(conf > 0))
        spec_m.bl_views, spec_m.bl_confidence = views, conf
    mu, cov, diag = estimate_moments(rets, spec_m, p["rf"])

    cons = build_constraints(assets, types, classes, p)

    log("Building candidate portfolios…")
    weights, errors = {}, {}

    def add(name, solve, resolve=None):
        try:
            w = solve(cons)
            if cons.max_assets and resolve:
                w = apply_cardinality(w, cons.max_assets, resolve, cons)
            weights[name] = w
        except PortfolioLabError as e:
            errors[name] = str(e)

    add("Min Variance", lambda c: solve_min_variance(cov, c),
        lambda k: solve_min_variance(cov[np.ix_(k, k)], cons.subset(k)))
    add("Risk Parity", lambda c: solve_risk_parity(cov, c),
        lambda k: solve_risk_parity(cov[np.ix_(k, k)], cons.subset(k)))
    if p["mu_method"] != "none":
        add("Max Sharpe", lambda c: solve_max_sharpe(mu, cov, c, p["rf"]),
            lambda k: solve_max_sharpe(mu[k], cov[np.ix_(k, k)], cons.subset(k), p["rf"]))
    if p.get("do_cvar"):
        add("Min CVaR", lambda c: solve_min_cvar(rets, c, 0.95),
            lambda k: solve_min_cvar(rets.iloc[:, k], cons.subset(k), 0.95))

    W_boot = None
    if p.get("do_resample", True) and p["mu_method"] != "none":
        log("Stress-testing the weights on resampled history…")
        try:
            w_rs, W_boot = resampled_weights(rets, cons, p["rf"], spec_m, "max_sharpe", 60)
            weights["Resampled"] = w_rs
        except PortfolioLabError as e:
            errors["Resampled"] = str(e)

    weights["Equal weight (1/N)"] = _clean(np.ones(len(assets)) / len(assets), cons)
    w64 = sixty_forty(assets, classes)
    if w64 is not None:
        weights["Benchmark 60/40"] = w64
    if not weights:
        raise OptimizationError("No portfolio could be built. " + " ".join(errors.values()))

    log("Measuring performance after fees…")
    bench_key = "Benchmark 60/40" if "Benchmark 60/40" in weights else "Equal weight (1/N)"
    bench = portfolio_path(rets, weights[bench_key], rebalance=p["rebalance"],
                           costs=p["costs"])["returns"]
    stats = {k: portfolio_stats(rets, w, p["rf"], rebalance=p["rebalance"],
                                costs=p["costs"], bench=bench)
             for k, w in weights.items()}

    fv, fr, n_failed = np.array([]), np.array([]), 0
    cloud = (np.array([0.0]), np.array([0.0]), np.array([0.0]))
    if p["mu_method"] != "none":
        log("Mapping the risk/reward frontier…")
        fv, fr, n_failed = efficient_frontier(mu, cov, cons, 50)
        cloud = random_cloud(mu, cov, cons, p["rf"])

    oos = {}
    if p.get("do_oos", True):
        methods = [("Max Sharpe", "max_sharpe"), ("Min Variance", "min_variance"),
                   ("Risk Parity", "risk_parity"), ("Equal weight (1/N)", "equal_weight")]
        if p["mu_method"] == "none":
            methods = [m for m in methods if m[1] != "max_sharpe"]
        for label, m in methods:
            log(f"Reality check — {label}…")
            try:
                oos[label] = walk_forward(rets, method=m, cons=cons, rf=p["rf"],
                                          spec=spec_m, lookback_years=p["lookback"],
                                          reb_freq=p["oos_freq"], costs=p["costs"])
            except PortfolioLabError as e:
                errors[f"Reality check {label}"] = str(e)

    snapshot = {k: str(v) for k, v in p.items()
                if k not in ("costs", "run", "holdings")}
    snapshot["holdings"] = list(holdings)
    snapshot["costs"] = {"broker_bps": p["costs"].broker_bps,
                         "spread_bps": p["costs"].spread_bps,
                         "ter_bps": p["costs"].ter_bps}

    picks = [k for k in weights if k not in ("Equal weight (1/N)", "Benchmark 60/40")]
    return dict(prices=prices, returns=rets, assets=assets, types=types, classes=classes,
                sectors=sectors, countries=countries,
                mu=mu, cov=cov, diag=diag, spec=spec_m, cons=cons, weights=weights,
                stats=stats, errors=errors, W_boot=W_boot, frontier=(fv, fr),
                frontier_failed=n_failed, cloud=cloud, oos=oos, quality=quality,
                dropped=dropped, failed_downloads=failed_dl, fx_info=fx_info,
                params=p, bench_key=bench_key,
                bl_has_views=bl_has_views, holdings=holdings,
                recommended=recommended_portfolio(p["profile"], picks or list(weights)),
                audit=build_audit(snapshot, prices, weights, stats))


# ══ Interface ═══════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root{--bg:#05080D;--bg2:#0A1220;--bd:rgba(212,175,55,.18);--gold:#D4AF37;--tx:#EEF2F7;--mu:#64748B;}
*{font-family:'Inter',-apple-system,sans-serif!important;box-sizing:border-box;}
[data-testid="stAppViewContainer"]{background:var(--bg)!important;}
[data-testid="stHeader"]{background:rgba(5,8,13,.95)!important;border-bottom:1px solid var(--bd)!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#060B14,#0B1828)!important;border-right:1px solid var(--bd)!important;}
[data-testid="stSidebar"] *{color:var(--tx)!important;}
.stTabs [data-baseweb="tab-list"]{border-bottom:1px solid var(--bd)!important;gap:0!important;flex-wrap:wrap;}
.stTabs [data-baseweb="tab"]{color:var(--mu)!important;font-size:.84rem!important;padding:11px 16px!important;background:transparent!important;border-bottom:2px solid transparent!important;}
.stTabs [aria-selected="true"]{color:var(--gold)!important;border-bottom:2px solid var(--gold)!important;background:rgba(212,175,55,.05)!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:20px!important;}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#B8860B,#D4AF37,#B8860B)!important;color:#05080D!important;border:none!important;font-weight:700!important;letter-spacing:.12em!important;text-transform:uppercase!important;padding:.75rem 1rem!important;border-radius:8px!important;}
.stButton>button:not([kind="primary"]){background:rgba(13,18,28,.8)!important;color:var(--tx)!important;border:1px solid var(--bd)!important;border-radius:7px!important;font-size:.78rem!important;}
.stButton>button:not([kind="primary"]):hover{border-color:var(--gold)!important;color:var(--gold)!important;}
input,textarea{background:var(--bg2)!important;border:1px solid rgba(212,175,55,.2)!important;color:var(--tx)!important;}
.stSelectbox>div>div{background:var(--bg2)!important;border:1px solid rgba(212,175,55,.2)!important;color:var(--tx)!important;}
[data-testid="stDataFrame"] th{background:#0D1828!important;color:var(--mu)!important;font-size:.7rem!important;text-transform:uppercase!important;}
[data-testid="stDataFrame"] td{color:var(--tx)!important;font-size:.82rem!important;}
[data-testid="stMarkdownContainer"] p{color:var(--tx)!important;}
code{background:rgba(212,175,55,.1)!important;color:var(--gold)!important;border-radius:4px!important;padding:1px 5px!important;}
hr{border:none!important;height:1px!important;background:linear-gradient(90deg,transparent,var(--gold),transparent)!important;opacity:.25!important;}
.note{border-radius:10px;padding:12px 16px;font-size:.82rem;line-height:1.65;margin:6px 0 16px 0;}
.legal{position:sticky;top:0;z-index:99;background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.28);
border-radius:8px;padding:7px 14px;color:#FCA5A5;font-size:.72rem;margin-bottom:14px;}
::-webkit-scrollbar{width:5px;height:5px;}::-webkit-scrollbar-thumb{background:rgba(212,175,55,.22);border-radius:3px;}
</style>"""

TONES = {"info": ("rgba(59,130,246,.07)", "rgba(59,130,246,.28)", "#93C5FD"),
         "warn": ("rgba(245,158,11,.07)", "rgba(245,158,11,.3)", "#FCD34D"),
         "good": ("rgba(16,185,129,.07)", "rgba(16,185,129,.3)", "#6EE7B7"),
         "bad": ("rgba(239,68,68,.07)", "rgba(239,68,68,.3)", "#FCA5A5"),
         "gold": ("rgba(212,175,55,.06)", "rgba(212,175,55,.25)", "#E5C76B")}


def note(text: str, tone: str = "info"):
    bg, bd, fg = TONES[tone]
    st.markdown(f'<div class="note" style="background:{bg};border:1px solid {bd};'
                f'color:{fg};">{text}</div>', unsafe_allow_html=True)


def pct(x, d=2, signed=False):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x*100:{'+' if signed else ''}.{d}f}%"


def num(x, d=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def money(x, sym):
    return f"{sym}{x:,.0f}".replace(",", " ")


def kpi(label, value, sub="", color=GOLD):
    s = f'<div style="color:#64748B;font-size:.63rem;margin-top:4px;">{sub}</div>' if sub else ""
    return (f'<div style="background:linear-gradient(135deg,#0D1828,#121F33);'
            f'border:1px solid rgba(212,175,55,.18);border-radius:14px;padding:15px 17px;'
            f'height:100%;"><div style="color:#64748B;font-size:.63rem;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:.1em;">{label}</div>'
            f'<div style="color:{color};font-size:1.32rem;font-weight:700;margin-top:6px;'
            f'line-height:1.1;">{value}</div>{s}</div>')


def heading(text, sub=""):
    s = f'<div style="color:#64748B;font-size:.74rem;margin-top:3px;">{sub}</div>' if sub else ""
    st.markdown(f'<div style="display:flex;gap:12px;margin:22px 0 12px 0;">'
                f'<div style="width:3px;height:26px;border-radius:2px;background:'
                f'linear-gradient(180deg,#D4AF37,rgba(212,175,55,0));"></div><div>'
                f'<div style="font-size:.96rem;font-weight:600;color:#EEF2F7;">{text}</div>'
                f'{s}</div></div>', unsafe_allow_html=True)


def side_step(n, title, hint=""):
    st.sidebar.markdown(
        f'<div style="margin:18px 0 6px 0;padding-bottom:6px;border-bottom:1px solid '
        f'rgba(212,175,55,.12);"><span style="background:#D4AF37;color:#05080D;'
        f'border-radius:50%;padding:1px 7px;font-size:.68rem;font-weight:700;">{n}</span>'
        f'<span style="font-size:.72rem;font-weight:700;color:#94A3B8;letter-spacing:.1em;'
        f'text-transform:uppercase;margin-left:8px;">{title}</span>'
        + (f'<div style="color:#64748B;font-size:.68rem;margin-top:5px;">{hint}</div>'
           if hint else "") + '</div>', unsafe_allow_html=True)


DEFAULT_HOLDINGS = {
    "SPY": {"symbol": "SPY", "exchange": "", "description": "S&P 500 ETF",
            "type": "ETF", "currency": "USD", "source": "yahoo", "asset_class": "Stocks"},
    "AGG": {"symbol": "AGG", "exchange": "", "description": "US aggregate bonds",
            "type": "ETF", "currency": "USD", "source": "yahoo", "asset_class": "Bonds"},
    "GLD": {"symbol": "GLD", "exchange": "", "description": "Gold",
            "type": "ETF", "currency": "USD", "source": "yahoo",
            "asset_class": "Commodities"},
}


def holdings_picker():
    """Searchable dropdown (TradingView) with a plain-ticker fallback."""
    if "holdings" not in st.session_state:
        st.session_state.holdings = dict(DEFAULT_HOLDINGS)

    sources = ["Browse catalogue", "Search TradingView", "Type a ticker"]
    default = 0 if HAS_FD else (1 if HAS_TV else 2)
    src = st.sidebar.radio("How do you want to add holdings?", sources, index=default,
                           key="src",
                           help="The catalogue is a 300,000-instrument offline "
                                "database that also knows each company's sector and "
                                "country, which unlocks sector and country limits. "
                                "TradingView search is live. Ticker mode is for when "
                                "you already know the symbol.")
    if src == sources[0] and not HAS_FD:
        st.sidebar.warning("Install financedatabase to browse the catalogue.")
        src = sources[1] if HAS_TV else sources[2]
    if src == sources[1] and not HAS_TV:
        st.sidebar.warning("Install tvdatafeed to search TradingView.")
        src = sources[2]

    if src == sources[0]:
        try:
            with st.spinner("Loading catalogue…"):
                cat = load_catalogue(st.sidebar.checkbox(
                    "Major exchanges only", True, key="major",
                    help="Uncheck to include smaller venues. Slower, and prices are "
                         "often patchy there."))
        except DataError as e:
            st.sidebar.error(str(e)); cat = None
        if cat is not None:
            st.sidebar.caption(f"{len(cat):,} instruments available".replace(",", " "))
            kinds = st.sidebar.multiselect("Limit to", sorted(cat["type"].unique()),
                                           default=["Stock", "ETF"], key="kinds")
            q = st.sidebar.text_input("Search by name or symbol", key="cq",
                                      placeholder="apple, world, gold…")
            hits = search_catalogue(cat, q, kinds or None)
            if hits:
                pick = st.sidebar.selectbox("Matches", [h["label"] for h in hits],
                                            key="cpick", label_visibility="collapsed")
                row = next(h for h in hits if h["label"] == pick)
                st.sidebar.caption(
                    f"{row['type']} · {row['currency']} · "
                    f"{row.get('country') or 'country unknown'}"
                    + (f" · {row['sector']}" if row.get("sector") else "")
                    + (f" · ISIN {row['isin']}" if row.get("isin") else ""))
                if st.sidebar.button("＋ Add to portfolio", use_container_width=True,
                                     key="add_cat"):
                    if row["symbol"] in st.session_state.holdings:
                        st.sidebar.warning("Already in your portfolio.")
                    else:
                        st.session_state.holdings[row["symbol"]] = catalogue_holding(row)
                        st.rerun()
            elif q:
                st.sidebar.caption("No match.")

    elif src == sources[1]:
        q = st.sidebar.text_input("Search a company, ETF or crypto", key="q",
                                  placeholder="apple, world etf, bitcoin…",
                                  help="Type at least two letters, then pick a listing "
                                       "from the dropdown below.")
        if q and len(q) >= 2:
            try:
                results = tv_search(q)
            except DataError as e:
                st.sidebar.error(str(e)); results = []
            if results:
                pick = st.sidebar.selectbox("Matches", [r["label"] for r in results],
                                            key="pick", label_visibility="collapsed")
                if st.sidebar.button("＋ Add to portfolio", use_container_width=True,
                                     key="add_tv"):
                    r = next(x for x in results if x["label"] == pick)
                    if r["key"] in st.session_state.holdings:
                        st.sidebar.warning("Already in your portfolio.")
                    else:
                        st.session_state.holdings[r["key"]] = {
                            **r, "sector": "Unclassified", "country": "Unknown",
                            "asset_class": classify_asset(r["symbol"],
                                                          r["type"].upper())}
                        st.rerun()
            else:
                st.sidebar.caption("No match yet.")
    else:
        c1, c2 = st.sidebar.columns([3, 1])
        t = c1.text_input("Ticker", key="ticker", placeholder="AAPL, VWCE.DE…",
                          label_visibility="collapsed")
        if c2.button("＋", key="add_yf"):
            for raw in t.replace(";", ",").split(","):
                k = raw.strip().upper()
                if k and k not in st.session_state.holdings:
                    row = None
                    if HAS_FD:
                        try:
                            hits = search_catalogue(load_catalogue(), k)
                            row = next((h for h in hits
                                        if h["symbol"] == k), None)
                        except DataError:
                            row = None
                    if row:
                        st.session_state.holdings[k] = catalogue_holding(row)
                    else:
                        typ, cls = asset_profile(k) if HAS_YF else ("Stock", "Stocks")
                        st.session_state.holdings[k] = {
                            "symbol": k, "exchange": "", "description": k, "type": typ,
                            "currency": "USD", "source": "yahoo", "asset_class": cls,
                            "sector": "Unclassified", "country": "Unknown"}
            st.rerun()

    st.sidebar.caption(f"{len(st.session_state.holdings)} holding(s)")
    for k, h in list(st.session_state.holdings.items()):
        c1, c2 = st.sidebar.columns([6, 1])
        extra = " · ".join(x for x in (h.get("country"), h.get("sector"))
                           if x and x not in ("Unknown", "Unclassified"))
        c1.markdown(f'<div style="font-size:.78rem;"><code>{h["symbol"]}</code> '
                    f'<span style="color:#64748B;">{h["type"]} · {h["currency"]}</span>'
                    f'<br><span style="color:#475569;font-size:.68rem;">'
                    f'{h.get("description","")[:38]}{" · " + extra if extra else ""}'
                    f'</span></div>', unsafe_allow_html=True)
        if c2.button("✕", key=f"rm_{k}"):
            if len(st.session_state.holdings) > 2:
                del st.session_state.holdings[k]
                st.rerun()
            else:
                st.sidebar.warning("Keep at least two holdings.")
    return st.session_state.holdings


def group_limits_ui(title, mapping: dict, prefix: str) -> dict:
    """Min/max sliders, shown only for the groups actually present."""
    present = sorted({v for v in mapping.values() if v})
    out = {}
    # A single group covering everything can only be capped at 100%, so hide it.
    if len(present) < 2:
        return out
    st.markdown(f"**{title}**")
    for g in present:
        members = [k for k, v in mapping.items() if v == g]
        c1, c2 = st.columns(2)
        lo = c1.slider(f"{g} min %", 0, 100, 0, 5, key=f"{prefix}_{g}_lo",
                       help=f"At least this share of the portfolio must sit in "
                            f"{g.lower()} ({len(members)} holding(s)).")
        hi = c2.slider(f"{g} max %", 0, 100, 100, 5, key=f"{prefix}_{g}_hi",
                       help=f"At most this share of the portfolio may sit in {g.lower()}.")
        if lo > 0 or hi < 100:
            out[g] = (lo / 100, hi / 100)
    return out


def sidebar() -> dict:
    st.sidebar.markdown(
        '<div style="padding:14px 0 10px;border-bottom:1px solid rgba(212,175,55,.14);">'
        '<div style="font-family:Playfair Display,serif;font-size:1.15rem;font-weight:700;'
        'color:#D4AF37;">📈 Portfolio Lab</div><div style="color:#64748B;font-size:.65rem;'
        'letter-spacing:.14em;text-transform:uppercase;margin-top:4px;">'
        'Build · test honestly · project</div></div>', unsafe_allow_html=True)

    side_step(1, "Your money", "How much you have and what you are aiming for.")
    base = st.sidebar.selectbox("Currency", list(CURRENCIES), index=0, key="ccy",
                                format_func=lambda c: f"{CURRENCIES[c]['flag']}  {c}",
                                help="Everything is converted into this currency "
                                     "before any calculation, so the numbers are "
                                     "what you would actually see in your account.")
    sym = CURRENCIES[base]["symbol"]
    initial = st.sidebar.number_input(f"Starting amount ({sym})", 100, 100_000_000,
                                      10_000, 500, key="init",
                                      help="What you can invest today.")
    monthly = st.sidebar.number_input(f"Added every month ({sym})", 0, 1_000_000, 200, 50,
                                      key="mth", help="Leave at 0 for a one-off "
                                                      "investment. Regular investing "
                                                      "changes the risk profile a lot.")
    horizon = st.sidebar.slider("Years until you need the money", 1, 40, 10, key="hz",
                                help="Drives the projection and how much short-term "
                                     "risk makes sense.")
    goal = st.sidebar.number_input(f"Target amount ({sym}, 0 = none)", 0, 1_000_000_000,
                                   0, 1000, key="goal",
                                   help="We will tell you the odds of reaching it.")

    side_step(2, "Your comfort with risk", "This sets automatic guardrails.")
    profile = st.sidebar.radio("Risk profile", list(RISK_PROFILES), index=1, key="prof",
                               help="Caps how much can go into a single holding and "
                                    "into risky assets overall.")
    st.sidebar.caption(RISK_PROFILES[profile][2])

    side_step(3, "Your holdings", "What the portfolio is made of.")
    holdings = holdings_picker()

    rf_auto, rf_src = fetch_risk_free(base) if HAS_YF else (CURRENCIES[base]["rf"], "default")
    adv = {}
    with st.sidebar.expander("⚙️ Advanced settings", expanded=False):
        st.caption("Sensible defaults are already applied. Only change these if you "
                   "know what you are adjusting.")
        c1, c2 = st.columns(2)
        adv["start"] = c1.date_input("History from", date(2015, 1, 1), key="d0",
                                     help="Longer history means better estimates, but "
                                          "older data may describe a different market.")
        adv["end"] = c2.date_input("History to", date.today(), key="d1")
        adv["rf"] = st.slider("Risk-free rate (%)", 0.0, 10.0,
                              float(round(rf_auto * 100, 2)), .05, key="rf",
                              help=f"What cash earns. Source: {rf_src}. Used as the "
                                   f"benchmark every risky asset must beat.") / 100

        st.markdown("---")
        mu_label = st.selectbox("Expected returns from",
                                ["Market equilibrium (Black-Litterman)",
                                 "Past returns (noisy)", "Don't estimate returns"],
                                index=0, key="mu",
                                help="Past returns are mostly noise over short "
                                     "histories. Market equilibrium starts from what "
                                     "the market itself implies. The third option "
                                     "ignores returns and optimises risk only.")
        adv["mu_method"] = {"Past returns (noisy)": "historical",
                            "Market equilibrium (Black-Litterman)": "black_litterman",
                            "Don't estimate returns": "none"}[mu_label]
        adv["cov_method"] = "ledoit_wolf" if st.selectbox(
            "Risk estimate", ["Stabilised (recommended)", "Raw historical"], index=0,
            key="cov", help="Stabilised shrinks noisy correlations toward a simpler "
                            "structure, which holds up much better out of sample."
        ).startswith("Stabilised") else "sample"
        adv["est_freq"] = {"Automatic": "auto", "Daily": "D", "Weekly": "W",
                           "Monthly": "M"}[st.selectbox(
            "Measure returns", ["Automatic", "Daily", "Weekly", "Monthly"], index=0,
            key="freq", help="Daily returns understate how much markets in different "
                             "time zones move together. Automatic switches to weekly "
                             "when your holdings span several exchanges.")]

        adv["bl_views"], adv["bl_conf"] = {}, {}
        if adv["mu_method"] == "black_litterman":
            st.caption("Optional: tell the model where you disagree with the market. "
                       "With no views, the result simply mirrors the market mix.")
            for k, h in holdings.items():
                v = st.number_input(f"{h['symbol']} — your expected annual return (%)",
                                    -50.0, 100.0, 0.0, .5, key=f"v_{k}")
                c = st.slider(f"{h['symbol']} — how sure are you?", 0.0, 1.0, 0.0, .05,
                              key=f"c_{k}", help="0 means no view at all.")
                adv["bl_views"][k], adv["bl_conf"][k] = v / 100, c

        st.markdown("---")
        adv["broker"] = st.number_input("Broker fee (bps per trade)", 0.0, 200.0,
                                        DEFAULT_BROKER_BPS, 1.0, key="brk",
                                        help="10 bps = 0.10% of the amount traded.")
        adv["spread"] = st.number_input("Half spread (bps)", 0.0, 200.0,
                                        DEFAULT_SPREAD_BPS, 1.0, key="sp",
                                        help="Hidden cost between buy and sell price.")
        adv["ter"] = st.number_input("Fund ongoing charges (bps per year)", 0.0, 300.0,
                                     DEFAULT_TER_BPS, 1.0, key="ter",
                                     help="Typical ETF charges 20 bps = 0.20% a year.")
        adv["rebalance"] = REBALANCE_CHOICES[st.selectbox(
            "Bring back to target", list(REBALANCE_CHOICES), index=4, key="reb",
            help="Rebalancing keeps your risk stable but costs fees. Threshold "
                 "rebalancing usually costs the least.")]

        st.markdown("---")
        adv["use_asset_limits"] = st.checkbox("Limit each holding", key="ual",
                                              help="Set a floor and a cap per line.")
        adv["per_asset"] = {}
        if adv["use_asset_limits"]:
            for k, h in holdings.items():
                c1, c2 = st.columns(2)
                lo = c1.slider(f"{h['symbol']} min %", 0, 100, 0, 1, key=f"mn_{k}")
                hi = c2.slider(f"{h['symbol']} max %", 0, 100, 100, 1, key=f"mx_{k}")
                adv["per_asset"][k] = (lo / 100, hi / 100)
        adv["type_limits"] = group_limits_ui(
            "Limits by instrument type", {k: h["type"] for k, h in holdings.items()},
            "tl")
        adv["class_limits"] = group_limits_ui(
            "Limits by asset class",
            {k: h.get("asset_class", "Stocks") for k, h in holdings.items()}, "cl")
        adv["sector_limits"] = group_limits_ui(
            "Limits by sector",
            {k: h.get("sector", "Unclassified") for k, h in holdings.items()}, "sl")
        adv["country_limits"] = group_limits_ui(
            "Limits by country",
            {k: h.get("country", "Unknown") for k, h in holdings.items()}, "col")
        adv["max_assets"] = st.number_input("Max number of holdings (0 = no limit)",
                                            0, 50, 0, 1, key="maxn",
                                            help="Fewer lines is easier to manage.") or None
        adv["apply_profile"] = st.checkbox("Apply risk-profile guardrails", True,
                                           key="gp")

        st.markdown("---")
        adv["do_oos"] = st.checkbox("Run the reality check", True, key="oos",
                                    help="Re-optimises through history using only past "
                                         "data. Slower, but it is the only honest test.")
        adv["lookback"] = st.slider("Training window (years)", 1.0, 8.0, 3.0, .5,
                                    key="lb", disabled=not adv["do_oos"])
        adv["oos_freq"] = {"Monthly": "M", "Quarterly": "Q", "Yearly": "A"}[
            st.selectbox("Re-optimise every", ["Monthly", "Quarterly", "Yearly"],
                         index=1, key="of", disabled=not adv["do_oos"])]
        adv["do_resample"] = st.checkbox("Stabilise weights (resampling)", True, key="rs")
        adv["do_cvar"] = st.checkbox("Add a crash-focused portfolio (min CVaR)", False,
                                     key="cv", help="Minimises average loss on the "
                                                    "worst 5% of days.")
        adv["fractional"] = st.checkbox("My broker allows fractional shares", False,
                                        key="frac")
        adv["lot_size"] = st.number_input("Lot size", 1, 1000, 1, key="lot",
                                          help="Some listings trade in blocks.")
        adv["tax"] = st.selectbox("Tax on gains", list(TAX_REGIMES), index=0, key="tax",
                                  help=" ".join(TAX_REGIMES[k][1] for k in list(TAX_REGIMES)[:1]))
        st.caption(TAX_REGIMES[adv["tax"]][1])
        adv["inflation"] = st.slider("Expected inflation (%)", 0.0, 8.0, 2.0, .1,
                                     key="inf", help="Used to show tomorrow's money in "
                                                     "today's purchasing power.") / 100
        adv["allow_static_fx"] = st.checkbox("Allow approximate exchange rates", False,
                                             key="sfx",
                                             help="Off by default: we would rather stop "
                                                  "than show you wrong amounts.")

    st.sidebar.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    ack = st.sidebar.checkbox("I understand this is a simulation, not advice.",
                              key="ack")
    run = st.sidebar.button("▶  BUILD MY PORTFOLIO", type="primary",
                            use_container_width=True, disabled=not ack)

    return dict(base_ccy=base, sym=sym, initial=float(initial), monthly=float(monthly),
                horizon=horizon, goal=float(goal), profile=profile, holdings=holdings,
                costs=CostModel(adv["broker"], adv["spread"], adv["ter"]),
                run=run, **{k: v for k, v in adv.items()
                            if k not in ("broker", "spread", "ter")})


# ══ Tabs ════════════════════════════════════════════════════════════════════

def tab_portfolio(res):
    p, rec = res["params"], res["recommended"]
    w, s = res["weights"][rec], res["stats"][rec]
    names = {a: res["holdings"][a]["symbol"] for a in res["assets"]}
    note(f"<b>Your suggested mix: {rec}.</b> It was chosen to match your "
         f"<b>{p['profile'].lower()}</b> risk profile out of every mix we tested. "
         f"The percentages below are how much of your money goes into each holding. "
         f"Every number on this page is <b>after</b> trading fees and fund charges, "
         f"and measured on history from {res['prices'].index.min().date()} to "
         f"{res['prices'].index.max().date()}.", "gold")

    c1, c2 = st.columns([2, 3])
    with c1:
        st.plotly_chart(chart_pie(w, [names[a] for a in res["assets"]], rec),
                        use_container_width=True, key="pie_main")
    with c2:
        cards = [("Growth per year", pct(s["cagr"], 1, True),
                  "what this mix returned annually"),
                 ("Typical swing", pct(s["vol"], 1), "how much it bounces in a year"),
                 ("Worst fall", pct(s["max_drawdown"], 1),
                  f"took {s['underwater_days']} days to recover", RED),
                 ("Return per unit of risk", num(s["sharpe"], 2),
                  "above 1.0 is good"),
                 ("Bad-day loss", pct(s["cvar95"], 2),
                  "average of the worst 5% of days", RED),
                 ("Yearly trading", f"{s['annual_turnover']*100:.0f}%",
                  "how much gets bought and sold")]
        for row in (cards[:3], cards[3:]):
            cols = st.columns(3)
            for col, c in zip(cols, row):
                col.markdown(kpi(c[0], c[1], c[2], c[3] if len(c) > 3 else GOLD),
                             unsafe_allow_html=True)

    heading("What is inside", "Grouped the way your broker would show it")
    df = pd.DataFrame({"Holding": [names[a] for a in res["assets"]],
                       "What it is": [res["holdings"][a].get("description", "")[:40]
                                      for a in res["assets"]],
                       "Type": [res["types"][a] for a in res["assets"]],
                       "Asset class": [res["classes"][a] for a in res["assets"]],
                       "Sector": [res["sectors"][a] for a in res["assets"]],
                       "Country": [res["countries"][a] for a in res["assets"]],
                       "Weight": [f"{x*100:.1f}%" for x in w],
                       f"Amount ({p['sym']})": [money(p["initial"] * x, "") for x in w]})
    st.dataframe(df, use_container_width=True, hide_index=True)
    for label, mapping in (("asset class", res["classes"]), ("sector", res["sectors"]),
                           ("country", res["countries"])):
        g = pd.DataFrame({"g": [mapping[a] for a in res["assets"]], "w": w})
        g = g[~g["g"].isin(["Unclassified", "Unknown", ""])].groupby("g")["w"].sum()
        if len(g) > 1:
            st.caption(f"By {label}: " + " · ".join(
                f"{k} {v*100:.0f}%" for k, v in g.sort_values(ascending=False).items()))

    with st.expander("See the other mixes we tested, and why they differ"):
        note("<b>Min Variance</b> takes the calmest possible route. "
             "<b>Risk Parity</b> gives every holding an equal share of the risk rather "
             "than of the money. <b>Max Sharpe</b> chases the best return per unit of "
             "risk, but leans hardest on return estimates, which are unreliable. "
             "<b>Resampled</b> is Max Sharpe averaged over many re-runs on shuffled "
             "history, which makes it far more stable. <b>Equal weight</b> simply "
             "splits evenly and is a surprisingly tough benchmark to beat.")
        st.dataframe(pd.DataFrame([
            {"Mix": k, "Growth/yr": pct(v["cagr"], 1, True), "Swing": pct(v["vol"], 1),
             "Return per risk": num(v["sharpe"], 2), "Worst fall": pct(v["max_drawdown"], 1),
             "Bad-day loss": pct(v["cvar95"], 2),
             "Trading/yr": f"{v['annual_turnover']*100:.0f}%"}
            for k, v in res["stats"].items()]), use_container_width=True, hide_index=True)
        cols = st.columns(min(3, len(res["weights"])))
        for i, (k, ww) in enumerate(res["weights"].items()):
            cols[i % len(cols)].plotly_chart(
                chart_pie(ww, [names[a] for a in res["assets"]], k),
                use_container_width=True, key=f"pie_{i}_{k}")
    if res["errors"]:
        with st.expander(f"⚠️ {len(res['errors'])} calculation(s) could not complete"):
            for k, v in res["errors"].items():
                st.markdown(f"**{k}** — {v}")


def tab_plan(res):
    p, sym = res["params"], res["params"]["sym"]
    note("This is the part that actually matters for your life: what your money could "
         "be worth, given what you put in and how long you leave it. We replay "
         "thousands of possible futures by reshuffling real market history in "
         "six-month blocks, so crashes and recoveries stay realistic. "
         "<b>The spread of outcomes is the message</b> — a single number would be "
         "false precision.", "gold")

    c1, c2, c3 = st.columns(3)
    pick = c1.selectbox("Mix to project", list(res["weights"]),
                        index=list(res["weights"]).index(res["recommended"]), key="pp")
    sims = c2.select_slider("Futures to simulate", [1000, 2500, 5000, 10000], 5000,
                            key="ps")
    block = c3.slider("Block length (months)", 1, 24, 6, key="pb",
                      help="Longer blocks keep market cycles intact.")
    tax = TAX_REGIMES[p["tax"]][0]
    try:
        sim = simulate_wealth(to_monthly(res["stats"][pick]["net_returns"]),
                              initial=p["initial"], monthly_contribution=p["monthly"],
                              horizon_years=p["horizon"], n_sims=int(sims),
                              block=int(block), inflation=p["inflation"], tax_rate=tax)
    except (DataError, ValueError) as e:
        st.error(str(e)); return

    q = sim["quantiles"]
    cards = [("You put in", money(sim["invested"], sym),
              f"{money(p['initial'], sym)} + {money(p['monthly'], sym)}/month"),
             ("Most likely", money(q[.5], sym), f"after {p['horizon']} years"),
             ("If things go badly", money(q[.05], sym), "1 in 20 outcomes are worse", RED),
             ("If things go well", money(q[.95], sym), "1 in 20 are better", GREEN),
             ("Chance of losing money", f"{sim['prob_loss']*100:.0f}%",
              "ending below what you put in",
              RED if sim["prob_loss"] > .15 else ORANGE),
             ("Your actual return", pct(sim["median_irr"], 1, True),
              "per year, counting each deposit")]
    for row in (cards[:3], cards[3:]):
        cols = st.columns(3)
        for col, c in zip(cols, row):
            col.markdown(kpi(c[0], c[1], c[2], c[3] if len(c) > 3 else GOLD),
                         unsafe_allow_html=True)

    st.plotly_chart(chart_fan(sim["trajectories"],
                              p["initial"] + p["monthly"] * np.arange(sim["horizon_months"] + 1),
                              sym), use_container_width=True, key="fan")
    st.caption("The gold line is the middle outcome. The shaded bands hold half and "
               "then nine tenths of all simulated futures. The dashed line is simply "
               "the cash you paid in.")

    if p["goal"] > 0:
        pr = prob_reach_goal(sim["terminal"], p["goal"])
        tone = "good" if pr > .7 else "warn" if pr > .4 else "bad"
        extra = ("" if pr > .7 else
                 " To improve the odds, the three levers are: invest more each month, "
                 "give it more time, or accept more risk — in that order of reliability.")
        note(f"<b>Odds of reaching {money(p['goal'], sym)} in {p['horizon']} years: "
             f"{pr*100:.0f}%.</b>{extra}", tone)

    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(chart_hist(sim["terminal"], sim["invested"],
                               p["goal"] or None, sym), use_container_width=True,
                    key="terminal_hist")
    with c2:
        heading("In today's money", "After tax and inflation")
        st.dataframe(pd.DataFrame({
            "Outcome": ["Bad (5%)", "Poor (25%)", "Middle", "Good (75%)", "Great (95%)"],
            "Before tax": [money(q[x], sym) for x in (.05, .25, .5, .75, .95)],
            "After tax & inflation": [money(sim["quantiles_real"][x], sym)
                                      for x in (.05, .25, .5, .75, .95)]}),
            use_container_width=True, hide_index=True)
        st.caption(f"Tax: {p['tax']} ({tax*100:.1f}%). Inflation: "
                   f"{p['inflation']*100:.1f}%/yr. Indicative only — confirm with a "
                   f"tax adviser.")

    heading("The bumpy part", "What you would have to sit through")
    dd = sim["max_drawdown"]
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Typical worst drop", pct(float(np.median(dd)), 1),
                    "at some point along the way", RED), unsafe_allow_html=True)
    c2.markdown(kpi("Rough ride", pct(float(np.quantile(dd, .05)), 1),
                    "1 in 20 futures are worse", RED), unsafe_allow_html=True)
    c3.markdown(kpi("Futures losing over 30%", f"{float((dd < -.30).mean())*100:.0f}%",
                    "at some point", ORANGE), unsafe_allow_html=True)
    note("Plans rarely fail because the maths was wrong. They fail because someone "
         "sells at the bottom. If the numbers above would make you abandon the plan, "
         "choose a calmer mix now rather than later.", "warn")


def tab_reality(res):
    p = res["params"]
    note("Anything can look brilliant when it is tuned on the same data it is judged "
         "on. Here we do the opposite: at each point in history the mix is rebuilt "
         "using <b>only what was knowable at the time</b>, then scored on the months "
         "that followed. We also compare it to simply splitting your money evenly, "
         "which is a famously hard benchmark to beat.", "info")

    if not res["oos"]:
        st.info("Turn on 'Run the reality check' in Advanced settings, or extend your "
                "history — this test needs a training window plus a test period.")
    else:
        st.plotly_chart(chart_lines({k: v["nav"] for k, v in res["oos"].items()},
                                    "Growth of 1 unit, using no future information",
                                    "Value"), use_container_width=True, key="oos_nav")
        naive = res["oos"].get("Equal weight (1/N)")
        rows = []
        for k, bt in res["oos"].items():
            stt = full_stats(bt["nav"], bt["returns"], p["rf"])
            row = {"Strategy": k, "Growth/yr": pct(stt["cagr"], 1, True),
                   "Swing": pct(stt["vol"], 1), "Return per risk": num(stt["sharpe"], 2),
                   "Worst fall": pct(stt["max_drawdown"], 1),
                   "Trading/yr": f"{bt['annual_turnover']*100:.0f}%",
                   "Rebuilds": bt["n_rebalances"]}
            if naive is not None and k != "Equal weight (1/N)":
                v = oos_verdict(bt, naive, p["rf"])
                row["Vs equal weight"] = f"{v['sharpe_gap']:+.2f}"
                row["Could be luck (p)"] = num(v["p_value"], 3)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if naive is not None and res["recommended"] in res["oos"]:
            v = oos_verdict(res["oos"][res["recommended"]], naive, p["rf"])
            tone = "good" if v["beats_naive"] and v["significant"] else \
                "warn" if v["beats_naive"] else "bad"
            verdict = ("beat an equal split, and the gap is unlikely to be luck"
                       if v["beats_naive"] and v["significant"] else
                       "edged ahead of an equal split, but the gap could easily be luck"
                       if v["beats_naive"] else "did not beat a simple equal split")
            note(f"<b>Verdict: {res['recommended']} {verdict}.</b> Return per unit of "
                 f"risk was {v['strategy']['sharpe']:.2f} against "
                 f"{v['naive']['sharpe']:.2f}. "
                 + ("Optimising is earning its keep here."
                    if v["beats_naive"] and v["significant"] else
                    "On this set of holdings, spreading your money evenly would have "
                    "worked about as well — and cost less to run."), tone)

    heading("How much do we really know?", "The honest limits of the estimates")
    d = res["diag"]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("History used", f"{d['t_obs']:,}".replace(",", " "),
                    f"{'weekly' if d['est_freq']=='W' else 'monthly' if d['est_freq']=='M' else 'daily'} observations"),
                unsafe_allow_html=True)
    c2.markdown(kpi("Noise smoothing", f"{d['shrinkage_delta']*100:.0f}%",
                    "how much we distrust raw correlations"), unsafe_allow_html=True)
    nr = float(np.median(d["noise_ratio"]))
    c3.markdown(kpi("Noise vs signal", f"{nr:.1f}×", "above 1 means mostly noise",
                    RED if nr > 1 else ORANGE), unsafe_allow_html=True)
    c4.markdown(kpi("Real diversification",
                    f"{effective_n(res['weights'][res['recommended']]):.1f}",
                    f"of {len(res['assets'])} holdings"), unsafe_allow_html=True)

    st.plotly_chart(chart_uncertainty(
        [res["holdings"][a]["symbol"] for a in res["assets"]],
        d["mu_hist"], d["se_mu"]), use_container_width=True, key="mu_ci")
    note("Each dot is an asset's average past return; the bar is the range we cannot "
         "rule out. <b>When a bar crosses zero, history cannot even confirm the asset "
         "makes money.</b> This is normal, and it is why we lean on risk-based mixes "
         "rather than chasing the highest past return.", "warn")

    if res["W_boot"] is not None:
        st.plotly_chart(chart_box([res["holdings"][a]["symbol"] for a in res["assets"]],
                                  res["W_boot"]), use_container_width=True,
                        key="weight_box")
        sp = (res["W_boot"].max(0) - res["W_boot"].min(0)).max() * 100
        st.caption(f"Re-running the optimiser on reshuffled history moves some weights "
                   f"by up to {sp:.0f} percentage points. Tall boxes mean the 'optimal' "
                   f"answer is not really identifiable from this much data.")

    if p["mu_method"] == "black_litterman" and not res["bl_has_views"]:
        note("You are using market equilibrium with no views of your own, so the "
             "Max Sharpe mix simply mirrors the market. That is the model working as "
             "designed — add views in Advanced settings to move away from it.")
    st.caption("The efficient frontier, factor exposures and the full metric set "
               "are in the Analytics tab.")


def tab_risk(res):
    p = res["params"]
    names = [res["holdings"][a]["symbol"] for a in res["assets"]]
    note("Volatility alone hides what hurts. These views focus on the losses: how "
         "deep, how long, how correlated your holdings really are, and how the mix "
         "behaved in real crises.", "info")

    heading("Falls from the previous peak")
    st.plotly_chart(chart_drawdown({k: v["nav"] for k, v in res["stats"].items()}),
                    use_container_width=True, key="dd")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_corr(res["returns"].rename(
            columns={a: n for a, n in zip(res["assets"], names)})),
            use_container_width=True, key="corr")
        st.caption("1.00 means they move in lockstep, so they diversify nothing. "
                   "Values near 0 or below are what real diversification looks like.")
    with c2:
        rc = risk_contributions(res["weights"][res["recommended"]], res["cov"])
        st.plotly_chart(chart_bars(names, list(rc),
                                   "Where the risk actually comes from",
                                   "Share of total risk (%)", fmt="{:.0f}%",
                                   colors=[PALETTE[i % len(PALETTE)] for i in range(len(names))]),
                        use_container_width=True, key="rc_main")
        st.caption("A holding can be 10% of your money and 40% of your risk. That gap "
                   "is the single most common surprise in a portfolio.")

    heading("Real crises", "Your current mix, dropped into past market shocks")
    note("Honest caveat: these weights were chosen using recent data, so nobody could "
         "have held this exact mix back then. Read it as a sensitivity check, not a "
         "prediction.", "warn")
    picked = st.multiselect("Crises to test", list(SCENARIOS),
                            default=list(SCENARIOS)[:2], key="sc")
    core = {k: v for k, v in res["weights"].items()
            if k in (res["recommended"], "Equal weight (1/N)")}
    for name in picked:
        start, end, color, desc = SCENARIOS[name]
        spec = tuple((k, h["source"], h["symbol"], h.get("exchange", ""))
                     for k, h in res["holdings"].items() if k in res["assets"])
        try:
            with st.spinner(f"Loading {name}…"):
                raw = fetch_universe(spec, start, end)
                ccy = {k: res["holdings"][k]["currency"] for k in raw.columns}
                fx = None
                if any(c != p["base_ccy"] for c in ccy.values()):
                    try:
                        fx = fetch_fx(tuple(sorted(set(ccy.values()))), p["base_ccy"],
                                      start, end)
                    except FXError:
                        fx = None
                pr, _ = convert_to_base(raw, ccy, p["base_ccy"], fx, True)
        except DataError as e:
            st.warning(f"**{name}** — {e}")
            continue
        sc = run_scenario(pr, core, res["assets"], name, (start, end),
                          rebalance=p["rebalance"], costs=p["costs"])
        if not sc or not sc.get("portfolios"):
            st.warning(f"**{name}** — none of your holdings existed with usable data "
                       f"between {start} and {end}.")
            continue
        if sc["missing"]:
            miss = ", ".join(res["holdings"][m]["symbol"] for m in sc["missing"])
            st.info(f"**{name}**: {miss} did not exist yet, so the result covers only "
                    f"the rest of the portfolio, rescaled to 100%.")
        st.markdown(f'<div style="border-left:4px solid {color};padding:10px 16px;'
                    f'background:#0D1828;border-radius:8px;margin:10px 0;">'
                    f'<b style="color:#EEF2F7;">{name}</b>'
                    f'<div style="color:#64748B;font-size:.74rem;">{start} → {end} · '
                    f'{desc}</div></div>', unsafe_allow_html=True)
        cols = st.columns(len(sc["portfolios"]))
        for col, (pn, stt) in zip(cols, sc["portfolios"].items()):
            col.markdown(kpi(pn, pct(stt["total_return"], 1, True),
                             f"worst fall {pct(stt['max_drawdown'],1)}",
                             GREEN if stt["total_return"] >= 0 else RED),
                         unsafe_allow_html=True)
        st.plotly_chart(chart_lines({k: v["cum_series"] / 100
                                     for k, v in sc["portfolios"].items()},
                                    f"{name} — value through the crisis", "Base 100",
                                    height=340, base100=True),
                        use_container_width=True, key=f"sc_{name}")


def tab_analytics(res):
    """Full quantitative detail: frontier, decomposition, factors, every metric."""
    p = res["params"]
    names = [res["holdings"][a]["symbol"] for a in res["assets"]]
    note("Everything the guided tabs summarise, in full. Definitions and formulas "
         "for every figure are in the Method tab.", "info")

    heading("Efficient frontier",
            "Mean-variance opportunity set under your constraints")
    if p["mu_method"] == "none":
        st.info("The frontier is a return-versus-risk map, so it needs expected "
                "returns. Pick a return model in Advanced settings.")
    else:
        core = {k: v for k, v in res["weights"].items() if k != "Benchmark 60/40"}
        st.plotly_chart(chart_frontier(res["mu"], res["cov"], p["rf"], names, core,
                                       res["cloud"], res["frontier"]),
                        use_container_width=True, key="frontier")
        lo, hi = res["frontier"][1][:1], res["frontier"][1][-1:]
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi("Attainable return band",
                        f"{pct(float(lo[0]) if len(lo) else float('nan'), 1)} → "
                        f"{pct(float(hi[0]) if len(hi) else float('nan'), 1)}",
                        "under your limits, by LP"), unsafe_allow_html=True)
        c2.markdown(kpi("Frontier points", f"{len(res['frontier'][0])}",
                        f"{res['frontier_failed']} infeasible and dropped"),
                    unsafe_allow_html=True)
        c3.markdown(kpi("Random mixes plotted", f"{len(res['cloud'][0]):,}".replace(",", " "),
                        "Dirichlet draws inside the constraint set"),
                    unsafe_allow_html=True)
        rec_w = res["weights"][res["recommended"]]
        c4.markdown(kpi("Tangency Sharpe", num(sharpe(rec_w, res["mu"], res["cov"],
                                                      p["rf"]), 3),
                        f"{res['recommended']}, in sample"), unsafe_allow_html=True)
        st.caption("Grey dots are random admissible mixes. The gold curve is the "
                   "minimum variance attainable at each target return, solved by "
                   "SLSQP; the dashed line is the capital market line through the "
                   "risk-free rate. Frontier bounds come from a linear program, so "
                   "they respect every group limit rather than assuming the full "
                   "range of asset returns is reachable.")

    heading("Portfolio decomposition", "Weights, risk shares and concentration")
    rows = []
    for k, w in res["weights"].items():
        rc = risk_contributions(w, res["cov"])
        rows.append({"Portfolio": k, "Return (arith.)": pct(port_return(w, res["mu"]), 2),
                     "Volatility": pct(port_vol(w, res["cov"]), 2),
                     "Sharpe": num(sharpe(w, res["mu"], res["cov"], p["rf"]), 3),
                     "Diversification ratio": num(diversification_ratio(w, res["cov"]), 2),
                     "Effective N": num(effective_n(w), 2),
                     "Max weight": pct(float(np.max(w)), 1),
                     "Max risk share": f"{np.max(rc):.0f}%",
                     "Herfindahl": num(float((w ** 2).sum()), 3)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    sel = st.multiselect("Risk contributions for", list(res["weights"]),
                         default=[res["recommended"], "Equal weight (1/N)"], key="rcsel")
    if sel:
        fig = go.Figure()
        for k in sel:
            fig.add_trace(go.Bar(name=k, x=names,
                                 y=risk_contributions(res["weights"][k], res["cov"]),
                                 marker=dict(color=PORT_COLORS.get(k, GOLD), opacity=.85,
                                             line=dict(color=BG, width=1))))
        fig.update_layout(**layout("Marginal risk contribution by holding (%)",
                                   "", "Share of portfolio variance (%)",
                                   barmode="group", height=400))
        st.plotly_chart(fig, use_container_width=True, key="rc_multi")

    heading("Full metric set", "Net of fees, on the analysis window")
    st.dataframe(pd.DataFrame([
        {"Portfolio": k, "CAGR": pct(v["cagr"], 2, True),
         "Arith. return": pct(v["net_returns"].mean() * PERIODS, 2, True),
         "Volatility": pct(v["vol"], 2), "Sharpe": num(v["sharpe"], 3),
         "Sortino": num(v["sortino"], 3), "Calmar": num(v["calmar"], 3),
         "Max DD": pct(v["max_drawdown"], 2),
         "Underwater (days)": v["underwater_days"], "Ulcer": num(v["ulcer"], 4),
         "VaR 95%": pct(v["var95"], 2), "CVaR 95%": pct(v["cvar95"], 2),
         "VaR 99%": pct(v["var99"], 2), "CVaR 99%": pct(v["cvar99"], 2),
         "Skew": num(v["skew"], 2), "Excess kurtosis": num(v["kurtosis"], 2),
         "Beta": num(v.get("beta"), 2), "Alpha": pct(v.get("alpha"), 2, True),
         "Tracking error": pct(v.get("tracking_error"), 2),
         "Info ratio": num(v.get("information_ratio"), 2),
         "R²": num(v.get("r2"), 2), "Turnover/yr": f"{v['annual_turnover']*100:.0f}%",
         "Fees paid": pct(v["total_fees"], 2)}
        for k, v in res["stats"].items()]), use_container_width=True, hide_index=True)
    st.caption(f"Beta, alpha, tracking error and information ratio are measured "
               f"against {res['bench_key']}. Risk-free rate {p['rf']*100:.2f}%.")

    heading("Stability of the estimates")
    d = res["diag"]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Estimation frequency",
                    {"D": "Daily", "W": "Weekly", "M": "Monthly"}[d["est_freq"]],
                    f"{d['periods_per_year']} periods/yr, {d['t_obs']} obs"),
                unsafe_allow_html=True)
    c2.markdown(kpi("Ledoit-Wolf δ", f"{d['shrinkage_delta']*100:.1f}%",
                    "weight on the constant-correlation target"), unsafe_allow_html=True)
    c3.markdown(kpi("Condition number", num(float(np.linalg.cond(res["cov"])), 1),
                    "of the covariance matrix"), unsafe_allow_html=True)
    c4.markdown(kpi("Median noise ratio", f"{float(np.median(d['noise_ratio'])):.2f}×",
                    "SE(μ) ÷ |μ|"), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Holding": names,
        "μ used": [pct(x, 2, True) for x in res["mu"]],
        "μ historical": [pct(x, 2, True) for x in d["mu_hist"]],
        "SE(μ)": [pct(x, 2) for x in d["se_mu"]],
        "95% CI": [f"{(m-1.96*e)*100:+.1f}% … {(m+1.96*e)*100:+.1f}%"
                   for m, e in zip(d["mu_hist"], d["se_mu"])],
        "σ": [pct(x, 2) for x in np.sqrt(np.diag(res["cov"]))],
        "Asset Sharpe": [num((res["mu"][i] - p["rf"]) / np.sqrt(res["cov"][i, i]), 2)
                         for i in range(len(names))]}),
        use_container_width=True, hide_index=True)

    heading("Rolling risk-adjusted return")
    win = st.slider("Window (trading days)", 60, 756, 252, 21, key="rollwin")
    st.plotly_chart(chart_rolling({k: v["net_returns"] for k, v in res["stats"].items()},
                                  p["rf"], win), use_container_width=True, key="rolling")
    st.caption("A Sharpe that swings widely across windows is a sign the in-sample "
               "figure is period-specific rather than a stable property.")

    heading("Factor exposure", "Is this a real allocation or a disguised index bet?")
    if not HAS_YF:
        st.info("yfinance is required to download the factor proxies.")
    elif st.button("Run factor regression", key="runfac"):
        try:
            with st.spinner("Downloading factor proxies…"):
                fp = fetch_prices(tuple(FACTOR_PROXIES.values()), str(p["start"]),
                                  str(p["end"]))
                fp = fp.rename(columns={v: k for k, v in FACTOR_PROXIES.items()})
                fr = simple_returns(fp)
            fac = factor_regression(res["stats"][res["recommended"]]["net_returns"],
                                    fr, p["rf"])
            st.plotly_chart(chart_factors(fac), use_container_width=True, key="factors")
            st.dataframe(fac, use_container_width=True, hide_index=True)
            st.caption("OLS of portfolio excess returns on ETF proxies "
                       f"({', '.join(FACTOR_PROXIES.values())}). A market beta near 1 "
                       "with a high R² means an index fund would deliver the same "
                       "exposure at lower cost. Proxies are USD-denominated, so for a "
                       "non-USD base currency part of the residual is currency, not alpha.")
        except (PortfolioLabError, KeyError) as e:
            st.error(str(e))


def tab_method(res):
    """Methodology and glossary - what every number means and how it is computed."""
    p = res["params"]
    note("Every figure in this app is defined here, with the formula and the "
         "assumption behind it. Nothing is computed by a black box.", "info")

    heading("Pipeline", "In order, from raw prices to an order list")
    st.markdown("""
1. **Prices** are split- and dividend-adjusted closes, so they are total return.
2. **Currency conversion** happens *before* returns are computed. Converting after
   would mix units inside the covariance matrix and corrupt every correlation.
3. **Common history** is truncated to the dates every holding shares, so all
   covariance pairs are estimated over the same window.
4. **Returns** are simple (arithmetic), never logarithmic: log returns are not
   additive across assets, so `w·μ_log` understates portfolio return by about σ²/2.
5. **Estimation frequency** defaults to weekly when holdings span several venues,
   because non-synchronous closes depress measured daily correlations.
6. **Σ** is shrunk toward a constant-correlation target (Ledoit-Wolf 2003); **μ**
   comes from market equilibrium (Black-Litterman), history, or is not estimated.
7. **Feasibility** of the constraint set is proved by a linear program before any
   solver runs. Infeasible sets raise an error instead of silently returning 1/N.
8. **Optimisation** uses SLSQP with multi-start for mean-variance and risk parity,
   and a linear program for minimum CVaR.
9. **Evaluation** is net of broker fees, half-spread and ongoing charges, on a
   single explicitly chosen rebalancing policy.
10. **Validation** re-optimises through history on a rolling window using only past
    data, and compares the result to 1/N with a Memmel test.
""")

    heading("Return and risk")
    st.markdown("""
| Term | Definition | Note |
|---|---|---|
| Arithmetic return | Mean period return × periods per year | What mean-variance optimises |
| CAGR | (V_end / V_start)^(1/years) − 1 | Compounded; ≈ arithmetic − σ²/2 |
| Volatility (σ) | Standard deviation × √periods | Symmetric: penalises gains too |
| Sharpe | (μ − r_f) / σ | Excess return per unit of total risk |
| Sortino | (μ − r_f) / downside deviation | Only deviations below the threshold |
| Calmar | CAGR / abs(max drawdown) | Return per unit of worst loss |
| Max drawdown | min(V_t / max(V_≤t) − 1) | Worst peak-to-trough fall |
| Underwater days | Longest run below the prior peak | Patience actually required |
| Ulcer index | √mean(drawdown²) | Penalises deep *and* long drawdowns |
| VaR 95% | 5th percentile of returns, sign-flipped | Historical, no normality assumed |
| CVaR 95% | Mean of returns below the VaR | Coherent; captures tail depth |
| Skew / excess kurtosis | 3rd / 4th standardised moments | Negative skew and fat tails are the danger case |
""")

    heading("Portfolio construction")
    st.markdown("""
| Method | Objective | Depends on μ? |
|---|---|---|
| Min Variance | min wᵀΣw | No |
| Risk Parity | equalise wᵢ(Σw)ᵢ/σ_p | No |
| Max Sharpe | max (wᵀμ − r_f)/√(wᵀΣw) | Yes, heavily |
| Min CVaR | Rockafellar-Uryasev linear program | No |
| Resampled | mean of Max Sharpe over bootstrap draws | Yes, but averaged |
| Equal weight | wᵢ = 1/N | No |

All are solved subject to Σw = 1, per-asset bounds, and group bounds
(gmin ≤ Σ_group w ≤ gmax) for instrument type, asset class, sector and country.
""")

    heading("Diagnostics")
    st.markdown("""
- **Diversification ratio** — (Σ wᵢσᵢ) / σ_p. Equals 1 for a single asset; rises as
  correlations fall. Below about 1.2 the portfolio is one bet in disguise.
- **Effective N** — 1 / Σwᵢ². The number of equally weighted holdings that would be
  as concentrated as yours.
- **Risk contribution** — wᵢ(Σw)ᵢ / σ_p, summing to 100%. A 10% position can supply
  40% of the risk; that gap is the most common surprise in a portfolio.
- **SE(μ)** — σ / √years. The standard error of an annualised mean return. With 10
  years of a 20% volatility asset it is about 6.3 points, which is why confidence
  intervals on expected returns usually straddle zero.
- **Ledoit-Wolf δ** — weight given to the structured target. Higher means the raw
  sample correlations were judged too noisy to trust.
- **Memmel test** — corrects the Jobson-Korkie statistic for the correlation between
  two Sharpe ratios. p below 0.05 means the gap versus 1/N is unlikely to be luck.
""")

    heading("Assumptions you are accepting")
    st.markdown(f"""
- Risk-free rate **{p['rf']*100:.2f}%**, constant over the whole window.
- Rebalancing **{[k for k, v in REBALANCE_CHOICES.items() if v == p['rebalance']][0]}**,
  costing {p['costs'].one_way*1e4:.0f} bps of traded notional each way, plus
  {p['costs'].ter_bps:.0f} bps a year of ongoing charges.
- Projections resample the observed distribution in blocks: no future crisis can be
  worse than the worst one already in your history.
- Public price feeds exclude delisted companies, so every backtest is mildly
  flattering (survivorship bias).
- Dividends are assumed reinvested with no foreign withholding tax.
- The maximum-holdings limit is solved by a greedy heuristic, not to a proven optimum.
""")


def tab_orders(res):
    p, sym = res["params"], res["params"]["sym"]
    note("Turning percentages into an order you could actually place. Most brokers "
         "will not sell you 3.47 shares, so we round down to whole units and show you "
         "the cash left over and how far that pushes you from the target mix.", "info")

    pick = st.selectbox("Mix to buy", list(res["weights"]),
                        index=list(res["weights"]).index(res["recommended"]), key="ord")
    prices = {a: float(res["prices"][a].iloc[-1]) for a in res["assets"]}
    plan = build_execution_plan(res["assets"], res["weights"][pick], p["initial"],
                                prices, lot_size=int(p.get("lot_size", 1)),
                                allow_fractional=p["fractional"], costs=p["costs"])
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Invested", money(plan["Amount invested"].sum(), sym),
                    f"{plan.attrs['invested_ratio']*100:.1f}% of your cash"),
                unsafe_allow_html=True)
    c2.markdown(kpi("Left over", money(plan.attrs["cash"], sym), "cannot buy whole units",
                    ORANGE if plan.attrs["cash"] > p["initial"] * .05 else GOLD),
                unsafe_allow_html=True)
    c3.markdown(kpi("Entry cost", money(plan.attrs["fees"], sym),
                    f"{p['costs'].one_way*1e4:.0f} bps"), unsafe_allow_html=True)
    c4.markdown(kpi("Biggest drift", f"{plan.attrs['max_gap']:.1f} pts", "from target",
                    RED if plan.attrs["max_gap"] > 5 else GOLD), unsafe_allow_html=True)

    d = plan.copy()
    d["Holding"] = [res["holdings"][a]["symbol"] for a in d["Asset"]]
    d["Target weight"] = d["Target weight"].map(lambda x: f"{x*100:.1f}%")
    d["Actual weight"] = d["Actual weight"].map(lambda x: f"{x*100:.1f}%")
    for c in ("Target amount", "Amount invested", "Price"):
        d[c] = d[c].map(lambda x: money(x, sym) if np.isfinite(x) else "-")
    d["Quantity"] = d["Quantity"].map(
        lambda x: (f"{x:,.4f}" if p["fractional"] else f"{x:,.0f}") if np.isfinite(x) else "-")
    d["Drift (pts)"] = d["Drift (pts)"].map(lambda x: f"{x:+.2f}")
    st.dataframe(d[["Holding", "Target weight", "Target amount", "Price", "Quantity",
                    "Amount invested", "Actual weight", "Drift (pts)"]],
                 use_container_width=True, hide_index=True)
    if plan.attrs["missing"]:
        st.warning("No price available for: " + ", ".join(plan.attrs["missing"]))
    st.caption(f"Prices are the last close of your analysis window, converted to "
               f"{p['base_ccy']}. Check the live price before placing any order.")
    st.download_button("⬇ Download this order list (CSV)",
                       plan.to_csv(index=False).encode(), "order_list.csv", "text/csv")

    if any(res["countries"][a] != "Unknown" for a in res["assets"]):
        heading("French PEA eligibility", "Indicative - confirm with your broker")
        st.dataframe(pd.DataFrame({
            "Holding": [res["holdings"][a]["symbol"] for a in res["assets"]],
            "Issuer country": [res["countries"][a] for a in res["assets"]],
            "Type": [res["types"][a] for a in res["assets"]],
            "ISIN": [res["holdings"][a].get("isin", "") or "-" for a in res["assets"]],
            "PEA": ["✅ likely" if pea_eligible(res["countries"][a],
                                                res["holdings"][a].get("exchange", ""),
                                                res["types"][a]) else "❌ unlikely"
                    for a in res["assets"]]}),
            use_container_width=True, hide_index=True)
        st.caption("Based on the issuer's country from the catalogue, which is the "
                   "criterion that actually matters - not the listing venue.")

    note(f"With <b>{money(p['monthly'], sym)} a month</b> going in, you do not need to "
         f"rebalance often. Your current setting is "
         f"<b>{[k for k,v in REBALANCE_CHOICES.items() if v==p['rebalance']][0].lower()}"
         f"</b>. Directing new deposits toward whatever is underweight is usually "
         f"cheaper than selling.", "gold")


def tab_data(res):
    p = res["params"]
    note("Everything here is what the numbers were built from. If something looks "
         "wrong on the other tabs, it usually starts here.", "info")
    heading("Data quality")
    st.dataframe(res["quality"], use_container_width=True, hide_index=True)
    if res["dropped"]:
        st.warning("Dropped for lack of history: " + ", ".join(res["dropped"]))
    if res.get("failed_downloads"):
        st.warning("Could not download: " + ", ".join(res["failed_downloads"]))

    i = res["fx_info"]
    if i["converted"]:
        (st.success if i["method"] == "series" else st.warning)(
            f"Converted to {i['base']} before any calculation using "
            f"{'daily exchange rates' if i['method']=='series' else 'fixed approximate rates'}.")
    else:
        st.info(f"Everything already trades in {p['base_ccy']}; no conversion needed.")

    heading("Limits actually applied", "Not what the sidebar shows - what the optimiser used")
    c = res["cons"]
    if getattr(c, "relaxed", None):
        note("Your risk profile's guardrails were slightly widened so a valid "
             "portfolio could exist at all: " + "; ".join(c.relaxed)
             + ". This happens when you hold only a few positions.", "warn")
    st.dataframe(pd.DataFrame({
        "Holding": [res["holdings"][a]["symbol"] for a in res["assets"]],
        "Type": [res["types"][a] for a in res["assets"]],
        "Asset class": [res["classes"][a] for a in res["assets"]],
        "Sector": [res["sectors"][a] for a in res["assets"]],
        "Country": [res["countries"][a] for a in res["assets"]],
        "Min": [f"{x*100:.0f}%" for x in c.min_w],
        "Max": [f"{x*100:.0f}%" for x in c.max_w]}),
        use_container_width=True, hide_index=True)
    if c.groups:
        st.dataframe(pd.DataFrame([
            {"Group": g, "Holdings": ", ".join(res["holdings"][res["assets"][i]]["symbol"]
                                               for i in idx),
             "Min": f"{lo*100:.0f}%", "Max": f"{hi*100:.0f}%"}
            for g, (idx, lo, hi) in c.groups.items()]),
            use_container_width=True, hide_index=True)

    heading("Audit trail", "So any allocation can be reproduced and justified later")
    st.json(res["audit"], expanded=False)
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Audit record (JSON)",
                       json.dumps(res["audit"], indent=2).encode(),
                       f"audit_{res['audit']['run_id']}.json", "application/json")
    out = pd.DataFrame({"Holding": [res["holdings"][a]["symbol"] for a in res["assets"]],
                        "Type": [res["types"][a] for a in res["assets"]],
                        "Asset class": [res["classes"][a] for a in res["assets"]],
                        "Expected return": res["mu"],
                        "Estimate error": res["diag"]["se_mu"],
                        "Volatility": np.sqrt(np.diag(res["cov"]))})
    for k, w in res["weights"].items():
        out[k] = w
    c2.download_button("⬇ Full results (CSV)", out.to_csv(index=False).encode(),
                       "portfolio_lab.csv", "text/csv")

    st.markdown("""<div style="margin-top:28px;padding:18px;border-top:1px solid
rgba(212,175,55,.12);color:#64748B;font-size:.73rem;line-height:1.7;">
<b style="color:#94A3B8;">What this tool cannot do</b><br>
• Market data comes from public feeds that exclude delisted companies, so every
backtest here is slightly flattering.<br>
• Expected returns stay noisy even after smoothing and resampling.<br>
• Tax and account-eligibility figures are indicative, not a tax opinion.<br>
• Dividends are assumed reinvested; foreign withholding tax is ignored.<br>
• The holdings limit is solved by a shortcut, not to a proven optimum.<br><br>
Educational simulation. Not investment advice.</div>""", unsafe_allow_html=True)


def landing():
    st.markdown("""<div style="background:linear-gradient(135deg,#0D1828,#121F33);
border:1px solid rgba(212,175,55,.18);border-radius:16px;padding:34px;margin-top:14px;">
<div style="font-size:2.2rem;">📊</div>
<div style="font-size:1.15rem;font-weight:600;color:#EEF2F7;margin:10px 0;">
Let's build a portfolio you can actually live with</div>
<div style="color:#94A3B8;font-size:.88rem;line-height:1.8;max-width:680px;">
Three steps in the sidebar: tell us about your money, how much risk you can stomach,
and what you want to hold. Then press the button.<br><br>
<b style="color:#D4AF37;">What makes this different:</b> most tools show you the mix
that would have worked best in the past, which is easy and misleading. This one also
rebuilds the portfolio through history using only what was knowable at the time,
compares it against simply splitting your money evenly, and shows you how uncertain
the estimates really are. Fees, rounding to whole shares, and tax are all included.
</div></div>""", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Portfolio Lab", page_icon="📈", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div style="display:flex;align-items:center;gap:15px;padding:16px 0;
border-bottom:1px solid rgba(212,175,55,.15);margin-bottom:16px;">
<div style="width:44px;height:44px;background:linear-gradient(135deg,#B8860B,#D4AF37);
border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:21px;">📈</div>
<div><div style="font-family:Playfair Display,serif;font-size:1.8rem;font-weight:700;
background:linear-gradient(135deg,#C9A440,#F0D060,#C9A440);-webkit-background-clip:text;
-webkit-text-fill-color:transparent;">Portfolio Lab</div>
<div style="color:#64748B;font-size:.71rem;letter-spacing:.14em;text-transform:uppercase;">
Build · test honestly · project</div></div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="legal">⚖️ Educational simulation — not personalised '
                'investment advice. Past performance does not predict future returns.'
                '</div>', unsafe_allow_html=True)

    p = sidebar()
    if p["run"]:
        status = st.status("Working…", expanded=True)
        try:
            st.session_state.res = run_analysis(p, status.write)
            status.update(label=f"Done · run {st.session_state.res['audit']['run_id']}",
                          state="complete", expanded=False)
        except PortfolioLabError as e:
            status.update(label="Stopped", state="error", expanded=False)
            st.error(f"**We stopped rather than show you something wrong.** {e}")
            st.stop()
        except Exception as e:
            status.update(label="Unexpected error", state="error", expanded=False)
            st.error(f"**Unexpected error:** {type(e).__name__} — {e}")
            st.stop()

    if "res" not in st.session_state:
        landing()
        return
    res = st.session_state.res
    tabs = st.tabs(["  🥧  Your portfolio  ", "  🔮  Your plan  ",
                    "  🎯  Reality check  ", "  ⚠️  Risk  ",
                    "  📐  Analytics  ", "  🧾  What to buy  ",
                    "  📖  Method  ", "  🗂️  Data  "])
    for tab, fn in zip(tabs, [tab_portfolio, tab_plan, tab_reality, tab_risk,
                              tab_analytics, tab_orders, tab_method, tab_data]):
        with tab:
            try:
                fn(res)
            except PortfolioLabError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error in this tab: {type(e).__name__} — {e}")


# ══ Self-tests ══════════════════════════════════════════════════════════════

def _toy(n=3, T=1000, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mus, sds = np.linspace(.08, .02, n) / PERIODS, np.linspace(.20, .05, n) / np.sqrt(PERIODS)
    return pd.DataFrame(rng.normal(mus, sds, (T, n)),
                        index=pd.bdate_range("2015-01-01", periods=T),
                        columns=[f"A{i}" for i in range(n)])


def run_self_tests(verbose=True) -> int:
    checks, failed = [], []

    def test(name):
        def deco(fn):
            checks.append((name, fn)); return fn
        return deco

    @test("port_vol with identity covariance")
    def _():
        assert abs(port_vol([.6, .4], np.eye(2) * .04) - np.sqrt(.04 * .52)) < 1e-12

    @test("max drawdown on a known series")
    def _():
        nav = pd.Series([100, 120, 60, 90], index=pd.bdate_range("2020-01-01", periods=4))
        assert abs(max_drawdown(nav) + .5) < 1e-12 and underwater_days(nav) == 2

    @test("CVaR is at least as large as VaR")
    def _():
        r = pd.Series(np.random.default_rng(1).standard_t(4, 3000) / 100)
        assert historical_cvar(r) >= historical_var(r)

    @test("Sortino ignores upside volatility")
    def _():
        b = pd.Series(np.random.default_rng(3).normal(4e-4, .01, 2000))
        u = b.copy(); u[u > 0] *= 3
        assert sortino(u) > sortino(b)

    @test("risk contributions sum to 100%")
    def _():
        A = np.random.default_rng(5).normal(size=(4, 4))
        assert abs(risk_contributions([.4, .3, .2, .1], A @ A.T / 10).sum() - 100) < 1e-8

    @test("Ledoit-Wolf returns a valid PSD matrix and delta in [0,1]")
    def _():
        cov, d = ledoit_wolf_cov(_toy(4, 300))
        assert 0 <= d <= 1 and np.all(np.linalg.eigvalsh(cov) > -1e-10)

    @test("Ledoit-Wolf shrinks more when history is short")
    def _():
        rng = np.random.default_rng(4)
        def make(T):
            f = rng.normal(0, .01, (T, 1))
            return pd.DataFrame(f @ rng.uniform(.5, 1.5, (1, 6)) + rng.normal(0, .008, (T, 6)),
                                index=pd.bdate_range("2015-01-01", periods=T))
        assert ledoit_wolf_cov(make(120))[1] > ledoit_wolf_cov(make(1500))[1]

    @test("weekly estimation fixes time-zone correlation bias")
    def _():
        rng = np.random.default_rng(3); T = 3000
        f = rng.normal(0, .009, T)
        r = pd.DataFrame({"US": f + rng.normal(0, .005, T),
                          "EU": .55 * f + .45 * np.roll(f, 1) + rng.normal(0, .005, T)},
                         index=pd.bdate_range("2014-01-01", periods=T))
        _, cd, dd = estimate_moments(r, ModelSpec(cov_method="sample", est_freq="D"))
        _, cw, dw = estimate_moments(r, ModelSpec(cov_method="sample", est_freq="W"))
        corr = lambda c: c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])
        assert dw["periods_per_year"] == 52 and dd["est_freq"] == "D"
        assert corr(cw) > corr(cd) + .10 and np.sqrt(cw[1, 1]) > np.sqrt(cd[1, 1]) * 1.1

    @test("falls back to daily when a coarser frequency leaves too few points")
    def _():
        assert estimate_moments(_toy(2, 60), ModelSpec(est_freq="M"))[2]["est_freq"] == "D"

    @test("simple returns, not log returns")
    def _():
        p = pd.DataFrame({"A": [100., 110., 121.]}, index=pd.bdate_range("2020-01-01", periods=3))
        assert np.allclose(simple_returns(p)["A"].values, [.1, .1])

    @test("infeasible minimums raise instead of silently returning 1/n")
    def _():
        for c in (Constraints(3, np.full(3, .5), np.ones(3)),
                  Constraints(3, np.zeros(3), np.full(3, .2))):
            try:
                check_feasibility(c)
            except InfeasibleConstraints:
                continue
            raise AssertionError("no error raised")

    @test("group maximum below individual minimums is rejected")
    def _():
        c = Constraints(3, np.array([.4, .4, 0]), np.ones(3), {"g": ([0, 1], 0., .5)})
        try:
            check_feasibility(c)
        except InfeasibleConstraints as e:
            assert "already exceeded" in str(e); return
        raise AssertionError("no error raised")

    @test("a cap on a group holding everything is explained clearly")
    def _():
        c = Constraints(3, groups={"Type: ETF": ([0, 1, 2], 0., .9)})
        try:
            check_feasibility(c)
        except InfeasibleConstraints as e:
            assert "Every holding belongs to this group" in str(e); return
        raise AssertionError("no error raised")

    @test("group minimum above reachable caps is rejected")
    def _():
        c = Constraints(3, np.zeros(3), np.array([.2, .2, 1.]), {"g": ([0, 1], .8, 1.)})
        try:
            check_feasibility(c)
        except InfeasibleConstraints as e:
            assert "cannot be reached" in str(e); return
        raise AssertionError("no error raised")

    @test("group minimum and maximum are both honoured by the solver")
    def _():
        cov = sample_cov(_toy(4, 700))
        cons = Constraints(4, groups={"bonds": ([2, 3], .30, .50)})
        for w in (solve_min_variance(cov, cons), solve_risk_parity(cov, cons)):
            assert .30 - 1e-4 <= w[[2, 3]].sum() <= .50 + 1e-4

    @test("group limits also bind in the CVaR program")
    def _():
        r = _toy(4, 400)
        w = solve_min_cvar(r, Constraints(4, groups={"g": ([0, 1], .40, .60)}))
        assert .40 - 1e-4 <= w[[0, 1]].sum() <= .60 + 1e-4

    @test("min variance is not beaten by equal weight")
    def _():
        cov = sample_cov(_toy(3, 600))
        assert port_vol(solve_min_variance(cov, Constraints(3)), cov) <= \
            port_vol(np.ones(3) / 3, cov) + 1e-9

    @test("risk parity equalises risk contributions")
    def _():
        cov = np.diag([.04, .09, .01])
        rc = risk_contributions(solve_risk_parity(cov, Constraints(3)), cov)
        assert rc.max() - rc.min() < 1.0

    @test("max Sharpe refuses to run without expected returns")
    def _():
        try:
            solve_max_sharpe(np.zeros(3), np.eye(3) * .04, Constraints(3), 0.)
        except OptimizationError:
            return
        raise AssertionError("should have refused")

    @test("feasible return range respects caps")
    def _():
        mu = np.array([.10, .05, .02])
        lo, hi = feasible_return_range(mu, Constraints(3, np.zeros(3), np.full(3, .4)))
        assert abs(hi - (.4 * .10 + .4 * .05 + .2 * .02)) < 1e-9 and lo < hi

    @test("efficient frontier is monotone in return")
    def _():
        mu, cov, _ = estimate_moments(_toy(3, 600), ModelSpec(cov_method="sample"))
        fv, fr, _ = efficient_frontier(mu, cov, Constraints(3), 12)
        assert len(fv) >= 5 and np.all(np.diff(fr) > -1e-9)

    @test("min CVaR beats equal weight on CVaR")
    def _():
        r = _toy(3, 400)
        w = solve_min_cvar(r, Constraints(3))
        f = lambda ww: historical_cvar(pd.Series(r.values @ ww, index=r.index))
        assert abs(w.sum() - 1) < 1e-6 and f(w) <= f(np.ones(3) / 3) + 1e-6

    @test("Black-Litterman with no views returns the equilibrium")
    def _():
        cov = np.diag([.04, .09])
        s = ModelSpec(bl_prior_w=np.array([.6, .4]))
        assert np.allclose(black_litterman_mu(cov, s, .02),
                           2.5 * (cov @ np.array([.6, .4])) + .02)

    @test("Black-Litterman shifts toward a stated view")
    def _():
        cov = np.diag([.04, .09])
        base = ModelSpec(bl_prior_w=np.array([.5, .5]))
        v = ModelSpec(bl_prior_w=np.array([.5, .5]), bl_views=np.array([.25, np.nan]),
                      bl_confidence=np.array([.9, 0.]))
        assert black_litterman_mu(cov, v, 0.)[0] > black_litterman_mu(cov, base, 0.)[0]

    @test("flat returns and zero fees leave value unchanged")
    def _():
        r = pd.DataFrame(0., index=pd.bdate_range("2020-01-01", periods=100),
                         columns=["A", "B"])
        p = portfolio_path(r, [.5, .5], costs=CostModel(0, 0, 0), charge_initial=False)
        assert abs(p["nav"].iloc[-1] - 1) < 1e-12

    @test("fees always reduce value")
    def _():
        r = _toy(2, 500)
        a = portfolio_path(r, [.5, .5], rebalance="M", costs=CostModel(0, 0, 0))
        b = portfolio_path(r, [.5, .5], rebalance="M", costs=CostModel(20, 10, 50))
        assert b["nav"].iloc[-1] < a["nav"].iloc[-1]

    @test("threshold rebalancing trades less than monthly")
    def _():
        r = _toy(3, 1200)
        w = [.4, .35, .25]
        m = portfolio_path(r, w, rebalance="M", costs=CostModel(0, 0, 0))
        t = portfolio_path(r, w, rebalance="T5", costs=CostModel(0, 0, 0))
        assert t["annual_turnover"] < m["annual_turnover"]

    @test("buy-and-hold differs from rebalanced, and trades nothing")
    def _():
        r = _toy(2, 500)
        bh = portfolio_path(r, [.5, .5], rebalance="none", costs=CostModel(0, 0, 0))
        rb = portfolio_path(r, [.5, .5], rebalance="D", costs=CostModel(0, 0, 0))
        assert bh["annual_turnover"] == 0 and abs(bh["nav"].iloc[-1] - rb["nav"].iloc[-1]) > 1e-6

    @test("total return and drawdown share the same baseline")
    def _():
        p = portfolio_path(_toy(3, 400), [.5, .3, .2], rebalance="M",
                           costs=CostModel(0, 0, 0))
        s = full_stats(p["nav"], p["returns"], 0.)
        assert abs(s["total_return"] - (p["nav"].iloc[-1] - 1)) < 1e-9

    @test("rebalance flags fire once per period")
    def _():
        idx = pd.bdate_range("2020-01-01", "2020-12-31")
        assert rebalance_flags(idx, "M").sum() == 11
        assert rebalance_flags(idx, "none").sum() == 0
        assert rebalance_flags(idx, "D").sum() == len(idx)

    @test("walk-forward never uses future data")
    def _():
        r = _toy(3, 1400)
        bt = walk_forward(r, method="min_variance", cons=Constraints(3), rf=0.,
                          spec=ModelSpec(cov_method="sample"), lookback_years=2.,
                          reb_freq="Q", costs=CostModel(0, 0, 0))
        d0 = bt["weights"].index[0]
        assert d0 >= r.index[int(2 * PERIODS) - 1]
        assert bt["nav"].index[0] == d0 and bt["returns"].index[0] > d0

    @test("walk-forward refuses a history that is too short")
    def _():
        try:
            walk_forward(_toy(3, 200), method="min_variance", cons=Constraints(3),
                         rf=0., spec=ModelSpec(), lookback_years=3.)
        except DataError:
            return
        raise AssertionError("should have refused")

    @test("block bootstrap keeps to the observed support")
    def _():
        h = np.array([.01, -.02, .03] * 20)
        pth = block_bootstrap(h, 60, 200, 6)
        assert pth.shape == (200, 60) and np.all(np.isin(np.round(pth, 10), np.round(h, 10)))

    @test("deterministic returns compound exactly")
    def _():
        s = simulate_wealth(np.full(60, .01), initial=1000, monthly_contribution=0,
                            horizon_years=1, n_sims=50, block=3, inflation=0., tax_rate=0.)
        assert abs(np.median(s["terminal"]) - 1000 * 1.01 ** 12) < 1e-6

    @test("money-weighted return on a doubling equals 100%")
    def _():
        assert abs(money_weighted_return(np.concatenate([[-100.], np.zeros(12)]), 200.) - 1) < 1e-6

    @test("contributions raise the final value")
    def _():
        h = np.full(120, .005)
        a = simulate_wealth(h, initial=1000, monthly_contribution=0, horizon_years=5,
                            n_sims=50, block=6)
        b = simulate_wealth(h, initial=1000, monthly_contribution=100, horizon_years=5,
                            n_sims=50, block=6)
        assert np.median(b["terminal"]) > np.median(a["terminal"])

    @test("tax applies only to gains")
    def _():
        s = simulate_wealth(np.zeros(24), initial=1000, monthly_contribution=0,
                            horizon_years=2, n_sims=20, block=6, inflation=0., tax_rate=.30)
        assert abs(np.median(s["after_tax"]) - 1000) < 1e-6

    @test("monthly aggregation drops partial months")
    def _():
        m = to_monthly(pd.Series(.001, index=pd.bdate_range("2020-01-20", "2022-03-10")))
        assert len(m) == 25 and np.ptp(m) < .01

    @test("order plan rounds to whole units and reports leftover cash")
    def _():
        pl = build_execution_plan(["A", "B"], [.5, .5], 1000, {"A": 300., "B": 70.},
                                  costs=CostModel(0, 0, 0))
        assert list(pl["Quantity"]) == [1., 7.] and abs(pl.attrs["cash"] - 210) < 1e-9

    @test("lot size is respected")
    def _():
        pl = build_execution_plan(["A"], [1.], 1000, {"A": 90.}, lot_size=5,
                                  allow_fractional=False, costs=CostModel(0, 0, 0))
        assert pl["Quantity"].iloc[0] == 10.

    @test("fractional shares hit the target exactly")
    def _():
        pl = build_execution_plan(["A", "B"], [.7, .3], 1000, {"A": 137., "B": 41.},
                                  allow_fractional=True, costs=CostModel(0, 0, 0))
        assert pl.attrs["max_gap"] < 1e-6

    @test("currency conversion happens before returns are computed")
    def _():
        idx = pd.bdate_range("2020-01-01", periods=50)
        pr = pd.DataFrame({"US": np.linspace(100, 110, 50),
                           "EU": np.linspace(50, 55, 50)}, index=idx)
        fx = pd.DataFrame({"USD": np.linspace(.90, 1.00, 50), "EUR": np.ones(50)}, index=idx)
        c, i = convert_to_base(pr, {"US": "USD", "EU": "EUR"}, "EUR", fx)
        assert i["method"] == "series" and abs(c["US"].iloc[0] - 90) < 1e-9
        assert abs(c["US"].pct_change().iloc[1] - pr["US"].pct_change().iloc[1]) > 1e-6

    @test("missing FX raises instead of guessing")
    def _():
        pr = pd.DataFrame({"X": np.ones(10)}, index=pd.bdate_range("2020-01-01", periods=10))
        try:
            convert_to_base(pr, {"X": "JPY"}, "EUR", None, allow_static=False)
        except FXError:
            return
        raise AssertionError("no error raised")

    @test("common history is trimmed and exclusions reported")
    def _():
        idx = pd.bdate_range("2018-01-01", periods=800)
        df = pd.DataFrame({"a": np.linspace(1, 2, 800), "b": np.linspace(1, 3, 800),
                           "c": [np.nan] * 700 + list(np.linspace(1, 1.1, 100))}, index=idx)
        sub, dropped = align_common_history(df, 252)
        assert dropped == ["c"] and len(sub.columns) == 2

    @test("quality report flags a frozen price series")
    def _():
        rep = data_quality_report(pd.DataFrame(
            {"flat": np.ones(400)}, index=pd.bdate_range("2020-01-01", periods=400)))
        assert "clean" not in rep.iloc[0]["Warnings"]

    @test("instrument type and asset class are detected")
    def _():
        assert instrument_type("SPY", "ETF") == "ETF"
        assert instrument_type("MC.PA", "Stock") == "Stock"
        assert classify_asset("MC.PA", "STOCK") == "Stocks"
        assert instrument_type("BTC-USD") == "Crypto" and instrument_type("^GSPC") == "Index"
        assert classify_asset("AGG") == "Bonds" and classify_asset("GLD") == "Commodities"
        assert classify_asset("AAPL", "EQUITY") == "Stocks"

    @test("type and class limits are turned into group constraints")
    def _():
        assets = ["a", "b", "c"]
        types = {"a": "Stock", "b": "ETF", "c": "ETF"}
        classes = {"a": "Stocks", "b": "Bonds", "c": "Stocks"}
        cons = build_constraints(assets, types, classes,
                                 {"profile": "Aggressive", "apply_profile": False,
                                  "type_limits": {"ETF": (.2, .6)},
                                  "class_limits": {"Bonds": (.1, .4)}})
        assert cons.groups["Type: ETF"] == ([1, 2], .2, .6)
        assert cons.groups["Class: Bonds"] == ([1], .1, .4)

    @test("catalogue rows normalise into holdings")
    def _():
        row = {"symbol": "MC.PA", "name": "LVMH", "type": "Stock", "exchange": "PAR",
               "currency": "EUR", "sector": "Consumer Discretionary",
               "country": "France", "isin": "FR0000121014", "market_cap": "Mega Cap"}
        h = catalogue_holding(row)
        assert h["source"] == "yahoo" and h["currency"] == "EUR"
        assert h["country"] == "France" and h["asset_class"] == "Stocks"

    @test("catalogue search ranks exact symbol matches first")
    def _():
        cat = pd.DataFrame({"symbol": ["AAPLX", "AAPL"], "name": ["Fund", "Apple Inc"],
                            "type": ["Fund", "Stock"], "exchange": ["NMS", "NMS"],
                            "currency": ["USD", "USD"], "sector": ["", "Technology"],
                            "country": ["", "United States"],
                            "market_cap": ["", "Mega Cap"], "isin": ["", "US0378331005"]})
        cat["haystack"] = (cat["symbol"] + " " + cat["name"]).str.lower()
        cat["label"] = cat["symbol"]
        assert search_catalogue(cat, "aapl")[0]["symbol"] == "AAPL"
        assert search_catalogue(cat, "a") == []
        assert [h["symbol"] for h in search_catalogue(cat, "aapl", ["Fund"])] == ["AAPLX"]

    @test("PEA eligibility follows the issuer country, not the listing venue")
    def _():
        assert pea_eligible("France", "PAR", "Stock")
        assert not pea_eligible("United States", "PAR", "Stock")   # US firm listed in Paris
        assert not pea_eligible("France", "PAR", "Crypto")
        assert pea_eligible("", "PAR", "Stock")                    # fallback on venue

    @test("sector and country limits become group constraints")
    def _():
        assets = ["a", "b", "c"]
        cons = build_constraints(
            assets, {x: "Stock" for x in assets},
            {x: "Stocks" for x in assets},
            {"profile": "Aggressive", "apply_profile": False,
             "sectors": {"a": "Tech", "b": "Tech", "c": "Energy"},
             "countries": {"a": "France", "b": "United States", "c": "France"},
             "sector_limits": {"Tech": (0.0, .5)},
             "country_limits": {"France": (.3, .8)}})
        assert cons.groups["Sector: Tech"] == ([0, 1], 0.0, .5)
        assert cons.groups["Country: France"] == ([0, 2], .3, .8)

    @test("a single-group dimension produces no constraint")
    def _():
        cons = build_constraints(
            ["a", "b"], {"a": "Stock", "b": "Stock"}, {"a": "Stocks", "b": "Stocks"},
            {"profile": "Aggressive", "apply_profile": False,
             "sectors": {"a": "Unclassified", "b": "Unclassified"},
             "sector_limits": {}})
        assert not cons.groups

    @test("risk profile guardrails cap single holdings and risky assets")
    def _():
        assets = list("abcd")
        cons = build_constraints(assets, {a: "Stock" for a in assets},
                                 {"a": "Stocks", "b": "Stocks", "c": "Bonds", "d": "Bonds"},
                                 {"profile": "Cautious", "apply_profile": True})
        # 4 lines capped at 25% plus a 40% risky cap cannot reach 100%, so the
        # per-holding cap widens to 30% instead of failing.
        assert np.allclose(cons.max_w, .30) and cons.groups["Risky assets"][2] == .40
        assert cons.relaxed
        wide = build_constraints(list("abcdefgh"), {a: "Stock" for a in "abcdefgh"},
                                 {a: ("Stocks" if a < "e" else "Bonds") for a in "abcdefgh"},
                                 {"profile": "Cautious", "apply_profile": True})
        assert np.allclose(wide.max_w, .25) and not wide.relaxed

    @test("60/40 needs both stocks and bonds")
    def _():
        w = sixty_forty(["S", "E", "B"], {"S": "Stocks", "E": "Stocks", "B": "Bonds"})
        assert abs(w[0] - .30) < 1e-12 and abs(w[2] - .40) < 1e-12
        assert sixty_forty(["S"], {"S": "Stocks"}) is None

    @test("recommendation follows the risk profile")
    def _():
        avail = ["Max Sharpe", "Min Variance", "Risk Parity", "Resampled"]
        assert recommended_portfolio("Cautious", avail) == "Min Variance"
        assert recommended_portfolio("Balanced", avail) == "Risk Parity"
        assert recommended_portfolio("Aggressive", avail) == "Resampled"

    @test("factor regression recovers a beta of one against itself")
    def _():
        r = _toy(2, 500)
        f = factor_regression(r["A0"], r[["A0"]].rename(columns={"A0": "Market"}))
        assert abs(float(f.loc[f["Factor"] == "Market", "Coefficient"].iloc[0]) - 1) < 1e-6

    @test("out-of-sample verdict spots a better strategy")
    def _():
        idx = pd.bdate_range("2020-01-01", periods=800)
        rng = np.random.default_rng(11)
        g = pd.Series(rng.normal(8e-4, .008, 800), index=idx)
        b = pd.Series(rng.normal(1e-4, .012, 800), index=idx)
        v = oos_verdict({"nav": (1 + g).cumprod(), "returns": g},
                        {"nav": (1 + b).cumprod(), "returns": b}, 0.)
        assert v["beats_naive"] and v["sharpe_gap"] > 0

    @test("every chart call carries a unique key (Streamlit duplicate-ID guard)")
    def _():
        import inspect
        src = inspect.getsource(sys.modules[__name__])
        keys, calls = [], 0
        for m in re.finditer(r"plotly_chart\(", src):
            i, depth = src.index("(", m.start()), 0
            for j in range(i, len(src)):
                depth += (src[j] == "(") - (src[j] == ")")
                if depth == 0:
                    break
            body = src[i:j]
            calls += 1
            k = re.search(r"key=(f?\"[^\"]*\"|[A-Za-z_][\w.]*)", body)
            assert k, f"plotly_chart without key: {body[:70]}"
            keys.append(k.group(1))
        assert calls >= 10
        assert len(set(keys)) == len(keys), "duplicate chart keys"

    @test("data fingerprint is stable and sensitive")
    def _():
        df = pd.DataFrame({"A": [1., 2., 3.]}, index=pd.bdate_range("2020-01-01", periods=3))
        d2 = df.copy(); d2.iloc[0, 0] = 1.0001
        assert data_fingerprint(df) == data_fingerprint(df.copy())
        assert data_fingerprint(df) != data_fingerprint(d2)

    ok = 0
    for name, fn in checks:
        try:
            fn(); ok += 1
            if verbose:
                print(f"  ✅  {name}")
        except Exception as e:
            failed.append(name)
            if verbose:
                print(f"  ❌  {name}\n        → {type(e).__name__}: {e}")
    if verbose:
        print(f"\n{ok}/{len(checks)} passed" + (f", {len(failed)} failed" if failed else ""))
    return 0 if not failed else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_self_tests())
    if not HAS_ST:
        print("Streamlit not installed.\n  pip install -r requirements.txt\n"
              "  streamlit run portfolio_lab.py")
        sys.exit(1)
    main()
