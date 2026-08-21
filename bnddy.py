"""
═══════════════════════════════════════════════════════════════════════════════
PORTFOLIO LAB — Streamlit

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
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

# ─────────────────────────────────────────────────────────────────────────────
# Imports tolérants : le cœur mathématique doit s'importer et se tester sans
# Streamlit / Plotly / yfinance (P4.4 — testabilité).
# ─────────────────────────────────────────────────────────────────────────────
try:
    import streamlit as st
    HAS_ST = True
except Exception:                                    # pragma: no cover
    st, HAS_ST = None, False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:                                    # pragma: no cover
    go, HAS_PLOTLY = None, False

try:
    import yfinance as yf
    HAS_YF = True
except Exception:                                    # pragma: no cover
    yf, HAS_YF = None, False


def cache_data(**kw) -> Callable:
    """Décorateur de cache neutre hors Streamlit."""
    def deco(fn):
        return st.cache_data(**kw)(fn) if HAS_ST else fn
    return deco


# ═══════════════════════════════════════════════════════════════════════════
# ERREURS EXPLICITES  (P1.1 — on échoue bruyamment, on ne devine pas)
# ═══════════════════════════════════════════════════════════════════════════

class PortfolioLabError(Exception):
    """Erreur métier destinée à être affichée telle quelle à l'utilisateur."""


class DataError(PortfolioLabError):
    """Données indisponibles, insuffisantes ou incohérentes."""


class FXError(DataError):
    """Taux de change indisponible — P1.3 : jamais de repli silencieux."""


class InfeasibleConstraints(PortfolioLabError):
    """Le jeu de contraintes n'admet aucun portefeuille valide."""


class OptimizationError(PortfolioLabError):
    """Le solveur n'a pas convergé — on le dit au lieu de rendre 1/n."""


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

PERIODS = 252          # jours de bourse par an
MONTHS = 12
# Fréquence d'estimation de μ et Σ. Les rendements quotidiens sous-estiment
# fortement les corrélations entre places de cotation situées dans des fuseaux
# horaires différents : Tokyo et Paris clôturent avant New York, si bien qu'une
# partie du choc mondial du jour n'apparaît chez eux que le lendemain.
PERIODS_PER = {"D": 252, "W": 52, "M": 12}
FREQ_RULE = {"D": None, "W": "W-FRI", "M": "ME"}
SEED = 42
N_MC_CLOUD = 12_000    # nuage de portefeuilles aléatoires (vectorisé, P1.10)

BG, PANEL = "#05080D", "#0A1220"
GRID = "rgba(255,255,255,0.05)"
TEXT, MUTED = "#EEF2F7", "#64748B"
GOLD, GREEN, RED = "#D4AF37", "#10B981", "#EF4444"
BLUE, ORANGE, PURPLE = "#3B82F6", "#F59E0B", "#8B5CF6"
CYAN, PINK = "#06B6D4", "#EC4899"

PALETTE = [GOLD, BLUE, GREEN, RED, PURPLE, ORANGE, CYAN, PINK, "#84CC16", "#F97316"]
PORT_COLORS = {
    "Max Sharpe": GOLD, "Min Variance": BLUE, "Risk Parity": PURPLE,
    "Min CVaR": CYAN, "Equal weight (1/N)": MUTED, "Benchmark 60/40": ORANGE,
}

# ─── Devises ────────────────────────────────────────────────────────────────
# rf_default : taux court indicatif, révisable par l'utilisateur (P3.2).
# Ce n'est PAS une donnée de marché temps réel : l'UI le signale comme hypothèse.
CURRENCIES: dict[str, dict] = {
    "EUR": {"symbol": "€",  "flag": "🇪🇺", "rf_default": 0.0200, "rf_proxy": None},
    "USD": {"symbol": "$",  "flag": "🇺🇸", "rf_default": 0.0425, "rf_proxy": "^IRX"},
    "GBP": {"symbol": "£",  "flag": "🇬🇧", "rf_default": 0.0400, "rf_proxy": None},
    "CHF": {"symbol": "CHF", "flag": "🇨🇭", "rf_default": 0.0050, "rf_proxy": None},
    "JPY": {"symbol": "¥",  "flag": "🇯🇵", "rf_default": 0.0050, "rf_proxy": None},
    "CAD": {"symbol": "C$", "flag": "🇨🇦", "rf_default": 0.0300, "rf_proxy": None},
    "AUD": {"symbol": "A$", "flag": "🇦🇺", "rf_default": 0.0385, "rf_proxy": None},
}

# ─── Classes d'actifs (P3.4) ────────────────────────────────────────────────
ASSET_CLASSES = ["Equity", "Bond", "Commodity", "Crypto", "Real estate", "Cash", "Other"]

CLASS_HINTS = {
    "Bond": ("AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "GOVT",
             "AGGH", "IEAG", "EUNA", "VGEA", "IBTA", "SEGA"),
    "Commodity": ("GLD", "IAU", "SLV", "DBC", "PDBC", "USO", "GC=F", "SI=F", "CL=F"),
    "Real estate": ("VNQ", "IYR", "SCHH", "REET", "RWO", "IPRP", "EPRA"),
    "Cash": ("BIL", "SHV", "SGOV", "ERNA", "XEON", "CSH2"),
}

# ─── Frais par défaut (P2.4) ────────────────────────────────────────────────
DEFAULT_BROKER_BPS = 10.0     # commission aller simple, en points de base
DEFAULT_SPREAD_BPS = 5.0      # demi-spread payé à l'achat comme à la vente
DEFAULT_TER_BPS = 20.0        # frais courants annuels moyens d'un ETF

# ─── Fiscalité indicative France (P3.1) ─────────────────────────────────────
TAX_REGIMES = {
    "CTO — flat tax 30 %": {"rate": 0.300, "note": "12,8 % IR + 17,2 % PS, à la cession."},
    "PEA (> 5 ans) — 17,2 %": {"rate": 0.172, "note": "Prélèvements sociaux seuls. Actifs éligibles UE uniquement."},
    "Assurance-vie (> 8 ans)": {"rate": 0.247, "note": "Après abattement annuel, hypothèse simplifiée."},
    "Aucune (brut)": {"rate": 0.000, "note": "Rendement avant impôt."},
}

# ─── Scénarios de stress (définitions immuables) ────────────────────────────
# P1.4 : ce dictionnaire n'est JAMAIS muté. Le scénario personnalisé vit
# dans st.session_state, donc isolé par session utilisateur.
SCENARIO_PRESETS: dict[str, dict] = {
    "2008 — Crise financière": {
        "start": "2008-09-01", "end": "2009-03-31", "color": RED,
        "description": "Faillite de Lehman Brothers. S&P 500 −57 % du sommet au creux.",
    },
    "2020 — Krach Covid": {
        "start": "2020-02-19", "end": "2020-03-23", "color": ORANGE,
        "description": "Marché baissier le plus rapide de l'histoire : −34 % en 33 jours.",
    },
    "2022 — Choc de taux": {
        "start": "2022-01-01", "end": "2022-12-31", "color": PURPLE,
        "description": "Fed +425 pb. Actions et obligations baissent ensemble.",
    },
    "2000-2002 — Bulle internet": {
        "start": "2000-03-10", "end": "2002-10-09", "color": PINK,
        "description": "NASDAQ −78 %. Portefeuilles technologiques dévastés.",
    },
    "2011 — Dette souveraine": {
        "start": "2011-07-01", "end": "2011-10-03", "color": CYAN,
        "description": "Craintes sur la Grèce, l'Italie et l'Espagne.",
    },
    "2015 — Dévaluation du yuan": {
        "start": "2015-08-10", "end": "2016-02-11", "color": "#84CC16",
        "description": "Ralentissement chinois, effondrement du pétrole.",
    },
}

# ─── Proxies factoriels (P4.3) ──────────────────────────────────────────────
FACTOR_PROXIES = {
    "Marché": "SPY", "Taille": "IWM", "Value": "VLUE",
    "Momentum": "MTUM", "Qualité": "QUAL", "Faible volatilité": "USMV",
}


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURES DE PARAMÈTRES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CostModel:
    """Frottements. P2.4 — absents du prototype, ils changent les conclusions."""
    broker_bps: float = DEFAULT_BROKER_BPS
    spread_bps: float = DEFAULT_SPREAD_BPS
    ter_bps: np.ndarray | float = DEFAULT_TER_BPS   # scalaire ou vecteur par actif

    @property
    def one_way(self) -> float:
        """Coût aller simple, en fraction du notionnel échangé."""
        return (self.broker_bps + self.spread_bps) / 1e4

    def ter_vector(self, n: int) -> np.ndarray:
        if np.isscalar(self.ter_bps):
            return np.full(n, float(self.ter_bps) / 1e4)
        v = np.asarray(self.ter_bps, dtype=float) / 1e4
        if v.size != n:
            raise ValueError("TER vector length mismatch")
        return v


@dataclass
class Constraints:
    """
    Contraintes linéaires du programme d'optimisation.
    P1.1 — la faisabilité est vérifiée par LP avant tout appel au solveur.
    P3.4 — groupes (classes d'actifs) et cardinalité.
    """
    n: int
    min_w: np.ndarray = None
    max_w: np.ndarray = None
    # nom → (indices, poids mini du groupe, poids maxi du groupe)
    groups: dict[str, tuple[list[int], float, float]] = field(default_factory=dict)
    max_assets: Optional[int] = None          # cardinalité (heuristique)

    def __post_init__(self):
        if self.min_w is None:
            self.min_w = np.zeros(self.n)
        if self.max_w is None:
            self.max_w = np.ones(self.n)
        self.min_w = np.asarray(self.min_w, dtype=float)
        self.max_w = np.asarray(self.max_w, dtype=float)
        if self.min_w.size != self.n or self.max_w.size != self.n:
            raise ValueError("min_w / max_w : dimension incohérente")

    # ── Représentations pour les solveurs ────────────────────────────────
    def bounds(self, floor: float = 0.0) -> list[tuple[float, float]]:
        return [(max(float(self.min_w[i]), floor), float(self.max_w[i]))
                for i in range(self.n)]

    def group_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Renvoie (A_ub, b_ub) tels que A_ub @ w <= b_ub."""
        rows, rhs = [], []
        for _, (idx, gmin, gmax) in self.groups.items():
            sel = np.zeros(self.n)
            sel[idx] = 1.0
            rows.append(sel.copy()); rhs.append(float(gmax))     #  Σ w ≤ gmax
            rows.append(-sel.copy()); rhs.append(-float(gmin))   # -Σ w ≤ -gmin
        if not rows:
            return np.zeros((0, self.n)), np.zeros(0)
        return np.vstack(rows), np.array(rhs)

    def scipy_constraints(self) -> list[dict]:
        """Contraintes SLSQP : somme = 1 + inégalités de groupe."""
        cons: list[dict] = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
        A, b = self.group_matrix()
        for i in range(A.shape[0]):
            cons.append({
                "type": "ineq",
                "fun": (lambda w, a=A[i].copy(), bb=float(b[i]): bb - float(a @ w)),
            })
        return cons

    def subset(self, idx: Sequence[int]) -> "Constraints":
        """Restriction à un sous-ensemble d'actifs (scénarios, cardinalité)."""
        idx = list(idx)
        pos = {old: new for new, old in enumerate(idx)}
        groups = {}
        for name, (gidx, gmin, gmax) in self.groups.items():
            kept = [pos[i] for i in gidx if i in pos]
            if kept:
                groups[name] = (kept, 0.0, gmax)   # borne basse relâchée
        return Constraints(len(idx), self.min_w[idx], self.max_w[idx], groups,
                           self.max_assets)


@dataclass
class ModelSpec:
    """Choix d'estimateurs — le point le plus déterminant pour la robustesse."""
    mu_method: str = "historical"   # historical | black_litterman | none
    cov_method: str = "ledoit_wolf"  # sample | ledoit_wolf
    est_freq: str = "D"             # D | W | M — fréquence d'estimation de μ et Σ
    bl_delta: float = 2.5           # aversion au risque du marché
    bl_tau: float = 0.05
    bl_views: Optional[np.ndarray] = None       # vues absolues, en rendement annuel
    bl_confidence: Optional[np.ndarray] = None  # 0 → aucune, 1 → certitude
    bl_prior_w: Optional[np.ndarray] = None     # poids d'équilibre (cap. boursière)


# ═══════════════════════════════════════════════════════════════════════════
# CŒUR MATHÉMATIQUE — aucune dépendance à Streamlit, entièrement testable
# ═══════════════════════════════════════════════════════════════════════════

# ─── Rendements et moments ──────────────────────────────────────────────────

def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    P1.5 — rendements SIMPLES, pas logarithmiques.
    Les log-rendements ne sont pas additifs entre actifs : w @ log_mu
    sous-estime le rendement du portefeuille d'environ σ²/2. On travaille donc
    en arithmétique pour μ et Σ, et on convertit explicitement en géométrique
    (CAGR) là où c'est la grandeur pertinente.
    """
    return prices.pct_change().dropna(how="all")


def resample_returns(returns: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Recompose les rendements à la fréquence demandée (composition exacte)."""
    rule = FREQ_RULE.get(freq)
    if rule is None:
        return returns
    return ((1.0 + returns).resample(rule).prod() - 1.0).dropna(how="all")


def annualized_mu(returns: pd.DataFrame, periods: int = PERIODS) -> np.ndarray:
    """Espérance arithmétique annualisée."""
    return returns.mean().values * periods


def geometric_to_arithmetic(g: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """μ_arith ≈ μ_geo + σ²/2 — la correction que le prototype omettait."""
    return g + 0.5 * sigma ** 2


def arithmetic_to_geometric(mu: float, sigma: float) -> float:
    """Rendement composé attendu (approximation d'ordre 2)."""
    return mu - 0.5 * sigma ** 2


def sample_cov(returns: pd.DataFrame, periods: int = PERIODS) -> np.ndarray:
    return returns.cov().values * periods


def ledoit_wolf_cov(returns: pd.DataFrame, periods: int = PERIODS) -> tuple[np.ndarray, float]:
    """
    P2.1 — Shrinkage de Ledoit-Wolf vers la cible « corrélation constante »
    (Ledoit & Wolf, 2003). Implémenté ici pour éviter une dépendance et pour
    que l'intensité de shrinkage δ soit exposée à l'utilisateur.

    Renvoie (Σ annualisée, δ) avec δ ∈ [0, 1] : 0 = covariance empirique pure,
    1 = cible entièrement structurée.
    """
    X = returns.values
    X = X[~np.isnan(X).any(axis=1)]
    t, n = X.shape
    if t < 10 or n < 2:
        return sample_cov(returns, periods), 0.0

    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / t                                    # covariance empirique
    var = np.diag(S)
    sd = np.sqrt(np.maximum(var, 1e-18))
    corr = S / np.outer(sd, sd)
    off = ~np.eye(n, dtype=bool)
    r_bar = corr[off].mean()                               # corrélation moyenne
    F = r_bar * np.outer(sd, sd)                           # cible
    np.fill_diagonal(F, var)

    # π : variance d'estimation de S
    Xc2 = Xc ** 2
    pi_mat = (Xc2.T @ Xc2) / t - S ** 2
    pi_hat = pi_mat.sum()

    # ρ : covariance entre erreurs d'estimation de S et de F
    term = ((Xc ** 3).T @ Xc) / t - var[:, None] * S
    theta_ii_ij = term
    theta_jj_ij = term.T
    rho_hat = np.diag(pi_mat).sum() + (
        r_bar * ((np.sqrt(np.outer(var, 1.0 / np.maximum(var, 1e-18))) * theta_ii_ij
                  + np.sqrt(np.outer(1.0 / np.maximum(var, 1e-18), var)) * theta_jj_ij) / 2.0)[off].sum()
    )

    gamma_hat = float(((F - S) ** 2).sum())                # γ : biais de la cible
    if gamma_hat <= 1e-18:
        delta = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = float(np.clip(kappa / t, 0.0, 1.0))

    shrunk = delta * F + (1.0 - delta) * S
    shrunk = 0.5 * (shrunk + shrunk.T)                     # symétrisation
    return shrunk * periods, delta


def nearest_psd(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Projection sur le cône des matrices semi-définies positives."""
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals) @ vecs.T


def estimate_moments(returns: pd.DataFrame, spec: ModelSpec,
                     rf: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Estime (μ, Σ) selon la spécification.
    Renvoie aussi un diagnostic : intensité de shrinkage, erreur-type sur μ.
    """
    n = returns.shape[1]
    freq = spec.est_freq if spec.est_freq in PERIODS_PER else "D"
    r = resample_returns(returns, freq)
    if len(r) < 30 and freq != "D":
        # Trop peu d'observations à cette fréquence : on revient au quotidien
        # plutôt que d'estimer une covariance sur une poignée de points.
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
    # Diagnostic clé : erreur-type de μ annualisé = σ / √(nombre d'années)
    se_mu = vol / np.sqrt(max(t / per, 1e-9))

    if spec.mu_method == "none":
        # Aucun μ crédible : seuls min-variance / risk parity ont du sens.
        mu = np.zeros(n)
    elif spec.mu_method == "black_litterman":
        mu = black_litterman_mu(cov, spec, rf)
    else:
        mu = mu_hist

    diag = {
        "est_freq": freq,
        "periods_per_year": per,
        "shrinkage_delta": delta,
        "se_mu": se_mu,
        "mu_hist": mu_hist,
        "t_obs": t,
        # Ratio bruit/signal : > 1 signifie que l'erreur d'estimation dépasse
        # l'estimation elle-même. C'est presque toujours le cas.
        "noise_ratio": np.abs(se_mu) / np.maximum(np.abs(mu_hist), 1e-9),
    }
    return mu, cov, diag


def black_litterman_mu(cov: np.ndarray, spec: ModelSpec, rf: float) -> np.ndarray:
    """
    P4.1 — Black-Litterman avec vues absolues par actif.
    π = δ Σ w_mkt  (rendements d'équilibre implicites)
    μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹π + PᵀΩ⁻¹Q]
    """
    n = cov.shape[0]
    w_mkt = spec.bl_prior_w if spec.bl_prior_w is not None else np.ones(n) / n
    w_mkt = np.asarray(w_mkt, float)
    w_mkt = w_mkt / w_mkt.sum()
    pi = spec.bl_delta * (cov @ w_mkt) + rf

    if spec.bl_views is None:
        return pi

    views = np.asarray(spec.bl_views, float)
    conf = np.asarray(spec.bl_confidence if spec.bl_confidence is not None
                      else np.full(n, 0.5), float)
    active = ~np.isnan(views) & (conf > 1e-6)
    if not active.any():
        return pi

    P = np.eye(n)[active]
    Q = views[active]
    tau_cov = spec.bl_tau * cov
    base = np.diag(P @ tau_cov @ P.T)
    omega = np.diag(base / np.clip(conf[active], 1e-3, 1.0))

    inv_tau = np.linalg.pinv(tau_cov)
    inv_om = np.linalg.pinv(omega)
    post = np.linalg.pinv(inv_tau + P.T @ inv_om @ P) @ (inv_tau @ pi + P.T @ inv_om @ Q)
    return post


# ─── Mesures de performance et de risque (P2.3) ─────────────────────────────

def port_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(np.asarray(w) @ np.asarray(mu))


def port_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(float(np.asarray(w) @ cov @ np.asarray(w)), 0.0)))


def sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    v = port_vol(w, cov)
    return (port_return(w, mu) - rf) / v if v > 1e-9 else 0.0


def pmetrics(w, mu, cov, rf) -> tuple[float, float, float]:
    """(rendement, volatilité, Sharpe) — signature conservée du prototype."""
    r, v = port_return(w, mu), port_vol(w, cov)
    return r, v, ((r - rf) / v if v > 1e-9 else 0.0)


def drawdown_series(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def max_drawdown(nav: pd.Series) -> float:
    return float(drawdown_series(nav).min())


def underwater_days(nav: pd.Series) -> int:
    """Plus longue période sous le précédent sommet — absente du prototype."""
    dd = drawdown_series(nav).values
    longest = run = 0
    for x in dd:
        run = run + 1 if x < -1e-12 else 0
        longest = max(longest, run)
    return int(longest)


def ulcer_index(nav: pd.Series) -> float:
    dd = drawdown_series(nav).values
    return float(np.sqrt(np.mean(dd ** 2)))


def cagr(nav: pd.Series, periods: int = PERIODS) -> float:
    n_years = len(nav) / periods
    if n_years <= 0.1 or nav.iloc[0] <= 0:
        return 0.0
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / n_years) - 1.0)


def ann_vol_from_series(r: pd.Series, periods: int = PERIODS) -> float:
    return float(r.std(ddof=1) * np.sqrt(periods))


def downside_deviation(r: pd.Series, mar: float = 0.0, periods: int = PERIODS) -> float:
    """Écart-type des seuls rendements sous le seuil (MAR annualisé)."""
    thr = mar / periods
    d = np.minimum(r.values - thr, 0.0)
    return float(np.sqrt(np.mean(d ** 2)) * np.sqrt(periods))


def sortino(r: pd.Series, rf: float = 0.0, periods: int = PERIODS) -> float:
    dd = downside_deviation(r, rf, periods)
    if dd < 1e-12:
        return 0.0
    return float((r.mean() * periods - rf) / dd)


def calmar(nav: pd.Series, periods: int = PERIODS) -> float:
    mdd = abs(max_drawdown(nav))
    return float(cagr(nav, periods) / mdd) if mdd > 1e-9 else 0.0


def historical_var(r: pd.Series, alpha: float = 0.95) -> float:
    """VaR historique (perte positive). Aucune hypothèse de normalité."""
    if len(r) < 20:
        return float("nan")
    return float(-np.quantile(r.values, 1.0 - alpha))


def historical_cvar(r: pd.Series, alpha: float = 0.95) -> float:
    """
    CVaR / Expected Shortfall : perte moyenne au-delà de la VaR.
    Bien plus pertinent que la volatilité sur des rendements à queues épaisses.
    """
    if len(r) < 20:
        return float("nan")
    q = np.quantile(r.values, 1.0 - alpha)
    tail = r.values[r.values <= q]
    return float(-tail.mean()) if tail.size else float("nan")


def tail_stats(r: pd.Series) -> dict:
    x = r.values
    return {
        "skew": float(pd.Series(x).skew()),
        "kurtosis": float(pd.Series(x).kurtosis()),   # excès de kurtosis
        "worst_day": float(x.min()) if x.size else float("nan"),
        "best_day": float(x.max()) if x.size else float("nan"),
    }


def benchmark_stats(r: pd.Series, rb: pd.Series, rf: float = 0.0,
                    periods: int = PERIODS) -> dict:
    """P2.3 — beta, alpha, tracking error, information ratio, R²."""
    df = pd.concat([r, rb], axis=1).dropna()
    if len(df) < 30:
        return {k: float("nan") for k in
                ("beta", "alpha", "tracking_error", "information_ratio", "r2", "corr")}
    x = df.iloc[:, 1].values - rf / periods
    y = df.iloc[:, 0].values - rf / periods
    vx = x.var(ddof=1)
    beta = float(np.cov(y, x, ddof=1)[0, 1] / vx) if vx > 1e-18 else float("nan")
    alpha = float((y.mean() - beta * x.mean()) * periods)
    active = df.iloc[:, 0].values - df.iloc[:, 1].values
    te = float(active.std(ddof=1) * np.sqrt(periods))
    ir = float(active.mean() * periods / te) if te > 1e-12 else float("nan")
    corr = float(np.corrcoef(y, x)[0, 1])
    return {"beta": beta, "alpha": alpha, "tracking_error": te,
            "information_ratio": ir, "r2": corr ** 2, "corr": corr}


def full_stats(nav: pd.Series, rets: pd.Series, rf: float,
               bench_rets: Optional[pd.Series] = None) -> dict:
    """Tableau de bord complet d'une série de portefeuille."""
    out = {
        "cagr": cagr(nav),
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "vol": ann_vol_from_series(rets),
        "sharpe": (rets.mean() * PERIODS - rf) / ann_vol_from_series(rets)
                  if ann_vol_from_series(rets) > 1e-12 else 0.0,
        "sortino": sortino(rets, rf),
        "calmar": calmar(nav),
        "max_drawdown": max_drawdown(nav),
        "underwater_days": underwater_days(nav),
        "ulcer": ulcer_index(nav),
        "var95": historical_var(rets, 0.95),
        "cvar95": historical_cvar(rets, 0.95),
        "var99": historical_var(rets, 0.99),
        "cvar99": historical_cvar(rets, 0.99),
    }
    out.update(tail_stats(rets))
    if bench_rets is not None:
        out.update(benchmark_stats(rets, bench_rets, rf))
    return out


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Contribution au risque en % (somme = 100)."""
    w = np.asarray(w, float)
    pv = np.sqrt(max(float(w @ cov @ w), 0.0))
    if pv < 1e-12:
        return np.zeros(len(w))
    rc = w * ((cov @ w) / pv)
    tot = rc.sum()
    return rc / tot * 100.0 if abs(tot) > 1e-12 else rc * 0.0


def diversification_ratio(w: np.ndarray, cov: np.ndarray) -> float:
    """Σ wᵢσᵢ / σ_p. Vaut 1 pour un actif unique, croît avec la diversification."""
    vol = np.sqrt(np.diag(cov))
    pv = port_vol(w, cov)
    return float((w @ vol) / pv) if pv > 1e-12 else 1.0


def effective_n(w: np.ndarray) -> float:
    """Nombre effectif de lignes (inverse de Herfindahl)."""
    w = np.asarray(w, float)
    h = float((w ** 2).sum())
    return 1.0 / h if h > 1e-12 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════

def check_feasibility(cons: Constraints) -> None:
    """
    P1.1 — LE correctif le plus important du prototype.
    On résout un problème de faisabilité par programmation linéaire. Si aucun
    portefeuille ne satisfait les contraintes, on lève une erreur explicite
    au lieu de renvoyer silencieusement 1/n (qui violait les contraintes tout
    en étant affiché comme « portefeuille optimisé »).
    """
    n = cons.n
    if np.any(cons.min_w > cons.max_w + 1e-12):
        bad = np.where(cons.min_w > cons.max_w)[0].tolist()
        raise InfeasibleConstraints(
            f"Poids minimum supérieur au maximum pour la ou les positions {bad}."
        )
    if cons.min_w.sum() > 1.0 + 1e-9:
        raise InfeasibleConstraints(
            f"La somme des poids minimum vaut {cons.min_w.sum():.1%} : "
            f"elle ne peut pas dépasser 100 %. Baissez les planchers."
        )
    if cons.max_w.sum() < 1.0 - 1e-9:
        raise InfeasibleConstraints(
            f"La somme des poids maximum vaut {cons.max_w.sum():.1%} : "
            f"elle doit atteindre au moins 100 %. Relevez les plafonds "
            f"ou ajoutez des actifs."
        )
    A_ub, b_ub = cons.group_matrix()
    res = linprog(
        c=np.zeros(n),
        A_ub=A_ub if A_ub.shape[0] else None,
        b_ub=b_ub if A_ub.shape[0] else None,
        A_eq=np.ones((1, n)), b_eq=np.array([1.0]),
        bounds=cons.bounds(), method="highs",
    )
    if not res.success:
        raise InfeasibleConstraints(
            "Les contraintes par actif et par classe d'actifs sont "
            "incompatibles entre elles : aucun portefeuille ne les satisfait "
            "toutes. Relâchez une borne de groupe ou un plafond individuel."
        )


def feasible_return_range(mu: np.ndarray, cons: Constraints) -> tuple[float, float]:
    """
    P1.6 — bornes de la frontière RÉELLEMENT atteignables sous contraintes.
    Le prototype prenait hi = max(mu) * 0.98, infaisable dès qu'un plafond
    limitait l'actif le plus performant : les points échouaient en silence.
    """
    n = cons.n
    A_ub, b_ub = cons.group_matrix()
    kw = dict(
        A_ub=A_ub if A_ub.shape[0] else None,
        b_ub=b_ub if A_ub.shape[0] else None,
        A_eq=np.ones((1, n)), b_eq=np.array([1.0]),
        bounds=cons.bounds(), method="highs",
    )
    lo_res = linprog(c=mu, **kw)
    hi_res = linprog(c=-mu, **kw)
    if not (lo_res.success and hi_res.success):
        raise InfeasibleConstraints("Bornes de rendement non calculables.")
    return float(mu @ lo_res.x), float(mu @ hi_res.x)


def _solve(fun, w0, cons: Constraints, floor: float = 0.0, **opts):
    return minimize(fun, w0, method="SLSQP", bounds=cons.bounds(floor),
                    constraints=cons.scipy_constraints(),
                    options={"ftol": 1e-12, "maxiter": 800, **opts})


def _start_points(cons: Constraints, n_starts: int, rng) -> list[np.ndarray]:
    """Points de départ admissibles : équipondéré + tirages de Dirichlet projetés."""
    pts = [np.clip(np.ones(cons.n) / cons.n, cons.min_w, cons.max_w)]
    pts[0] = pts[0] / pts[0].sum()
    for _ in range(max(n_starts - 1, 0)):
        w = rng.dirichlet(np.ones(cons.n))
        w = np.clip(w, cons.min_w, cons.max_w)
        s = w.sum()
        pts.append(w / s if s > 1e-9 else pts[0])
    return pts


def solve_min_variance(cov: np.ndarray, cons: Constraints) -> np.ndarray:
    check_feasibility(cons)
    rng = np.random.default_rng(SEED)
    best, best_w = np.inf, None
    for w0 in _start_points(cons, 5, rng):
        r = _solve(lambda w: float(w @ cov @ w), w0, cons)
        if r.success and r.fun < best:
            best, best_w = r.fun, r.x
    if best_w is None:
        raise OptimizationError(
            "Le solveur n'a pas convergé pour le portefeuille de variance "
            "minimale. Vérifiez que la matrice de covariance est bien définie "
            "(actifs dupliqués ou historique trop court ?)."
        )
    return _clean(best_w, cons)


def solve_max_sharpe(mu: np.ndarray, cov: np.ndarray, cons: Constraints,
                     rf: float, n_starts: int = 40) -> np.ndarray:
    check_feasibility(cons)
    if np.allclose(mu, 0.0):
        raise OptimizationError(
            "Le ratio de Sharpe ne peut pas être maximisé sans estimation de "
            "rendement espéré. Choisissez un estimateur de μ (historique ou "
            "Black-Litterman), ou utilisez Min Variance / Risk Parity."
        )
    rng = np.random.default_rng(SEED)

    def neg_sharpe(w):
        v = np.sqrt(max(float(w @ cov @ w), 1e-18))
        return -(float(w @ mu) - rf) / v

    best, best_w = -np.inf, None
    for w0 in _start_points(cons, n_starts, rng):
        r = _solve(neg_sharpe, w0, cons)
        if r.success and -r.fun > best:
            best, best_w = -r.fun, r.x
    if best_w is None:
        raise OptimizationError("Le solveur n'a pas convergé pour le portefeuille tangent.")
    return _clean(best_w, cons)


def solve_risk_parity(cov: np.ndarray, cons: Constraints,
                      n_starts: int = 20) -> np.ndarray:
    """Égalisation des contributions au risque. Ne dépend pas de μ → robuste."""
    check_feasibility(cons)
    rng = np.random.default_rng(SEED)

    def obj(w):
        pv = np.sqrt(max(float(w @ cov @ w), 1e-18))
        rc = w * ((cov @ w) / pv)
        return float(((rc - rc.mean()) ** 2).sum()) * 1e4

    best, best_w = np.inf, None
    for w0 in _start_points(cons, n_starts, rng):
        r = _solve(obj, w0, cons, floor=1e-4)
        if r.success and r.fun < best:
            best, best_w = r.fun, r.x
    if best_w is None:
        raise OptimizationError("Le solveur n'a pas convergé pour la parité des risques.")
    return _clean(best_w, cons)


def solve_target_return(mu: np.ndarray, cov: np.ndarray, cons: Constraints,
                        target: float) -> Optional[np.ndarray]:
    """Variance minimale sous contrainte de rendement — brique de la frontière."""
    extra = cons.scipy_constraints() + [
        {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t}
    ]
    rng = np.random.default_rng(SEED)
    for w0 in _start_points(cons, 3, rng):
        r = minimize(lambda w: float(w @ cov @ w), w0, method="SLSQP",
                     bounds=cons.bounds(), constraints=extra,
                     options={"ftol": 1e-12, "maxiter": 500})
        if r.success:
            return _clean(r.x, cons)
    return None


def solve_min_cvar(returns: pd.DataFrame, cons: Constraints,
                   alpha: float = 0.95,
                   target_return: Optional[float] = None,
                   mu: Optional[np.ndarray] = None) -> np.ndarray:
    """
    P4.2 — Minimisation du CVaR (Rockafellar & Uryasev, 2000), formulée en
    programme linéaire. Variables : [w (n), ζ (1), u (T)].

        min  ζ + 1/(T(1-α)) Σ uₜ
        s.c. uₜ ≥ −rₜ·w − ζ ,  uₜ ≥ 0 ,  Σw = 1 ,  bornes , contraintes de groupe

    Le CVaR mesure la perte moyenne dans la queue : il capte le risque
    réellement redouté par un particulier, contrairement à la variance.
    """
    from scipy import sparse

    check_feasibility(cons)
    R = returns.values
    T, n = R.shape
    if T < 60:
        raise DataError("Historique insuffisant pour une optimisation CVaR (< 60 jours).")

    c = np.concatenate([np.zeros(n), [1.0], np.full(T, 1.0 / (T * (1.0 - alpha)))])

    # −R w − ζ − u ≤ 0
    rows = sparse.hstack([
        sparse.csr_matrix(-R),
        sparse.csr_matrix(-np.ones((T, 1))),
        -sparse.identity(T, format="csr"),
    ], format="csr")
    A_ub, b_ub = [rows], [np.zeros(T)]

    Ag, bg = cons.group_matrix()
    if Ag.shape[0]:
        A_ub.append(sparse.hstack(
            [sparse.csr_matrix(Ag), sparse.csr_matrix((Ag.shape[0], 1 + T))], format="csr"))
        b_ub.append(bg)
    if target_return is not None and mu is not None:
        A_ub.append(sparse.hstack(
            [sparse.csr_matrix(-mu.reshape(1, -1)),
             sparse.csr_matrix((1, 1 + T))], format="csr"))
        b_ub.append(np.array([-target_return]))

    A_eq = sparse.hstack(
        [sparse.csr_matrix(np.ones((1, n))), sparse.csr_matrix((1, 1 + T))], format="csr")
    bounds = cons.bounds() + [(None, None)] + [(0, None)] * T

    res = linprog(c, A_ub=sparse.vstack(A_ub, format="csr"),
                  b_ub=np.concatenate(b_ub),
                  A_eq=A_eq, b_eq=np.array([1.0]),
                  bounds=bounds, method="highs")
    if not res.success:
        raise OptimizationError(f"Optimisation CVaR échouée : {res.message}")
    return _clean(res.x[:n], cons)


def _clean(w: np.ndarray, cons: Constraints, tol: float = 1e-6) -> np.ndarray:
    """Nettoyage numérique : re-projection dans les bornes, somme exacte à 1."""
    w = np.asarray(w, float)
    w[np.abs(w) < tol] = 0.0
    w = np.clip(w, cons.min_w, cons.max_w)
    s = w.sum()
    if s <= 1e-9:
        raise OptimizationError("Poids dégénérés (somme nulle).")
    return w / s


def apply_cardinality(w: np.ndarray, k: int, resolve: Callable[[list[int]], np.ndarray],
                      cons: Constraints) -> np.ndarray:
    """
    P3.4 — cardinalité maximale par heuristique : on garde les k plus gros
    poids puis on ré-optimise sur ce sous-ensemble. (L'optimum exact est un
    problème mixte entier, hors périmètre ici — l'heuristique est signalée.)
    """
    if k is None or k >= cons.n or (w > 1e-6).sum() <= k:
        return w
    keep = sorted(np.argsort(-w)[:k].tolist())
    if np.any(cons.min_w[[i for i in range(cons.n) if i not in keep]] > 1e-9):
        raise InfeasibleConstraints(
            "La cardinalité maximale est incompatible avec un poids plancher "
            "imposé sur un actif exclu."
        )
    w_sub = resolve(keep)
    full = np.zeros(cons.n)
    full[keep] = w_sub
    return full


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, cons: Constraints,
                       n_points: int = 60) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Frontière efficiente entre les bornes réellement atteignables (P1.6).
    Renvoie (volatilités, rendements, nb de points infaisables) — ce dernier
    est affiché à l'utilisateur au lieu d'être avalé.
    """
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


def resampled_frontier_weights(returns: pd.DataFrame, cons: Constraints, rf: float,
                               spec: ModelSpec, objective: str = "max_sharpe",
                               n_boot: int = 60, seed: int = SEED
                               ) -> tuple[np.ndarray, np.ndarray]:
    """
    P4.1 — Rééchantillonnage de Michaud. On rejoue l'optimisation sur des
    tirages bootstrap de l'historique, puis on moyenne les poids.

    C'est le remède le plus direct au problème central de Markowitz : sans
    cela, l'optimiseur concentre le portefeuille sur l'actif qui a le mieux
    performé par hasard. La dispersion des poids obtenue est aussi la mesure
    honnête de l'incertitude à afficher à l'utilisateur.
    """
    rng = np.random.default_rng(seed)
    T = len(returns)
    all_w = []
    for _ in range(n_boot):
        idx = rng.integers(0, T, T)                       # bootstrap i.i.d.
        sample = returns.iloc[idx]
        try:
            mu_b, cov_b, _ = estimate_moments(sample, spec, rf)
            if objective == "max_sharpe":
                w = solve_max_sharpe(mu_b, cov_b, cons, rf, n_starts=8)
            elif objective == "min_variance":
                w = solve_min_variance(cov_b, cons)
            else:
                w = solve_risk_parity(cov_b, cons, n_starts=5)
            all_w.append(w)
        except PortfolioLabError:
            continue
    if not all_w:
        raise OptimizationError("Rééchantillonnage impossible : aucun tirage n'a convergé.")
    W = np.array(all_w)
    mean_w = _clean(W.mean(axis=0), cons)
    return mean_w, W


def random_portfolio_cloud(mu: np.ndarray, cov: np.ndarray, cons: Constraints,
                           rf: float = 0.0, n: int = N_MC_CLOUD, seed: int = SEED
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    P1.10 — nuage de portefeuilles aléatoires ENTIÈREMENT VECTORISÉ.
    La boucle Python du prototype (8 000 itérations) devient trois produits
    matriciels. Les tirages hors bornes sont rejetés plutôt que déformés.
    """
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(cons.n), size=n)
    ok = np.all((W >= cons.min_w - 1e-9) & (W <= cons.max_w + 1e-9), axis=1)
    if ok.sum() < 50:            # contraintes serrées : on projette au lieu de rejeter
        W = np.clip(W, cons.min_w, cons.max_w)
        W = W / W.sum(axis=1, keepdims=True)
    else:
        W = W[ok]
    rets = W @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    with np.errstate(divide="ignore", invalid="ignore"):
        shs = np.where(vols > 1e-9, (rets - rf) / vols, 0.0)
    return rets, vols, shs


# ═══════════════════════════════════════════════════════════════════════════
# TRAJECTOIRE DE PORTEFEUILLE — DÉFINITION UNIQUE  (P1.2)
# ═══════════════════════════════════════════════════════════════════════════
# Le prototype mélangeait deux portefeuilles différents dans le même bandeau :
#   • cum_series  → poids constants, donc rééquilibrage QUOTIDIEN implicite
#                   (irréalisable, et gonflé par le « volatility pumping ») ;
#   • total_return → buy-and-hold.
# Ici, une seule fonction produit la NAV, le drawdown, le turnover et les
# frais, pour une politique de rééquilibrage explicitement choisie.

REBALANCE_CHOICES = {
    "Aucun (buy & hold)": "none",
    "Mensuel": "M",
    "Trimestriel": "Q",
    "Annuel": "A",
    "Quotidien (théorique)": "D",
}


def rebalance_flags(index: pd.DatetimeIndex, policy: str) -> np.ndarray:
    """Vrai sur les dates où l'on ramène les poids à la cible."""
    n = len(index)
    if policy == "none":
        return np.zeros(n, dtype=bool)
    if policy == "D":
        return np.ones(n, dtype=bool)
    freq = {"M": "M", "Q": "Q", "A": "Y"}[policy]
    per = pd.PeriodIndex(index, freq=freq)
    flags = np.zeros(n, dtype=bool)
    flags[:-1] = per[:-1].asi8 != per[1:].asi8    # dernier jour de chaque période
    return flags


def portfolio_path(returns: pd.DataFrame, w_target: np.ndarray, *,
                   rebalance: str = "M", costs: Optional[CostModel] = None,
                   charge_initial: bool = True) -> dict:
    """
    Simule une trajectoire nette de frais. Renvoie NAV, rendements nets,
    turnover cumulé, frais payés et dérive des poids.

    Convention de coût : le montant échangé est Σ|Δw| et le coût unitaire est
    le coût aller simple (courtage + demi-spread). Le TER est prélevé au
    prorata quotidien sur les poids réellement détenus (P2.4).
    """
    costs = costs or CostModel()
    R = returns.values
    idx = returns.index
    n = R.shape[1]
    w_target = np.asarray(w_target, float)
    if w_target.size != n:
        raise ValueError("Dimension des poids incohérente avec les rendements.")

    c = costs.one_way
    ter = costs.ter_vector(n)
    flags = rebalance_flags(idx, rebalance)

    # La NAV est ancrée à 1,0 AVANT la première séance : le rendement total,
    # le CAGR et le drawdown partagent ainsi exactement la même base, et les
    # frais d'entrée sont bien comptés dans la performance affichée.
    nav = 1.0
    if charge_initial:
        nav *= (1.0 - np.abs(w_target).sum() * c)
        fees_initial = np.abs(w_target).sum() * c
    else:
        fees_initial = 0.0

    w = w_target.copy()
    navs = np.empty(len(idx))
    turnover = np.zeros(len(idx))
    fees = fees_initial
    w_hist = np.empty((len(idx), n))

    for t in range(len(idx)):
        r = R[t]
        r = np.where(np.isnan(r), 0.0, r)
        gross = 1.0 + float(w @ r)
        gross = max(gross, 1e-12)
        nav *= gross
        w = w * (1.0 + r) / gross                       # dérive naturelle
        drag = float(w @ ter) / PERIODS                 # frais courants
        nav *= (1.0 - drag)
        fees += drag
        if flags[t]:
            traded = float(np.abs(w_target - w).sum())
            cost = traded * c
            nav *= (1.0 - cost)
            fees += cost
            turnover[t] = traded / 2.0                  # turnover unidirectionnel
            w = w_target.copy()
        navs[t] = nav
        w_hist[t] = w

    anchor = idx[0] - pd.Timedelta(days=1)
    nav_s = pd.Series(np.concatenate([[1.0], navs]),
                      index=pd.DatetimeIndex([anchor]).append(idx), name="nav")
    net_rets = nav_s.pct_change().dropna()
    return {
        "nav": nav_s,
        "returns": net_rets,
        "turnover": pd.Series(turnover, index=idx),
        "annual_turnover": float(turnover.sum() / max(len(idx) / PERIODS, 1e-9)),
        "total_fees": float(fees),
        "weights": pd.DataFrame(w_hist, index=idx, columns=returns.columns),
        "final_weights": w.copy(),
    }


def portfolio_stats(returns: pd.DataFrame, w: np.ndarray, rf: float, *,
                    rebalance: str = "M", costs: Optional[CostModel] = None,
                    bench_rets: Optional[pd.Series] = None) -> dict:
    path = portfolio_path(returns, w, rebalance=rebalance, costs=costs)
    stats = full_stats(path["nav"], path["returns"], rf, bench_rets)
    stats["annual_turnover"] = path["annual_turnover"]
    stats["total_fees"] = path["total_fees"]
    stats["nav"] = path["nav"]
    stats["net_returns"] = path["returns"]
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST WALK-FORWARD  (P2.2) — la brique qui manquait le plus
# ═══════════════════════════════════════════════════════════════════════════
# Le prototype optimisait et évaluait sur la MÊME période : 100 % in-sample.
# Tout Sharpe affiché était donc un artefact. Ici, à chaque date de
# rééquilibrage, on n'utilise que les données antérieures, puis on mesure la
# performance sur la période suivante — jamais vue par l'optimiseur.

def _period_end_dates(index: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    f = {"M": "M", "Q": "Q", "A": "Y"}.get(freq, "Q")
    per = pd.PeriodIndex(index, freq=f)
    ends = []
    for i in range(len(index) - 1):
        if per[i] != per[i + 1]:
            ends.append(index[i])
    return ends


def walk_forward_backtest(returns: pd.DataFrame, *, method: str, cons: Constraints,
                          rf: float, spec: ModelSpec,
                          lookback_years: float = 3.0,
                          reb_freq: str = "Q",
                          costs: Optional[CostModel] = None,
                          progress: Optional[Callable[[float, str], None]] = None
                          ) -> dict:
    """
    method ∈ {max_sharpe, min_variance, risk_parity, min_cvar, equal_weight}

    Renvoie la NAV hors échantillon, l'historique des poids, le turnover
    réalisé et le nombre de ré-optimisations en échec (signalé, pas masqué).
    """
    costs = costs or CostModel()
    lookback = int(lookback_years * PERIODS)
    idx = returns.index
    if len(idx) < lookback + 60:
        raise DataError(
            f"Historique trop court pour un backtest hors échantillon : "
            f"{len(idx)} jours disponibles, {lookback + 60} nécessaires "
            f"(fenêtre d'estimation {lookback_years:.0f} ans + période de test). "
            f"Allongez la période ou réduisez la fenêtre d'estimation."
        )

    reb_dates = [d for d in _period_end_dates(idx, reb_freq) if idx.get_loc(d) >= lookback]
    if not reb_dates:
        raise DataError("Aucune date de rééquilibrage exploitable sur la période.")

    c = costs.one_way
    ter = costs.ter_vector(returns.shape[1])
    R = returns.values

    nav, w = 1.0, np.zeros(returns.shape[1])
    navs, dates, w_rows, w_dates = [], [], [], []
    turnover_total, n_failed, first = 0.0, 0, True
    first_point = True          # ancrage de la NAV a 1.0 (meme base partout)

    bounds_idx = [idx.get_loc(d) for d in reb_dates] + [len(idx) - 1]

    for k in range(len(reb_dates)):
        d_pos = bounds_idx[k]
        end_pos = bounds_idx[k + 1]
        window = returns.iloc[max(0, d_pos - lookback):d_pos]

        # ── Estimation sur données passées uniquement ────────────────────
        try:
            if method == "equal_weight":
                w_new = _clean(np.ones(cons.n) / cons.n, cons)
            else:
                mu_w, cov_w, _ = estimate_moments(window, spec, rf)
                if method == "max_sharpe":
                    w_new = solve_max_sharpe(mu_w, cov_w, cons, rf, n_starts=12)
                elif method == "min_variance":
                    w_new = solve_min_variance(cov_w, cons)
                elif method == "risk_parity":
                    w_new = solve_risk_parity(cov_w, cons, n_starts=8)
                elif method == "min_cvar":
                    w_new = solve_min_cvar(window, cons, 0.95)
                else:
                    raise ValueError(f"Méthode inconnue : {method}")
        except PortfolioLabError:
            n_failed += 1
            w_new = w.copy() if not first else _clean(np.ones(cons.n) / cons.n, cons)

        traded = float(np.abs(w_new - w).sum()) if not first else float(np.abs(w_new).sum())
        nav *= (1.0 - traded * c)
        turnover_total += traded / 2.0
        if first_point:
            navs.append(1.0); dates.append(idx[d_pos]); first_point = False
        w = w_new.copy()
        first = False
        w_rows.append(w.copy()); w_dates.append(idx[d_pos])

        # ── Détention hors échantillon jusqu'au rééquilibrage suivant ────
        for t in range(d_pos + 1, end_pos + 1):
            r = np.nan_to_num(R[t])
            gross = max(1.0 + float(w @ r), 1e-12)
            nav *= gross
            w = w * (1.0 + r) / gross
            nav *= (1.0 - float(w @ ter) / PERIODS)
            navs.append(nav); dates.append(idx[t])

        if progress:
            progress((k + 1) / len(reb_dates), f"Ré-optimisation {k + 1}/{len(reb_dates)}")

    nav_s = pd.Series(navs, index=pd.DatetimeIndex(dates), name=method)
    nav_s = nav_s[~nav_s.index.duplicated(keep="last")]
    net = nav_s.pct_change().dropna()
    years = max(len(nav_s) / PERIODS, 1e-9)

    return {
        "nav": nav_s,
        "returns": net,
        "weights": pd.DataFrame(w_rows, index=pd.DatetimeIndex(w_dates),
                                columns=returns.columns),
        "annual_turnover": turnover_total / years,
        "n_rebalances": len(reb_dates),
        "n_failed": n_failed,
        "method": method,
    }


def buy_and_hold_nav(returns: pd.DataFrame, w: np.ndarray,
                     costs: Optional[CostModel] = None) -> pd.Series:
    return portfolio_path(returns, w, rebalance="none", costs=costs)["nav"]


def sixty_forty_weights(assets: Sequence[str], classes: dict[str, str]) -> Optional[np.ndarray]:
    """
    Construit un 60/40 à partir des actifs du portefeuille lorsque c'est
    possible : 60 % réparti sur les actions, 40 % sur les obligations.
    Renvoie None si l'une des deux poches est absente.
    """
    eq = [i for i, a in enumerate(assets) if classes.get(a) == "Equity"]
    bd = [i for i, a in enumerate(assets) if classes.get(a) == "Bond"]
    if not eq or not bd:
        return None
    w = np.zeros(len(assets))
    w[eq] = 0.60 / len(eq)
    w[bd] = 0.40 / len(bd)
    return w


def oos_verdict(strategy: dict, naive: dict, rf: float) -> dict:
    """
    Verdict explicite exigé par la littérature (DeMiguel, Garlappi & Uppal,
    2009) : l'optimisation bat-elle le simple 1/N hors échantillon ?
    L'utilisateur doit voir la réponse, pas seulement les jolis graphiques.
    """
    s = full_stats(strategy["nav"], strategy["returns"], rf)
    n = full_stats(naive["nav"], naive["returns"], rf)
    diff = s["sharpe"] - n["sharpe"]
    # Test de Jobson-Korkie / Memmel sur l'écart de Sharpe
    df = pd.concat([strategy["returns"], naive["returns"]], axis=1).dropna()
    p_value = float("nan")
    if len(df) > 60:
        T = len(df)
        # Rendements excédentaires, pour rester cohérent avec les Sharpe
        # affichés ailleurs (qui déduisent le taux sans risque).
        r1 = df.iloc[:, 0].values - rf / PERIODS
        r2 = df.iloc[:, 1].values - rf / PERIODS
        s1 = r1.mean() / r1.std(ddof=1) if r1.std(ddof=1) > 0 else 0.0
        s2 = r2.mean() / r2.std(ddof=1) if r2.std(ddof=1) > 0 else 0.0
        rho = float(np.corrcoef(r1, r2)[0, 1])
        var = (2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / T
        if var > 0:
            z = (s1 - s2) / np.sqrt(var)
            from scipy.stats import norm
            p_value = float(2 * (1 - norm.cdf(abs(z))))
    return {
        "strategy": s, "naive": n, "sharpe_gap": diff, "p_value": p_value,
        "beats_naive": bool(diff > 0),
        "significant": bool(p_value == p_value and p_value < 0.05),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PROJECTION FORWARD  (P3.1) — ce que l'utilisateur veut vraiment savoir
# ═══════════════════════════════════════════════════════════════════════════
# Le Monte-Carlo du prototype tirait des POIDS au hasard : décoratif, sans
# information. Ici on simule l'AVENIR du portefeuille choisi, avec versements
# programmés, par bootstrap par blocs (préserve l'autocorrélation et les
# regroupements de volatilité, contrairement au tirage i.i.d.).

def block_bootstrap_paths(monthly_returns: np.ndarray, horizon_months: int,
                          n_sims: int = 5000, block: int = 6,
                          seed: int = SEED) -> np.ndarray:
    """Matrice (n_sims × horizon_months) de rendements mensuels rééchantillonnés."""
    r = np.asarray(monthly_returns, float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 24:
        raise DataError(
            f"Historique trop court pour une projection fiable : {T} mois "
            f"disponibles, 24 minimum. Allongez la période d'analyse."
        )
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon_months / block))
    starts = rng.integers(0, T, size=(n_sims, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % T      # bootstrap circulaire
    return r[idx.reshape(n_sims, -1)[:, :horizon_months]]


def simulate_wealth(monthly_returns: np.ndarray, *, initial: float,
                    monthly_contribution: float, horizon_years: float,
                    n_sims: int = 5000, block: int = 6,
                    inflation: float = 0.02, tax_rate: float = 0.0,
                    fee_bps_annual: float = 0.0, seed: int = SEED) -> dict:
    """
    Distribution du capital final avec versements programmés (DCA).

    Réponses fournies : quelle fourchette de capital, quelle probabilité
    d'atteindre un objectif, quelle perte maximale traversée en chemin, et
    quel TRI — sachant qu'avec des versements, le TRI diffère du rendement
    géométrique du portefeuille (risque de séquence).
    """
    H = int(round(horizon_years * MONTHS))
    if H < 1:
        raise ValueError("Horizon trop court.")
    paths = block_bootstrap_paths(monthly_returns, H, n_sims, block, seed)
    paths = paths - (fee_bps_annual / 1e4) / MONTHS       # frais non déjà déduits

    n = paths.shape[0]
    wealth = np.full(n, float(initial))
    peak = wealth.copy()
    max_dd = np.zeros(n)
    invested = float(initial)
    traj = np.empty((n, H + 1))
    traj[:, 0] = wealth

    for m in range(H):
        wealth = wealth * (1.0 + paths[:, m]) + monthly_contribution
        invested += monthly_contribution
        peak = np.maximum(peak, wealth)
        max_dd = np.minimum(max_dd, wealth / np.maximum(peak, 1e-9) - 1.0)
        traj[:, m + 1] = wealth

    gains = np.maximum(wealth - invested, 0.0)
    after_tax = wealth - gains * tax_rate                 # fiscalité à la sortie
    real = after_tax / ((1.0 + inflation) ** horizon_years)

    cash = np.concatenate([[-initial], np.full(H, -monthly_contribution)])
    irr_med = money_weighted_return(cash, float(np.median(wealth)))

    qs = [0.05, 0.25, 0.50, 0.75, 0.95]
    return {
        "terminal": wealth, "after_tax": after_tax, "real": real,
        "trajectories": traj, "invested": invested,
        "max_drawdown": max_dd,
        "quantiles": {q: float(np.quantile(wealth, q)) for q in qs},
        "quantiles_real": {q: float(np.quantile(real, q)) for q in qs},
        "prob_loss": float((wealth < invested).mean()),
        "median_irr": irr_med,
        "horizon_months": H,
    }


def money_weighted_return(cashflows_monthly: np.ndarray, terminal: float) -> float:
    """TRI mensuel annualisé, par bissection sur la VAN."""
    # Convention : cf[0..H] contient les versements, le capital final est
    # valorisé au même instant que le dernier versement (t = H).
    cf = np.asarray(cashflows_monthly, float).copy()
    cf[-1] += terminal

    def npv(rate):
        t = np.arange(len(cf))
        return float(np.sum(cf / (1.0 + rate) ** t))

    lo, hi = -0.99 / MONTHS, 1.0
    if npv(lo) * npv(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return float((1.0 + 0.5 * (lo + hi)) ** MONTHS - 1.0)


def prob_reach_goal(terminal: np.ndarray, goal: float) -> float:
    return float((np.asarray(terminal) >= goal).mean())


def to_monthly(returns_series: pd.Series, min_days: int = 15) -> np.ndarray:
    """
    Agrège des rendements quotidiens en mensuels (composition exacte).

    Les mois incomplets de début et de fin de période sont écartés : conservés,
    ils injectaient dans le bootstrap des « mois » de quelques séances,
    sous-estimant à la fois le rendement et la volatilité mensuels.
    """
    grouped = (1.0 + returns_series).resample("ME")
    m = grouped.prod() - 1.0
    counts = grouped.count()
    m = m[counts >= min_days]
    return m.dropna().values


# ═══════════════════════════════════════════════════════════════════════════
# PLAN D'EXÉCUTION  (P3.3)
# ═══════════════════════════════════════════════════════════════════════════
# Le prototype affichait « 0,0347 action » : inexécutable chez la plupart des
# courtiers. On arrondit aux quantités réalisables, on montre le cash résiduel
# et l'écart aux poids cibles.

def build_execution_plan(assets: Sequence[str], w_target: np.ndarray, budget: float,
                         prices_base: dict[str, float], *,
                         lot_size: int = 1, allow_fractional: bool = False,
                         costs: Optional[CostModel] = None) -> pd.DataFrame:
    costs = costs or CostModel()
    rows = []
    invested = 0.0
    missing = []
    for i, a in enumerate(assets):
        target_amt = budget * float(w_target[i])
        px = prices_base.get(a)
        if not px or px <= 0 or not np.isfinite(px):
            missing.append(a)
            rows.append({"Actif": a, "Poids cible": w_target[i], "Montant cible": target_amt,
                         "Prix": np.nan, "Quantité": np.nan, "Montant investi": 0.0})
            continue
        if allow_fractional:
            qty = target_amt / px
        else:
            qty = np.floor(target_amt / (px * lot_size)) * lot_size
        amt = qty * px
        invested += amt
        rows.append({"Actif": a, "Poids cible": float(w_target[i]),
                     "Montant cible": target_amt, "Prix": float(px),
                     "Quantité": float(qty), "Montant investi": float(amt)})

    df = pd.DataFrame(rows)
    df["Poids réel"] = df["Montant investi"] / budget
    df["Écart (pts)"] = (df["Poids réel"] - df["Poids cible"]) * 100
    fees = invested * costs.one_way
    df.attrs["cash_residuel"] = float(budget - invested - fees)
    df.attrs["frais_entree"] = float(fees)
    df.attrs["taux_investi"] = float(invested / budget) if budget > 0 else 0.0
    df.attrs["ecart_max_pts"] = float(df["Écart (pts)"].abs().max()) if len(df) else 0.0
    df.attrs["prix_manquants"] = missing
    return df


def classify_asset(ticker: str, quote_type: str = "", exchange: str = "") -> str:
    """Classe d'actifs par heuristique — corrigible par l'utilisateur dans l'UI."""
    t = ticker.upper()
    if quote_type.upper() in ("CRYPTOCURRENCY",) or t.endswith("-USD") or t.endswith("-EUR"):
        return "Crypto"
    for cls, hints in CLASS_HINTS.items():
        if any(t == h or t.startswith(h + ".") for h in hints):
            return cls
    if quote_type.upper() in ("CURRENCY",):
        return "Cash"
    if quote_type.upper() in ("EQUITY", "ETF", "MUTUALFUND", "INDEX"):
        return "Equity"
    return "Other"


def pea_eligible(exchange: str, currency: str, quote_type: str) -> bool:
    """
    P3 — indication PEA (France). Heuristique : place de cotation UE/EEE et
    titre de capital. À vérifier auprès du courtier : l'éligibilité dépend du
    siège de l'émetteur, pas seulement de la place de cotation.
    """
    eu_ex = {"PAR", "AMS", "BRU", "LIS", "MIL", "MCE", "GER", "FRA", "XETRA",
             "EBS", "STO", "CPH", "HEL", "OSL", "DUB", "VIE", "WSE"}
    return (exchange or "").upper() in eu_ex and quote_type.upper() in ("EQUITY", "ETF")


# ═══════════════════════════════════════════════════════════════════════════
# EXPOSITION FACTORIELLE  (P4.3)
# ═══════════════════════════════════════════════════════════════════════════

def factor_regression(port_rets: pd.Series, factor_rets: pd.DataFrame,
                      rf: float = 0.0) -> pd.DataFrame:
    """
    Régression OLS des rendements du portefeuille sur des proxies factoriels.
    Objectif : savoir si le « portefeuille optimal » n'est pas simplement un
    pari momentum déguisé, et si les actifs sont réellement décorrélés.
    """
    df = pd.concat([port_rets.rename("port"), factor_rets], axis=1).dropna()
    if len(df) < 60:
        raise DataError("Historique commun insuffisant pour la régression factorielle.")
    y = df["port"].values - rf / PERIODS
    X = df.drop(columns=["port"]).values - rf / PERIODS
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = max(len(y) - X1.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov_b = s2 * np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.maximum(np.diag(cov_b), 0.0))
    tstat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    names = ["Alpha (ann.)"] + list(df.columns[1:])
    vals = [beta[0] * PERIODS] + list(beta[1:])
    out = pd.DataFrame({"Facteur": names, "Coefficient": vals,
                        "t-stat": tstat, "Significatif": np.abs(tstat) > 1.96})
    out.attrs["r2"] = r2
    return out


# ═══════════════════════════════════════════════════════════════════════════
# MOTEUR DE SCÉNARIOS
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario(prices_sc: pd.DataFrame, weights: dict[str, np.ndarray],
                 assets: Sequence[str], name: str, meta: dict, *,
                 rebalance: str = "none", costs: Optional[CostModel] = None) -> dict:
    """
    P1.2 appliqué aux scénarios : rendement total, drawdown et volatilité
    proviennent tous de LA MÊME trajectoire (portfolio_path).
    """
    if prices_sc is None or prices_sc.empty:
        return {}
    available = [a for a in assets
                 if a in prices_sc.columns and prices_sc[a].notna().sum() >= 5]
    if not available:
        return {}

    p = prices_sc[available].ffill().dropna(how="any")   # P1.9 : plus de bfill
    if len(p) < 5:
        return {}
    rets = simple_returns(p)
    if rets.empty:
        return {}

    idx = [list(assets).index(a) for a in available]
    results = {}
    for pname, w_full in weights.items():
        w = np.asarray(w_full, float)[idx]
        s = w.sum()
        if s < 1e-9:
            continue
        w = w / s                                        # re-normalisation
        path = portfolio_path(rets, w, rebalance=rebalance, costs=costs,
                              charge_initial=False)
        results[pname] = {
            "total_return": float(path["nav"].iloc[-1] - 1.0),
            "max_drawdown": max_drawdown(path["nav"]),
            "volatility": ann_vol_from_series(path["returns"]),
            "cvar95": historical_cvar(path["returns"], 0.95),
            "cum_series": path["nav"] * 100.0,
        }

    asset_stats = {}
    for a in available:
        s = p[a].dropna()
        if len(s) < 2:
            continue
        asset_stats[a] = {"total_return": float(s.iloc[-1] / s.iloc[0] - 1.0),
                          "max_drawdown": max_drawdown(s)}

    return {"portfolios": results, "assets": asset_stats, "prices": p,
            "available": available,
            "missing": [a for a in assets if a not in available],
            "name": name, "meta": meta,
            "renormalized": len(available) < len(assets)}


# ═══════════════════════════════════════════════════════════════════════════
# COUCHE DONNÉES
# ═══════════════════════════════════════════════════════════════════════════
# Avertissement assumé : yfinance est une API non officielle, sujette au
# rate-limiting et aux ruptures. Pour un usage professionnel, remplacer
# fetch_raw_prices() par un fournisseur contractuel (EOD Historical,
# Refinitiv, Bloomberg). Le reste du code est indépendant de la source.

FX_FALLBACK = {"EUR": 1.08, "GBP": 1.27, "CHF": 1.12, "JPY": 0.0067,
               "CAD": 0.74, "AUD": 0.65, "USD": 1.00}


def _require_yf() -> None:
    if not HAS_YF:
        raise DataError("yfinance n'est pas installé : pip install yfinance")


@cache_data(show_spinner=False, ttl=3600)
def fetch_raw_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Clôtures ajustées, dans la devise de cotation de chaque titre."""
    _require_yf()
    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        raise DataError(
            f"Aucune donnée renvoyée pour {', '.join(tickers)} entre {start} et {end}. "
            f"Vérifiez les symboles et la période."
        )
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        raw = raw["Close"] if "Close" in lvl0 else raw[lvl0[0]]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    raw = raw.ffill(limit=5)                        # P1.9 : ffill seul, jamais bfill
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    return raw.dropna(how="all").sort_index()


@cache_data(show_spinner=False, ttl=3600)
def fetch_fx_series(currencies: tuple, base: str, start: str, end: str) -> pd.DataFrame:
    """
    P1.3 — séries FX quotidiennes pour convertir chaque devise vers la base.
    Convention : rate[X] = nombre d'unités de BASE pour 1 unité de X.
    Échoue explicitement si une paire est introuvable.
    """
    _require_yf()
    needed = {c for c in currencies if c}
    usd_per: dict[str, pd.Series] = {}

    for c in needed | {base}:
        if c == "USD":
            continue
        s = None
        for sym, invert in ((f"{c}USD=X", False), (f"USD{c}=X", True)):
            try:
                d = yf.download(sym, start=start, end=end, progress=False,
                                auto_adjust=True)
                if d is not None and not d.empty:
                    col = d["Close"]
                    if isinstance(col, pd.DataFrame):
                        col = col.iloc[:, 0]
                    col = col.dropna()
                    if len(col) > 5:
                        s = (1.0 / col) if invert else col
                        break
            except Exception:
                continue
        if s is None:
            raise FXError(
                f"Taux de change {c}/USD introuvable sur la période demandée. "
                f"Les rendements ne peuvent pas être convertis correctement en "
                f"{base}. Retirez l'actif concerné, changez de devise de base, "
                f"ou activez le repli statique (résultats alors approximatifs)."
            )
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        usd_per[c] = s

    if not usd_per:
        # Tout est déjà en USD et la base est l'USD : aucune conversion utile.
        return pd.DataFrame()
    base_usd = (pd.Series(1.0, index=usd_per[next(iter(usd_per))].index)
                if base == "USD" else usd_per[base])
    out = {}
    for c in needed:
        cs = pd.Series(1.0, index=base_usd.index) if c == "USD" else usd_per[c]
        df = pd.concat([cs.rename("x"), base_usd.rename("b")], axis=1).ffill().dropna()
        out[c] = df["x"] / df["b"]
    return pd.DataFrame(out).sort_index()


def convert_to_base(prices: pd.DataFrame, ccy_map: dict[str, str], base: str,
                    fx: Optional[pd.DataFrame], allow_static: bool = False
                    ) -> tuple[pd.DataFrame, dict]:
    """
    P1.3 — LA correction : conversion AVANT tout calcul de rendement.
    Sans elle, un portefeuille AAPL (USD) + AIR.PA (EUR) produit une matrice de
    covariance fausse, car on mélange des rendements exprimés dans deux unités.
    """
    foreign = {a: c for a, c in ccy_map.items() if c and c != base}
    info = {"converted": list(foreign), "method": "none", "base": base}
    if not foreign:
        return prices.copy(), info

    out = prices.copy()
    if fx is not None and not fx.empty:
        aligned = fx.reindex(out.index).ffill().bfill()
        missing = [c for c in set(foreign.values()) if c not in aligned.columns
                   or aligned[c].isna().all()]
        if missing and not allow_static:
            raise FXError(f"Séries FX manquantes pour : {', '.join(sorted(missing))}.")
        for a, c in foreign.items():
            if a not in out.columns:
                continue
            if c in aligned.columns and not aligned[c].isna().all():
                out[a] = out[a] * aligned[c]
            elif allow_static:
                out[a] = out[a] * (FX_FALLBACK.get(c, 1.0) / FX_FALLBACK.get(base, 1.0))
        info["method"] = "series"
        return out, info

    if not allow_static:
        raise FXError(
            "Aucune série de change disponible alors que le portefeuille est "
            "multi-devises. Activez le repli statique pour continuer avec des "
            "résultats explicitement approximatifs."
        )
    for a, c in foreign.items():
        out[a] = out[a] * (FX_FALLBACK.get(c, 1.0) / FX_FALLBACK.get(base, 1.0))
    info["method"] = "static"
    return out, info


@cache_data(ttl=1800, show_spinner=False)
def validate_ticker(ticker: str, deep: bool = False) -> dict:
    """
    P1.11 — validation rapide via fast_info uniquement. L'appel à .info,
    lent et rate-limité, devient optionnel (bouton « détails »).
    """
    _require_yf()
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        price = getattr(fi, "last_price", None)
        ccy = (getattr(fi, "currency", None) or "USD").upper()
        exch = getattr(fi, "exchange", "") or ""
        if price is None or not np.isfinite(float(price)) or float(price) <= 0:
            return {"valid": False, "ticker": ticker, "name": ticker,
                    "error": "Aucune cotation renvoyée"}
        out = {"valid": True, "ticker": ticker, "name": ticker, "exchange": exch,
               "currency": ccy, "asset_type": "EQUITY", "price": float(price)}
        if deep:
            try:
                info = t.info
                out["name"] = info.get("longName") or info.get("shortName") or ticker
                out["asset_type"] = info.get("quoteType", "EQUITY")
                out["exchange"] = info.get("exchange", exch)
                out["currency"] = (info.get("currency") or ccy).upper()
                out["market_cap"] = info.get("marketCap")
            except Exception:
                pass
        return out
    except Exception as e:
        return {"valid": False, "ticker": ticker, "name": ticker, "error": str(e)[:120]}


def data_quality_report(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Contrôles absents du prototype : trous, prix figés, sauts aberrants,
    historique trop court. Une erreur de données produit un « optimum »
    parfaitement crédible et parfaitement faux.
    """
    rows = []
    rets = prices.pct_change()
    for c in prices.columns:
        s = prices[c]
        r = rets[c].dropna()
        gaps = int(s.isna().sum())
        stale = int((r.abs() < 1e-12).sum())
        jumps = int((r.abs() > 0.35).sum())
        flags = []
        if len(s.dropna()) < 250:
            flags.append("historique < 1 an")
        if gaps > len(s) * 0.05:
            flags.append("plus de 5 % de trous")
        if stale > len(r) * 0.15:
            flags.append("prix figé sur >15 % des séances")
        if jumps > 0:
            flags.append(f"{jumps} saut(s) > 35 %")
        if r.std() < 1e-9:
            flags.append("variance nulle")
        rows.append({
            "Actif": c,
            "Séances": int(s.notna().sum()),
            "Début": str(s.dropna().index.min().date()) if s.notna().any() else "—",
            "Trous": gaps, "Séances figées": stale, "Sauts > 35 %": jumps,
            "Alertes": " · ".join(flags) if flags else "✅ RAS",
        })
    return pd.DataFrame(rows)


def align_common_history(prices: pd.DataFrame, min_obs: int = 252
                         ) -> tuple[pd.DataFrame, list[str]]:
    """
    Tronque à l'historique commun. Le prototype gardait des colonnes de
    longueurs différentes, ce qui biaisait la covariance (paires estimées
    sur des périodes différentes).
    """
    kept = [c for c in prices.columns if prices[c].notna().sum() >= min_obs]
    dropped = [c for c in prices.columns if c not in kept]
    if len(kept) < 2:
        raise DataError(
            f"Moins de deux actifs disposent de {min_obs} séances d'historique. "
            f"Écartés : {', '.join(dropped) if dropped else '—'}."
        )
    sub = prices[kept].dropna(how="any")
    if len(sub) < min_obs:
        starts = {c: prices[c].dropna().index.min() for c in kept}
        late = max(starts, key=lambda k: starts[k])
        raise DataError(
            f"L'historique commun ne couvre que {len(sub)} séances. "
            f"L'actif le plus récent est {late} (depuis {starts[late].date()}). "
            f"Retirez-le ou avancez la date de début."
        )
    return sub, dropped


@cache_data(ttl=3600, show_spinner=False)
def fetch_market_caps(tickers: tuple) -> dict:
    """
    Capitalisations boursières pour le portefeuille d'équilibre de
    Black-Litterman. Appel lent (.info) : réservé à ce cas d'usage.
    """
    out = {}
    for t in tickers:
        v = validate_ticker(t, deep=True)
        mc = v.get("market_cap")
        if mc and mc > 0:
            out[t] = float(mc)
    return out


@cache_data(ttl=300, show_spinner=False)
def fetch_risk_free(currency: str) -> tuple[float, str]:
    """P3.2 — taux sans risque dérivé de la devise, avec source explicite."""
    cfg = CURRENCIES.get(currency, {})
    proxy = cfg.get("rf_proxy")
    if proxy and HAS_YF:
        try:
            d = yf.download(proxy, period="1mo", progress=False, auto_adjust=True)
            if d is not None and not d.empty:
                v = float(d["Close"].dropna().iloc[-1]) / 100.0
                if 0.0 <= v < 0.25:
                    return v, f"marché ({proxy})"
        except Exception:
            pass
    return float(cfg.get("rf_default", 0.02)), "hypothèse par défaut, à ajuster"


@cache_data(ttl=60, show_spinner=False)
def fetch_last_prices(tickers: tuple) -> dict:
    _require_yf()
    out = {}
    try:
        raw = yf.download(list(tickers), period="5d", progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return {}
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        close = close.ffill().iloc[-1]
        for t in tickers:
            if t in close.index and pd.notna(close[t]):
                out[str(t)] = float(close[t])
    except Exception:
        return out
    return out


# ═══════════════════════════════════════════════════════════════════════════
# JOURNAL D'AUDIT  (P4.5)
# ═══════════════════════════════════════════════════════════════════════════

def data_fingerprint(prices: pd.DataFrame) -> str:
    """Empreinte des données : permet de rejouer et de justifier une allocation."""
    h = hashlib.sha256()
    h.update(",".join(map(str, prices.columns)).encode())
    h.update(str(prices.index.min()).encode())
    h.update(str(prices.index.max()).encode())
    h.update(np.ascontiguousarray(prices.values, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def build_audit_record(params: dict, prices: pd.DataFrame, weights: dict,
                       stats: dict) -> dict:
    return {
        "run_id": hashlib.sha256(
            (str(datetime.now(timezone.utc)) + json.dumps(params, default=str)).encode()
        ).hexdigest()[:12],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_fingerprint": data_fingerprint(prices),
        "period": {"start": str(prices.index.min().date()),
                   "end": str(prices.index.max().date()),
                   "observations": int(len(prices))},
        "parameters": params,
        "weights": {k: [round(float(x), 6) for x in v] for k, v in weights.items()},
        "statistics": {k: {kk: (round(float(vv), 6) if isinstance(vv, (int, float, np.floating))
                                else str(vv))
                           for kk, vv in v.items() if not isinstance(vv, (pd.Series, pd.DataFrame))}
                       for k, v in stats.items()},
        "software": "Portfolio Lab",
        "disclaimer": "Simulation à but pédagogique. Ne constitue pas un conseil "
                      "en investissement personnalisé.",
    }


# ═══════════════════════════════════════════════════════════════════════════
# GRAPHIQUES  (identité visuelle du prototype conservée)
# ═══════════════════════════════════════════════════════════════════════════

def base_layout(title="", xtitle="", ytitle="", **kw) -> dict:
    return dict(
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        title=dict(text=title, font=dict(size=14, color=TEXT)),
        xaxis=dict(title=xtitle, gridcolor=GRID, zerolinecolor=GRID, color=MUTED,
                   title_font=dict(color=MUTED, size=11),
                   tickfont=dict(color=MUTED, size=10), linecolor=GRID),
        yaxis=dict(title=ytitle, gridcolor=GRID, zerolinecolor=GRID, color=MUTED,
                   title_font=dict(color=MUTED, size=11),
                   tickfont=dict(color=MUTED, size=10), linecolor=GRID),
        legend=dict(bgcolor="rgba(10,18,32,0.85)", bordercolor=GRID, borderwidth=1,
                    font=dict(size=11, color=TEXT)),
        margin=dict(l=58, r=25, t=58, b=52), **kw,
    )


def chart_frontier(mu, cov, rf, assets, weights: dict, cloud, frontier) -> "go.Figure":
    mc_r, mc_v, mc_s = cloud
    fv, fr = frontier
    vols = np.sqrt(np.diag(cov))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=mc_v * 100, y=mc_r * 100, mode="markers",
        marker=dict(size=3, color=mc_s, colorscale="RdYlGn", showscale=True,
                    colorbar=dict(title="Sharpe", x=0.99, thickness=12,
                                  tickfont=dict(color=MUTED, size=9),
                                  title_font=dict(color=MUTED, size=10))),
        opacity=0.30, name="Portefeuilles aléatoires",
        hovertemplate="Vol %{x:.2f}%  ·  Rdt %{y:.2f}%<extra></extra>"))
    if len(fv):
        fig.add_trace(go.Scatter(x=fv * 100, y=fr * 100, mode="lines",
                                 line=dict(color=GOLD, width=2.5),
                                 name="Frontière efficiente"))
    shapes = {"Max Sharpe": ("star", 24), "Min Variance": ("diamond", 18),
              "Risk Parity": ("pentagon", 18), "Min CVaR": ("hexagon", 17),
              "Rééchantillonné": ("cross", 17)}
    for name, w in weights.items():
        r, v, s = pmetrics(w, mu, cov, rf)
        sym, size = shapes.get(name, ("circle", 15))
        fig.add_trace(go.Scatter(
            x=[v * 100], y=[r * 100], mode="markers",
            marker=dict(size=size, symbol=sym, color=PORT_COLORS.get(name, GOLD),
                        line=dict(color=BG, width=2)),
            name=f"{name} · Sharpe {s:.2f}"))
    if "Max Sharpe" in weights:
        _, _, ts = pmetrics(weights["Max Sharpe"], mu, cov, rf)
        xr = np.linspace(0, float(mc_v.max()) * 100 * 1.2, 60)
        fig.add_trace(go.Scatter(x=xr, y=rf * 100 + ts * xr, mode="lines",
                                 line=dict(color="rgba(212,175,55,0.35)", width=1.5,
                                           dash="dash"),
                                 name="Droite de marché"))
    for i, a in enumerate(assets):
        fig.add_trace(go.Scatter(
            x=[vols[i] * 100], y=[mu[i] * 100], mode="markers+text",
            marker=dict(size=10, symbol="circle-open",
                        color=PALETTE[i % len(PALETTE)], line=dict(width=2.2)),
            text=[a], textposition="top right",
            textfont=dict(color=TEXT, size=10), name=a, showlegend=False))
    fig.update_layout(**base_layout("Frontière efficiente de Markowitz",
                                    "Volatilité annuelle (%)", "Rendement annuel (%)",
                                    height=560))
    return fig


def chart_mu_uncertainty(assets, mu, se_mu) -> "go.Figure":
    """
    Graphique clé de l'outil : l'intervalle de confiance sur μ.
    Il montre que le rendement espéré estimé sur l'historique est
    statistiquement indiscernable de zéro dans la plupart des cas — donc
    qu'un Sharpe à trois décimales n'a pas le sens qu'on lui prête.
    """
    order = np.argsort(mu)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mu[order] * 100, y=[assets[i] for i in order], mode="markers",
        marker=dict(size=11, color=GOLD, line=dict(color=BG, width=1.5)),
        error_x=dict(type="data", array=1.96 * se_mu[order] * 100,
                     color="rgba(212,175,55,0.45)", thickness=1.6, width=6),
        name="μ estimé ± 1,96 σ", hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=RED, width=1.2, dash="dash"))
    fig.update_layout(**base_layout(
        "Rendement espéré et son intervalle de confiance à 95 %",
        "Rendement annuel (%)", "",
        height=max(320, len(assets) * 44 + 130)))
    return fig


def chart_weight_uncertainty(assets, W_boot: np.ndarray) -> "go.Figure":
    """Dispersion des poids sur les tirages bootstrap (P4.1)."""
    fig = go.Figure()
    for i, a in enumerate(assets):
        fig.add_trace(go.Box(y=W_boot[:, i] * 100, name=a, boxpoints=False,
                             marker_color=PALETTE[i % len(PALETTE)],
                             line=dict(width=1.4)))
    fig.update_layout(**base_layout(
        "Dispersion des poids optimaux sur 60 rééchantillonnages",
        "", "Poids (%)", height=420, showlegend=False))
    return fig


def chart_oos(navs: dict[str, pd.Series], title: str) -> "go.Figure":
    fig = go.Figure()
    for name, nav in navs.items():
        fig.add_trace(go.Scatter(
            x=nav.index, y=nav.values * 100, mode="lines", name=name,
            line=dict(color=PORT_COLORS.get(name, GOLD),
                      width=2.6 if "1/N" not in name else 2.0,
                      dash="dot" if "1/N" in name or "60/40" in name else "solid"),
            hovertemplate=f"{name}: %{{y:.1f}}<extra></extra>"))
    fig.update_layout(**base_layout(title, "Date", "Base 100", height=460))
    return fig


def chart_fan(traj: np.ndarray, invested_path: np.ndarray, sym: str) -> "go.Figure":
    """Éventail de la projection : médiane et bandes de quantiles (P3.1)."""
    months = np.arange(traj.shape[1])
    qs = {q: np.quantile(traj, q, axis=0) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    fig = go.Figure()
    for lo, hi, op in ((0.05, 0.95, 0.12), (0.25, 0.75, 0.22)):
        fig.add_trace(go.Scatter(x=np.concatenate([months, months[::-1]]),
                                 y=np.concatenate([qs[hi], qs[lo][::-1]]),
                                 fill="toself", fillcolor=f"rgba(212,175,55,{op})",
                                 line=dict(width=0), hoverinfo="skip",
                                 name=f"{int(lo*100)}–{int(hi*100)}e centile"))
    fig.add_trace(go.Scatter(x=months, y=qs[0.5], mode="lines", name="Médiane",
                             line=dict(color=GOLD, width=2.6)))
    fig.add_trace(go.Scatter(x=months, y=invested_path, mode="lines",
                             name="Capital versé", line=dict(color=MUTED, width=1.6,
                                                             dash="dash")))
    fig.update_layout(**base_layout("Projection du capital (bootstrap par blocs)",
                                    "Mois", f"Capital ({sym})", height=460))
    return fig


def chart_terminal_hist(terminal: np.ndarray, invested: float, goal: Optional[float],
                        sym: str) -> "go.Figure":
    fig = go.Figure(go.Histogram(x=terminal, nbinsx=70,
                                 marker=dict(color=GOLD, opacity=0.75,
                                             line=dict(color=BG, width=0.5)),
                                 name="Capital final"))
    fig.add_vline(x=invested, line=dict(color=MUTED, width=1.6, dash="dash"),
                  annotation_text="Capital versé", annotation_font_color=MUTED)
    fig.add_vline(x=float(np.median(terminal)), line=dict(color=GREEN, width=1.8),
                  annotation_text="Médiane", annotation_font_color=GREEN)
    if goal:
        fig.add_vline(x=goal, line=dict(color=BLUE, width=1.8, dash="dot"),
                      annotation_text="Objectif", annotation_font_color=BLUE)
    fig.update_layout(**base_layout("Distribution du capital final",
                                    f"Capital ({sym})", "Simulations", height=380))
    return fig


def chart_prices(prices: pd.DataFrame) -> "go.Figure":
    norm = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for i, c in enumerate(norm.columns):
        fig.add_trace(go.Scatter(x=norm.index, y=norm[c], mode="lines", name=c,
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=1.8)))
    fig.update_layout(**base_layout("Prix normalisés en devise de base (base 100)",
                                    "Date", "Indice"))
    return fig


def chart_drawdown(navs: dict[str, pd.Series]) -> "go.Figure":
    fig = go.Figure()
    for name, nav in navs.items():
        dd = drawdown_series(nav) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=name,
                                 fill="tozeroy",
                                 line=dict(color=PORT_COLORS.get(name, GOLD), width=1.5),
                                 opacity=0.65))
    fig.update_layout(**base_layout("Drawdown des portefeuilles (%)", "Date",
                                    "Drawdown (%)"))
    return fig


def chart_pie(w, assets, title, subtitle="") -> "go.Figure":
    mask = np.asarray(w) > 0.005
    labels = [assets[i] for i in range(len(assets)) if mask[i]]
    vals = np.asarray(w)[mask]
    fig = go.Figure()
    if not len(vals):
        fig.add_annotation(text="Aucun poids significatif", xref="paper", yref="paper",
                           x=0.5, y=0.5, font=dict(color=TEXT, size=14), showarrow=False)
    else:
        fig.add_trace(go.Pie(labels=labels, values=vals, hole=0.46,
                             marker=dict(colors=PALETTE[:len(labels)],
                                         line=dict(color=BG, width=2.5)),
                             textinfo="label+percent", textfont=dict(size=13),
                             hovertemplate="%{label}: %{percent}<extra></extra>"))
    fig.update_layout(**base_layout(f"{title}{'  ·  ' + subtitle if subtitle else ''}",
                                    height=430))
    return fig


def chart_correlation(returns: pd.DataFrame) -> "go.Figure":
    corr = returns.corr()
    z = np.round(corr.values, 2)
    fig = go.Figure(go.Heatmap(
        z=z, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0.0, RED], [0.5, "#1A2540"], [1.0, GOLD]],
        zmid=0, zmin=-1, zmax=1, text=z, texttemplate="%{text:.2f}",
        textfont=dict(size=11),
        colorbar=dict(title="ρ", tickfont=dict(color=MUTED),
                      title_font=dict(color=MUTED))))
    fig.update_layout(**base_layout("Matrice de corrélation",
                                    height=max(380, len(returns.columns) * 62 + 90)))
    return fig


def chart_risk_contrib(assets, weights: dict, cov) -> "go.Figure":
    fig = go.Figure()
    for name, w in weights.items():
        fig.add_trace(go.Bar(name=name, x=list(assets), y=risk_contributions(w, cov),
                             marker=dict(color=PORT_COLORS.get(name, GOLD), opacity=0.85,
                                         line=dict(color=BG, width=1))))
    fig.update_layout(**base_layout("Contribution au risque par actif (%)", "Actif",
                                    "Contribution (%)", barmode="group", height=400))
    return fig


def chart_tail(rets: dict[str, pd.Series], alpha: float = 0.95) -> "go.Figure":
    """VaR et CVaR côte à côte : la volatilité seule masque le risque de queue."""
    names, var_v, cvar_v = [], [], []
    for n, r in rets.items():
        names.append(n)
        var_v.append(historical_var(r, alpha) * 100)
        cvar_v.append(historical_cvar(r, alpha) * 100)
    fig = go.Figure()
    fig.add_trace(go.Bar(name=f"VaR {int(alpha*100)} % (1 j)", x=names, y=var_v,
                         marker=dict(color=ORANGE, opacity=0.85),
                         text=[f"{v:.2f}%" for v in var_v], textposition="outside"))
    fig.add_trace(go.Bar(name=f"CVaR {int(alpha*100)} % (1 j)", x=names, y=cvar_v,
                         marker=dict(color=RED, opacity=0.85),
                         text=[f"{v:.2f}%" for v in cvar_v], textposition="outside"))
    fig.update_layout(**base_layout("Perte quotidienne dans la queue de distribution",
                                    "", "Perte (%)", barmode="group", height=380))
    return fig


def chart_factors(fac: pd.DataFrame) -> "go.Figure":
    body = fac[fac["Facteur"] != "Alpha (ann.)"]
    fig = go.Figure(go.Bar(
        x=body["Coefficient"], y=body["Facteur"], orientation="h",
        marker=dict(color=[GOLD if s else MUTED for s in body["Significatif"]],
                    opacity=0.88, line=dict(color=BG, width=1)),
        text=[f"{b:.2f} (t={t:.1f})" for b, t in zip(body["Coefficient"], body["t-stat"])],
        textposition="outside", textfont=dict(color=TEXT, size=11)))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(**base_layout(
        f"Exposition factorielle  ·  R² = {fac.attrs.get('r2', float('nan')):.2f}",
        "Bêta", "", height=360))
    return fig


def chart_scenario_cum(sc: dict) -> "go.Figure":
    fig = go.Figure()
    for name, stats in sc.get("portfolios", {}).items():
        cum = stats["cum_series"]
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name=name,
                                 line=dict(color=PORT_COLORS.get(name, GOLD), width=2.5)))
    prices = sc.get("prices", pd.DataFrame())
    if not prices.empty:
        norm = prices / prices.iloc[0] * 100
        for i, c in enumerate(norm.columns):
            fig.add_trace(go.Scatter(x=norm.index, y=norm[c], mode="lines", name=c,
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=1,
                                               dash="dot"), opacity=0.4))
    fig.update_layout(**base_layout(f"Scénario : {sc['name']} (base 100)", "Date",
                                    "Valeur", height=430))
    return fig


def chart_scenario_bars(sc: dict) -> "go.Figure":
    ports = sc.get("portfolios", {})
    names = list(ports)
    rets = [ports[n]["total_return"] * 100 for n in names]
    mdds = [ports[n]["max_drawdown"] * 100 for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Rendement total (%)", x=names, y=rets,
                         marker=dict(color=[GREEN if r >= 0 else RED for r in rets],
                                     opacity=0.85),
                         text=[f"{r:+.1f}%" for r in rets], textposition="outside"))
    fig.add_trace(go.Bar(name="Drawdown max (%)", x=names, y=mdds,
                         marker=dict(color=RED, opacity=0.55),
                         text=[f"{d:.1f}%" for d in mdds], textposition="outside"))
    fig.update_layout(**base_layout("Performance pendant le scénario", "", "%",
                                    barmode="group", height=380))
    return fig


def chart_alloc_bars(df: pd.DataFrame, sym: str) -> "go.Figure":
    d = df[df["Montant investi"] > 0]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Cible", x=d["Montant cible"], y=d["Actif"],
                         orientation="h", marker=dict(color=MUTED, opacity=0.5)))
    fig.add_trace(go.Bar(name="Exécutable", x=d["Montant investi"], y=d["Actif"],
                         orientation="h", marker=dict(color=GOLD, opacity=0.9),
                         text=[f"{sym}{v:,.0f}" for v in d["Montant investi"]],
                         textposition="outside"))
    fig.update_layout(**base_layout("Montant cible et montant réellement exécutable",
                                    f"Montant ({sym})", "", barmode="group",
                                    height=max(300, len(d) * 48 + 130)))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# INTERFACE — styles et composants
# ═══════════════════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
:root{--bg:#05080D;--bg2:#0A1220;--border:rgba(212,175,55,0.18);--gold:#D4AF37;
--gold-dim:#B8860B;--text:#EEF2F7;--muted:#64748B;--green:#10B981;--red:#EF4444;}
*,*::before,*::after{font-family:'Inter',-apple-system,sans-serif!important;box-sizing:border-box;}
[data-testid="stAppViewContainer"]{background:var(--bg)!important;}
[data-testid="stHeader"]{background:rgba(5,8,13,0.95)!important;backdrop-filter:blur(12px);border-bottom:1px solid var(--border)!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#060B14 0%,#0B1828 100%)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label,[data-testid="stSidebar"] div{color:var(--text)!important;}
[data-testid="stSidebar"] small{color:var(--muted)!important;font-size:0.72rem!important;}
[data-testid="stMetricValue"]{color:var(--gold)!important;font-size:1.4rem!important;font-weight:700!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:0.68rem!important;text-transform:uppercase!important;letter-spacing:0.1em!important;}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border)!important;gap:0!important;flex-wrap:wrap;}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;font-weight:500!important;font-size:0.8rem!important;padding:11px 15px!important;background:transparent!important;border-bottom:2px solid transparent!important;}
.stTabs [aria-selected="true"]{color:var(--gold)!important;border-bottom:2px solid var(--gold)!important;background:rgba(212,175,55,0.05)!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:22px!important;}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,var(--gold-dim),var(--gold),var(--gold-dim))!important;color:#05080D!important;border:none!important;font-weight:700!important;font-size:0.82rem!important;letter-spacing:0.14em!important;text-transform:uppercase!important;padding:0.75rem 1rem!important;border-radius:8px!important;box-shadow:0 0 28px rgba(212,175,55,0.32)!important;}
.stButton>button:not([kind="primary"]){background:rgba(13,18,28,0.8)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:7px!important;font-size:0.78rem!important;}
.stButton>button:not([kind="primary"]):hover{border-color:var(--gold)!important;color:var(--gold)!important;}
.stTextInput input,.stNumberInput input{background:var(--bg2)!important;border:1px solid rgba(212,175,55,0.2)!important;border-radius:8px!important;color:var(--text)!important;}
.stSelectbox>div>div{background:var(--bg2)!important;border:1px solid rgba(212,175,55,0.2)!important;border-radius:8px!important;color:var(--text)!important;}
[data-testid="stDataFrame"] th{background:linear-gradient(135deg,#0D1828,#121F33)!important;color:var(--muted)!important;font-size:0.7rem!important;text-transform:uppercase!important;letter-spacing:0.08em!important;}
[data-testid="stDataFrame"] td{color:var(--text)!important;font-size:0.82rem!important;}
hr{border:none!important;height:1px!important;background:linear-gradient(90deg,transparent,var(--gold),transparent)!important;opacity:0.25!important;}
[data-testid="stMarkdownContainer"] p{color:var(--text)!important;}
[data-testid="stMarkdownContainer"] code{background:rgba(212,175,55,0.1)!important;color:var(--gold)!important;border-radius:4px!important;padding:1px 5px!important;}
::-webkit-scrollbar{width:5px;height:5px;}::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:rgba(212,175,55,0.22);border-radius:3px;}
.legal{position:sticky;top:0;z-index:99;background:rgba(239,68,68,0.07);
border:1px solid rgba(239,68,68,0.28);border-radius:8px;padding:7px 14px;
color:#FCA5A5;font-size:0.71rem;letter-spacing:0.02em;margin-bottom:14px;}
@media (prefers-reduced-motion: no-preference){.fade-in{animation:fade-in .4s ease forwards;}}
@keyframes fade-in{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:translateY(0);}}
</style>
"""


def fmt_pct(x, dec=2, signed=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x*100:{'+' if signed else ''}.{dec}f} %"


def fmt_num(x, dec=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{dec}f}"


def legal_banner():
    st.markdown(
        '<div class="legal">⚖️ Simulation pédagogique — ne constitue pas un conseil '
        'en investissement personnalisé (art. L.541-1 CMF). Les performances '
        'passées ne préjugent pas des performances futures.</div>',
        unsafe_allow_html=True)


def page_header():
    st.markdown("""
<div class="fade-in" style="display:flex;align-items:center;gap:16px;padding:20px 0 18px 0;
 border-bottom:1px solid rgba(212,175,55,0.15);margin-bottom:20px;">
 <div style="width:46px;height:46px;flex-shrink:0;background:linear-gradient(135deg,#B8860B,#D4AF37);
  border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;">📈</div>
 <div><div style="font-family:'Playfair Display',serif;font-size:1.85rem;font-weight:700;
   line-height:1.15;background:linear-gradient(135deg,#C9A440,#F0D060,#C9A440);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
   Portfolio Lab</div>
  <div style="color:#64748B;font-size:0.71rem;letter-spacing:0.14em;text-transform:uppercase;
   margin-top:3px;">Optimisation · Test hors échantillon · Projection · d'après AEG</div>
 </div></div>""", unsafe_allow_html=True)


def section_title(text, sub=""):
    sub_html = f'<div style="color:#64748B;font-size:0.73rem;margin-top:3px;">{sub}</div>' if sub else ""
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin:24px 0 14px 0;">
 <div style="width:3px;height:26px;border-radius:2px;
  background:linear-gradient(180deg,#D4AF37,rgba(212,175,55,0));"></div>
 <div><div style="font-size:0.95rem;font-weight:600;color:#EEF2F7;">{text}</div>{sub_html}</div>
</div>""", unsafe_allow_html=True)


def kpi_card(label, value, note="", color=GOLD):
    note_html = (f'<div style="color:#64748B;font-size:0.63rem;margin-top:4px;">{note}</div>'
                 if note else "")
    return (f'<div style="background:linear-gradient(135deg,#0D1828,#121F33);'
            f'border:1px solid rgba(212,175,55,0.18);border-radius:14px;padding:16px 18px;'
            f'height:100%;"><div style="color:#64748B;font-size:0.64rem;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.1em;">{label}</div>'
            f'<div style="color:{color};font-size:1.35rem;font-weight:700;margin-top:6px;'
            f'line-height:1.1;">{value}</div>{note_html}</div>')


def sidebar_section(title):
    st.sidebar.markdown(
        f'<div style="font-size:0.68rem;font-weight:700;color:#64748B;text-transform:uppercase;'
        f'letter-spacing:0.12em;margin:16px 0 8px 0;padding-bottom:6px;'
        f'border-bottom:1px solid rgba(212,175,55,0.1);">{title}</div>',
        unsafe_allow_html=True)


def delta_color(val):
    s = str(val)
    if s.startswith("+"):
        return f"color:{GREEN}"
    if s.startswith("-") or s.startswith("−"):
        return f"color:{RED}"
    return f"color:{MUTED}"


# ─── Profils de risque (P3.2) ───────────────────────────────────────────────
RISK_PROFILES = {
    "Prudent": {"max_vol": 0.07, "max_single": 0.25, "equity_cap": 0.40,
                "note": "Volatilité cible ≤ 7 %, forte pondération obligataire."},
    "Équilibré": {"max_vol": 0.12, "max_single": 0.35, "equity_cap": 0.70,
                  "note": "Volatilité cible ≤ 12 %."},
    "Dynamique": {"max_vol": 0.18, "max_single": 0.45, "equity_cap": 0.90,
                  "note": "Volatilité cible ≤ 18 %, horizon long requis."},
    "Offensif": {"max_vol": 0.30, "max_single": 1.00, "equity_cap": 1.00,
                 "note": "Aucun garde-fou automatique."},
}


def render_sidebar() -> dict:
    """Toutes les entrées utilisateur. Renvoie un dictionnaire de paramètres."""
    st.sidebar.markdown("""
<div style="padding:16px 0 12px 0;border-bottom:1px solid rgba(212,175,55,0.14);">
 <div style="font-family:'Playfair Display',serif;font-size:1.12rem;font-weight:700;color:#D4AF37;">
  📈 Portfolio Lab</div>
 <div style="color:#64748B;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.14em;
  margin-top:4px;">Markowitz · robustesse · hors échantillon</div></div>""",
                        unsafe_allow_html=True)

    # ── Profil investisseur (P3.2) ───────────────────────────────────────
    sidebar_section("Profil investisseur")
    profile = st.sidebar.selectbox("Profil de risque", list(RISK_PROFILES),
                                   index=1, key="profile")
    st.sidebar.caption(RISK_PROFILES[profile]["note"])
    horizon = st.sidebar.slider("Horizon d'investissement (années)", 1, 40, 10, key="horizon")
    apply_profile = st.sidebar.checkbox(
        "Appliquer les garde-fous du profil", value=True, key="apply_profile",
        help="Plafonne le poids par ligne et l'exposition actions selon le profil.")

    # ── Devise et capital ────────────────────────────────────────────────
    sidebar_section("Devise et capital")
    base_ccy = st.sidebar.selectbox(
        "Devise de base", list(CURRENCIES), index=0, key="base_ccy",
        format_func=lambda c: f"{CURRENCIES[c]['flag']}  {c}")
    sym = CURRENCIES[base_ccy]["symbol"]
    allow_static_fx = st.sidebar.checkbox(
        "Autoriser un change statique en cas d'échec", value=False, key="static_fx",
        help="Par défaut l'application s'arrête si un taux de change est "
             "indisponible, plutôt que d'afficher des montants faux.")
    initial = st.sidebar.number_input(f"Capital initial ({sym})", 100, 100_000_000,
                                      10_000, step=500, key="initial")
    monthly = st.sidebar.number_input(f"Versement mensuel ({sym})", 0, 1_000_000,
                                      200, step=50, key="monthly")
    goal = st.sidebar.number_input(f"Objectif de capital ({sym}, 0 = aucun)", 0,
                                   1_000_000_000, 0, step=1000, key="goal")

    # ── Période et taux sans risque ──────────────────────────────────────
    sidebar_section("Période d'analyse")
    c1, c2 = st.sidebar.columns(2)
    start = c1.date_input("Début", date(2015, 1, 1), key="start")
    end = c2.date_input("Fin", date.today(), key="end")
    rf_auto, rf_src = (fetch_risk_free(base_ccy) if HAS_YF
                       else (CURRENCIES[base_ccy]["rf_default"], "défaut"))
    rf_pct = st.sidebar.slider("Taux sans risque (%)", 0.0, 10.0,
                               float(round(rf_auto * 100, 2)), 0.05, key="rf")
    st.sidebar.caption(f"Source : {rf_src}. Doit correspondre à la devise et à l'horizon.")

    # ── Univers ──────────────────────────────────────────────────────────
    sidebar_section("Univers d'investissement")
    if "tickers" not in st.session_state:
        st.session_state.tickers = ["SPY", "AGG", "GLD", "EFA"]
    ci, cb = st.sidebar.columns([3, 1])
    new_t = ci.text_input("Symbole", placeholder="SPY, CW8.PA, BTC-USD…",
                          label_visibility="collapsed", key="new_ticker")
    if cb.button("＋", key="add_ticker"):
        for raw in new_t.replace(";", ",").split(","):
            t = raw.strip().upper()
            if t and t not in st.session_state.tickers:
                st.session_state.tickers.append(t)
        st.rerun()

    for t in list(st.session_state.tickers):
        c1, c2 = st.sidebar.columns([5, 1])
        c1.markdown(f'<code style="color:#D4AF37;background:rgba(212,175,55,0.1);'
                    f'border:1px solid rgba(212,175,55,0.2);border-radius:4px;'
                    f'padding:2px 6px;font-size:0.82rem;">{t}</code>',
                    unsafe_allow_html=True)
        # P1.8 : clé indexée par TICKER, plus par position (plus de décalage
        # des états de widgets après suppression).
        if c2.button("✕", key=f"rm_{t}"):
            if len(st.session_state.tickers) > 2:
                st.session_state.tickers.remove(t)
                st.rerun()
            else:
                st.sidebar.warning("Deux actifs au minimum.")

    # ── Modèle d'estimation ──────────────────────────────────────────────
    sidebar_section("Modèle d'estimation")
    mu_label = st.sidebar.selectbox(
        "Rendement espéré (μ)",
        ["Historique (peu fiable)", "Black-Litterman", "Aucun — risque seul"],
        index=1, key="mu_method",
        help="μ estimé sur l'historique est presque du bruit : son erreur-type "
             "dépasse souvent sa valeur. Black-Litterman part de l'équilibre de "
             "marché ; « Aucun » n'active que Min Variance et Risk Parity.")
    mu_method = {"Historique (peu fiable)": "historical",
                 "Black-Litterman": "black_litterman",
                 "Aucun — risque seul": "none"}[mu_label]
    cov_label = st.sidebar.selectbox("Covariance (Σ)",
                                     ["Ledoit-Wolf (recommandé)", "Empirique"],
                                     index=0, key="cov_method")
    cov_method = "ledoit_wolf" if cov_label.startswith("Ledoit") else "sample"
    freq_label = st.sidebar.selectbox(
        "Fréquence d'estimation", ["Automatique", "Quotidienne", "Hebdomadaire",
                                   "Mensuelle"], index=0, key="est_freq",
        help="Les rendements quotidiens sous-estiment les corrélations entre "
             "places situées dans des fuseaux différents. « Automatique » passe "
             "en hebdomadaire dès que le portefeuille mêle plusieurs places.")
    est_freq = {"Automatique": "auto", "Quotidienne": "D",
                "Hebdomadaire": "W", "Mensuelle": "M"}[freq_label]

    # Black-Litterman n'a d'intérêt que si l'utilisateur exprime des vues :
    # sans vue, le résultat reproduit exactement le portefeuille d'équilibre.
    bl_views, bl_conf, bl_prior_mode = {}, {}, "equal"
    if mu_method == "black_litterman":
        with st.sidebar.expander("🎯 Vues de marché", expanded=False):
            st.caption("Sans aucune vue, l'optimisation renvoie le portefeuille "
                       "d'équilibre choisi ci-dessous. Une confiance à 0 désactive "
                       "la vue correspondante.")
            prior_choice = st.radio(
                "Portefeuille d'équilibre",
                ["Équipondéré", "Capitalisation boursière (lent)"],
                index=0, key="bl_prior")
            bl_prior_mode = "cap" if prior_choice.startswith("Capitalisation") else "equal"
            for t in st.session_state.tickers:
                st.markdown(f"**{t}**")
                bl_views[t] = st.number_input(
                    "Rendement annuel attendu (%)", -50.0, 100.0, 0.0, 0.5,
                    key=f"blv_{t}", label_visibility="collapsed") / 100.0
                bl_conf[t] = st.slider("Confiance", 0.0, 1.0, 0.0, 0.05,
                                       key=f"blc_{t}")

    # ── Frais ────────────────────────────────────────────────────────────
    sidebar_section("Frais et frottements")
    broker = st.sidebar.number_input("Courtage (pb, aller simple)", 0.0, 200.0,
                                     DEFAULT_BROKER_BPS, 1.0, key="broker")
    spread = st.sidebar.number_input("Demi-spread (pb)", 0.0, 200.0,
                                     DEFAULT_SPREAD_BPS, 1.0, key="spread")
    ter = st.sidebar.number_input("Frais courants annuels moyens (pb)", 0.0, 300.0,
                                  DEFAULT_TER_BPS, 1.0, key="ter")
    reb_label = st.sidebar.selectbox("Rééquilibrage", list(REBALANCE_CHOICES),
                                     index=2, key="rebalance",
                                     help="Définition unique utilisée partout : "
                                          "KPI, drawdowns et scénarios.")
    rebalance = REBALANCE_CHOICES[reb_label]

    # ── Contraintes ──────────────────────────────────────────────────────
    sidebar_section("Contraintes")
    use_cons = st.sidebar.checkbox("Contraintes par actif", value=False, key="use_cons")
    per_asset = {}
    if use_cons:
        for t in st.session_state.tickers:
            with st.sidebar.expander(f"⚖️ {t}", expanded=False):
                mn = st.slider("Min %", 0, 100, 0, 1, key=f"min_{t}")
                mx = st.slider("Max %", 0, 100, 100, 1, key=f"max_{t}")
                if mn > mx:
                    st.error("Le minimum dépasse le maximum.")
                per_asset[t] = (mn / 100.0, mx / 100.0)
    class_caps = {}
    if st.sidebar.checkbox("Plafonds par classe d'actifs", value=False, key="use_class"):
        for cls in ("Equity", "Bond", "Crypto", "Commodity"):
            class_caps[cls] = st.sidebar.slider(f"Max {cls} %", 0, 100, 100, 5,
                                                key=f"cls_{cls}") / 100.0
    max_assets = st.sidebar.number_input("Nombre maximum de lignes (0 = libre)", 0, 50,
                                         0, 1, key="max_assets")

    # ── Calculs optionnels ───────────────────────────────────────────────
    sidebar_section("Analyses")
    do_oos = st.sidebar.checkbox("Backtest hors échantillon", value=True, key="do_oos")
    lookback = st.sidebar.slider("Fenêtre d'estimation (années)", 1.0, 8.0, 3.0, 0.5,
                                 key="lookback", disabled=not do_oos)
    oos_freq = st.sidebar.selectbox("Fréquence de ré-optimisation",
                                    ["Mensuelle", "Trimestrielle", "Annuelle"], index=1,
                                    key="oos_freq", disabled=not do_oos)
    do_resample = st.sidebar.checkbox("Rééchantillonnage de Michaud", value=True,
                                      key="do_resample")
    do_cvar = st.sidebar.checkbox("Optimisation min-CVaR", value=False, key="do_cvar")

    # ── Exécution ────────────────────────────────────────────────────────
    sidebar_section("Exécution")
    fractional = st.sidebar.checkbox("Fractions d'actions autorisées", value=False,
                                     key="fractional")
    tax_label = st.sidebar.selectbox("Fiscalité (projection)", list(TAX_REGIMES),
                                     index=0, key="tax")
    inflation = st.sidebar.slider("Inflation annuelle (%)", 0.0, 8.0, 2.0, 0.1,
                                  key="inflation") / 100.0

    st.sidebar.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    ack = st.sidebar.checkbox(
        "Je comprends qu'il s'agit d'une simulation, pas d'un conseil.",
        value=False, key="ack")
    run = st.sidebar.button("▶  LANCER L'ANALYSE", type="primary",
                            use_container_width=True, disabled=not ack)
    if not ack:
        st.sidebar.caption("Cochez la case ci-dessus pour activer le calcul.")

    return dict(
        profile=profile, apply_profile=apply_profile, horizon=horizon,
        base_ccy=base_ccy, sym=sym, allow_static_fx=allow_static_fx,
        initial=float(initial), monthly=float(monthly), goal=float(goal),
        start=start, end=end, rf=rf_pct / 100.0, rf_src=rf_src,
        tickers=list(st.session_state.tickers),
        mu_method=mu_method, cov_method=cov_method, est_freq=est_freq,
        bl_views=bl_views, bl_conf=bl_conf, bl_prior_mode=bl_prior_mode,
        costs=CostModel(broker, spread, ter), rebalance=rebalance,
        reb_label=reb_label, per_asset=per_asset if use_cons else {},
        class_caps=class_caps, max_assets=int(max_assets) or None,
        do_oos=do_oos, lookback=float(lookback),
        oos_freq={"Mensuelle": "M", "Trimestrielle": "Q", "Annuelle": "A"}[oos_freq],
        do_resample=do_resample, do_cvar=do_cvar,
        fractional=fractional, tax=tax_label, inflation=inflation, run=run,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE D'ANALYSE
# ═══════════════════════════════════════════════════════════════════════════

def build_constraints(assets: Sequence[str], classes: dict[str, str], p: dict
                      ) -> Constraints:
    """Assemble contraintes par actif, garde-fous du profil et plafonds de classe."""
    n = len(assets)
    min_w, max_w = np.zeros(n), np.ones(n)
    for i, a in enumerate(assets):
        if a in p["per_asset"]:
            min_w[i], max_w[i] = p["per_asset"][a]
    if p["apply_profile"]:
        cap = RISK_PROFILES[p["profile"]]["max_single"]
        max_w = np.minimum(max_w, cap)

    groups: dict[str, tuple[list[int], float, float]] = {}
    for cls, cap in p["class_caps"].items():
        idx = [i for i, a in enumerate(assets) if classes.get(a) == cls]
        if idx and cap < 1.0:
            groups[cls] = (idx, 0.0, float(cap))
    if p["apply_profile"]:
        eq_cap = RISK_PROFILES[p["profile"]]["equity_cap"]
        idx = [i for i, a in enumerate(assets)
               if classes.get(a) in ("Equity", "Crypto", "Real estate")]
        if idx and eq_cap < 1.0:
            prev = groups.get("Equity", (idx, 0.0, 1.0))[2]
            groups["Actifs risqués"] = (idx, 0.0, float(min(eq_cap, prev)))

    cons = Constraints(n, min_w, max_w, groups, p["max_assets"])
    check_feasibility(cons)                                   # P1.1
    return cons


def run_analysis(p: dict, log: Callable[[str], None]) -> dict:
    """Exécute la chaîne complète. Lève PortfolioLabError avec un message clair."""
    if len(p["tickers"]) < 2:
        raise PortfolioLabError("Sélectionnez au moins deux actifs.")
    if p["start"] >= p["end"]:
        raise PortfolioLabError("La date de début doit précéder la date de fin.")

    # ── 1. Validation ────────────────────────────────────────────────────
    log("Validation des symboles…")
    val = {t: validate_ticker(t) for t in p["tickers"]}
    ok = [t for t, v in val.items() if v.get("valid")]
    if len(ok) < 2:
        bad = [t for t in p["tickers"] if t not in ok]
        raise DataError(f"Moins de deux symboles valides. À corriger : {', '.join(bad)}.")

    # ── 2. Prix bruts, en devise de cotation ─────────────────────────────
    log("Téléchargement des historiques…")
    raw = fetch_raw_prices(tuple(ok), str(p["start"]), str(p["end"]))
    ccy_map = {t: val[t].get("currency", "USD") for t in ok if t in raw.columns}

    # ── 3. Conversion FX AVANT les rendements (P1.3) ─────────────────────
    fx = None
    foreign = sorted({c for c in ccy_map.values() if c != p["base_ccy"]})
    if foreign:
        log(f"Conversion en {p['base_ccy']} ({', '.join(foreign)})…")
        try:
            fx = fetch_fx_series(tuple(sorted(set(ccy_map.values()))), p["base_ccy"],
                                 str(p["start"]), str(p["end"]))
        except FXError:
            if not p["allow_static_fx"]:
                raise
    prices, fx_info = convert_to_base(raw, ccy_map, p["base_ccy"], fx,
                                      p["allow_static_fx"])

    # ── 4. Historique commun ─────────────────────────────────────────────
    prices, dropped = align_common_history(prices)
    assets = list(prices.columns)
    classes = {a: classify_asset(a, val[a].get("asset_type", ""),
                                 val[a].get("exchange", "")) for a in assets}
    quality = data_quality_report(prices)

    # ── 5. Moments ───────────────────────────────────────────────────────
    log("Estimation des moments…")
    rets = simple_returns(prices)
    # Fréquence d'estimation : hebdomadaire dès que plusieurs places de
    # cotation coexistent, pour ne pas surestimer la diversification.
    est_freq = p.get("est_freq", "auto")
    venues = {(val[a].get("exchange") or "?") for a in assets}
    auto_multi = len(venues) > 1 or len({ccy_map.get(a) for a in assets}) > 1
    if est_freq == "auto":
        est_freq = "W" if auto_multi else "D"
    spec = ModelSpec(mu_method=p["mu_method"], cov_method=p["cov_method"],
                     est_freq=est_freq)
    bl_has_views = False
    if p["mu_method"] == "black_litterman":
        conf = np.array([float(p.get("bl_conf", {}).get(a, 0.0)) for a in assets])
        views = np.array([float(p.get("bl_views", {}).get(a, np.nan))
                          if conf[i] > 0 else np.nan for i, a in enumerate(assets)])
        bl_has_views = bool(np.any(conf > 0))
        prior_w = None
        if p.get("bl_prior_mode") == "cap":
            log("Récupération des capitalisations boursières…")
            caps = fetch_market_caps(tuple(assets))
            if len(caps) == len(assets):
                prior_w = np.array([caps[a] for a in assets])
            else:
                errors_prior = ", ".join(a for a in assets if a not in caps)
                log(f"Capitalisation indisponible pour {errors_prior} : "
                    f"prior équipondéré retenu.")
        spec.bl_views, spec.bl_confidence, spec.bl_prior_w = views, conf, prior_w
    mu, cov, diag = estimate_moments(rets, spec, p["rf"])

    # ── 6. Contraintes ───────────────────────────────────────────────────
    cons = build_constraints(assets, classes, p)

    # ── 7. Optimisations ─────────────────────────────────────────────────
    log("Optimisation…")
    weights, errors = {}, {}

    def _add(name: str, solver, resolver=None):
        """Résout, applique la cardinalité, et consigne l'échec sans l'avaler."""
        try:
            w = solver(cons)
            if cons.max_assets and resolver is not None:
                w = apply_cardinality(w, cons.max_assets, resolver, cons)
            weights[name] = w
        except PortfolioLabError as e:
            errors[name] = str(e)

    _add("Min Variance", lambda c: solve_min_variance(cov, c),
         lambda keep: solve_min_variance(cov[np.ix_(keep, keep)], cons.subset(keep)))
    _add("Risk Parity", lambda c: solve_risk_parity(cov, c),
         lambda keep: solve_risk_parity(cov[np.ix_(keep, keep)], cons.subset(keep)))
    if p["mu_method"] != "none":
        _add("Max Sharpe", lambda c: solve_max_sharpe(mu, cov, c, p["rf"]),
             lambda keep: solve_max_sharpe(mu[keep], cov[np.ix_(keep, keep)],
                                           cons.subset(keep), p["rf"]))
    if p["do_cvar"]:
        log("Optimisation min-CVaR…")
        _add("Min CVaR", lambda c: solve_min_cvar(rets, c, 0.95),
             lambda keep: solve_min_cvar(rets.iloc[:, keep], cons.subset(keep), 0.95))
    weights["Equal weight (1/N)"] = _clean(np.ones(len(assets)) / len(assets), cons)
    w6040 = sixty_forty_weights(assets, classes)
    if w6040 is not None:
        weights["Benchmark 60/40"] = w6040
    if not weights:
        raise OptimizationError("Aucun portefeuille n'a pu être calculé. " +
                                " ".join(errors.values()))

    # ── 8. Rééchantillonnage (P4.1) ──────────────────────────────────────
    W_boot = None
    if p["do_resample"] and p["mu_method"] != "none":
        log("Rééchantillonnage de Michaud (60 tirages)…")
        try:
            w_rs, W_boot = resampled_frontier_weights(rets, cons, p["rf"], spec,
                                                      "max_sharpe", n_boot=60)
            weights["Rééchantillonné"] = w_rs
        except PortfolioLabError as e:
            errors["Rééchantillonné"] = str(e)

    # ── 9. Statistiques nettes de frais, définition unique (P1.2) ────────
    log("Calcul des performances nettes de frais…")
    bench_key = "Benchmark 60/40" if "Benchmark 60/40" in weights else "Equal weight (1/N)"
    bench_path = portfolio_path(rets, weights[bench_key], rebalance=p["rebalance"],
                                costs=p["costs"])
    stats = {}
    for name, w in weights.items():
        stats[name] = portfolio_stats(rets, w, p["rf"], rebalance=p["rebalance"],
                                      costs=p["costs"],
                                      bench_rets=bench_path["returns"])

    # ── 10. Frontière et nuage ───────────────────────────────────────────
    log("Construction de la frontière…")
    n_failed = 0
    if p["mu_method"] != "none":
        fv, fr, n_failed = efficient_frontier(mu, cov, cons, 60)
        cloud = random_portfolio_cloud(mu, cov, cons, rf=p["rf"])
    else:
        fv, fr = np.array([]), np.array([])
        cloud = (np.array([0.0]), np.array([0.0]), np.array([0.0]))

    # ── 11. Backtest hors échantillon (P2.2) ─────────────────────────────
    oos = {}
    if p["do_oos"]:
        methods = [("Max Sharpe", "max_sharpe"), ("Min Variance", "min_variance"),
                   ("Risk Parity", "risk_parity"), ("Equal weight (1/N)", "equal_weight")]
        if p["mu_method"] == "none":
            methods = [m for m in methods if m[1] != "max_sharpe"]
        for label, m in methods:
            log(f"Backtest hors échantillon — {label}…")
            try:
                oos[label] = walk_forward_backtest(
                    rets, method=m, cons=cons, rf=p["rf"], spec=spec,
                    lookback_years=p["lookback"], reb_freq=p["oos_freq"],
                    costs=p["costs"],
                    progress=lambda f, msg, lb=label: log(f"{lb} — {msg}")
                    if f in (0.5, 1.0) else None)
            except PortfolioLabError as e:
                errors[f"OOS {label}"] = str(e)

    params_snapshot = {k: (str(v) if isinstance(v, (date, CostModel)) else v)
                       for k, v in p.items() if k not in ("costs", "run")}
    params_snapshot["costs"] = {"broker_bps": p["costs"].broker_bps,
                                "spread_bps": p["costs"].spread_bps,
                                "ter_bps": p["costs"].ter_bps}

    return dict(
        prices=prices, returns=rets, assets=assets, classes=classes,
        mu=mu, cov=cov, diag=diag, spec=spec, cons=cons,          # P1.7
        weights=weights, stats=stats, errors=errors, W_boot=W_boot,
        frontier=(fv, fr), frontier_failed=n_failed, cloud=cloud,
        oos=oos, quality=quality, dropped=dropped, fx_info=fx_info,
        validation=val, params=p, bench_key=bench_key, bl_has_views=bl_has_views,
        audit=build_audit_record(params_snapshot, prices, weights, stats),
    )


# ═══════════════════════════════════════════════════════════════════════════
# ONGLETS
# ═══════════════════════════════════════════════════════════════════════════

def tab_summary(res: dict):
    p, stats, weights = res["params"], res["stats"], res["weights"]
    rf = p["rf"]
    main = ("Max Sharpe" if "Max Sharpe" in weights else
            "Min Variance" if "Min Variance" in weights else list(weights)[0])
    s = stats[main]

    section_title(f"Portefeuille affiché : {main}",
                  f"Net de frais · rééquilibrage {p['reb_label'].lower()} · "
                  f"{len(res['prices'])} séances")
    cards = [
        ("CAGR", fmt_pct(s["cagr"], 2, True), "rendement composé net"),
        ("Volatilité", fmt_pct(s["vol"]), "annualisée"),
        ("Sharpe", fmt_num(s["sharpe"], 2), f"rf = {rf*100:.2f} %"),
        ("Sortino", fmt_num(s["sortino"], 2), "risque baissier seul"),
        ("Drawdown max", fmt_pct(s["max_drawdown"]), f"{s['underwater_days']} j sous l'eau"),
        ("CVaR 95 %", fmt_pct(s["cvar95"]), "perte moyenne de queue, 1 j"),
        ("Rotation", f"{s['annual_turnover']*100:.0f} %", "par an"),
    ]
    cols = st.columns(len(cards))
    for col, (lab, val, note) in zip(cols, cards):
        color = RED if lab in ("Drawdown max", "CVaR 95 %") else GOLD
        col.markdown(kpi_card(lab, val, note, color), unsafe_allow_html=True)

    # ── Verdict hors échantillon : l'information la plus importante ───────
    if res["oos"] and "Equal weight (1/N)" in res["oos"]:
        naive = res["oos"]["Equal weight (1/N)"]
        target = main if main in res["oos"] else None
        if target:
            v = oos_verdict(res["oos"][target], naive, rf)
            beat, sig = v["beats_naive"], v["significant"]
            color, icon = ((GREEN, "✅") if beat and sig else
                           (ORANGE, "≈") if beat else (RED, "⚠️"))
            verdict = ("bat le 1/N de façon statistiquement significative" if beat and sig
                       else "devance le 1/N, mais l'écart n'est pas significatif" if beat
                       else "ne bat pas le portefeuille équipondéré 1/N")
            st.markdown(
                f'<div style="margin-top:18px;background:linear-gradient(135deg,#0D1828,#121F33);'
                f'border:1px solid {color}44;border-left:4px solid {color};border-radius:12px;'
                f'padding:16px 20px;"><div style="color:{color};font-weight:700;'
                f'font-size:0.92rem;">{icon} Hors échantillon, « {target} » {verdict}.</div>'
                f'<div style="color:#94A3B8;font-size:0.78rem;margin-top:6px;">'
                f'Sharpe {v["strategy"]["sharpe"]:.2f} contre {v["naive"]["sharpe"]:.2f} '
                f'pour le 1/N · écart {v["sharpe_gap"]:+.2f} · '
                f'p = {v["p_value"]:.3f} (test de Memmel). '
                f'Ce chiffre, contrairement à celui du bandeau ci-dessus, ne dépend '
                f'd\'aucune donnée future.</div></div>', unsafe_allow_html=True)

    # ── Fiabilité de l'estimation ────────────────────────────────────────
    section_title("Fiabilité de l'estimation",
                  "Ce que l'optimiseur sait réellement — et ce qu'il devine")
    d = res["diag"]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Observations", f"{d['t_obs']:,}".replace(",", " "),
                         f"{d['t_obs']/PERIODS:.1f} ans"), unsafe_allow_html=True)
    freq_txt = {"D": "quotidienne", "W": "hebdomadaire", "M": "mensuelle"}[d["est_freq"]]
    c2.markdown(kpi_card("Shrinkage δ", f"{d['shrinkage_delta']*100:.0f} %",
                         f"Σ estimée en {freq_txt}"), unsafe_allow_html=True)
    med_noise = float(np.median(d["noise_ratio"]))
    c3.markdown(kpi_card("Bruit / signal sur μ", f"{med_noise:.1f}×",
                         "erreur-type ÷ estimation (médiane)",
                         RED if med_noise > 1 else ORANGE), unsafe_allow_html=True)
    c4.markdown(kpi_card("Lignes effectives", f"{effective_n(weights[main]):.1f}",
                         f"sur {len(res['assets'])} actifs"), unsafe_allow_html=True)
    if med_noise > 1:
        st.warning(
            "L'erreur d'estimation sur les rendements espérés dépasse les "
            "estimations elles-mêmes. Concrètement : les poids « Max Sharpe » "
            "reflètent surtout le hasard de la période retenue. Privilégiez "
            "Min Variance, Risk Parity ou le portefeuille rééchantillonné, et "
            "regardez d'abord l'onglet Hors échantillon.")

    if p["mu_method"] == "black_litterman" and not res.get("bl_has_views"):
        st.info(
            "Black-Litterman est actif mais aucune vue n'est exprimée : "
            "le portefeuille « Max Sharpe » reproduit alors exactement le "
            "portefeuille d'équilibre (équipondéré ou pondéré par les "
            "capitalisations). C'est le comportement attendu du modèle, pas un "
            "bug — saisissez des vues dans la barre latérale pour vous en écarter.")

    section_title("Comparaison des portefeuilles", "Toutes mesures nettes de frais")
    st.dataframe(_stats_table(stats, res["bench_key"]), use_container_width=True,
                 hide_index=True)
    if res["errors"]:
        with st.expander(f"⚠️ {len(res['errors'])} calcul(s) non abouti(s)", expanded=False):
            for k, v in res["errors"].items():
                st.markdown(f"**{k}** — {v}")


def _stats_table(stats: dict, bench_key: str) -> pd.DataFrame:
    rows = []
    for name, s in stats.items():
        rows.append({
            "Portefeuille": name,
            "CAGR": fmt_pct(s["cagr"], 2, True),
            "Volatilité": fmt_pct(s["vol"]),
            "Sharpe": fmt_num(s["sharpe"], 2),
            "Sortino": fmt_num(s["sortino"], 2),
            "Calmar": fmt_num(s["calmar"], 2),
            "DD max": fmt_pct(s["max_drawdown"]),
            "VaR 95 %": fmt_pct(s["var95"]),
            "CVaR 95 %": fmt_pct(s["cvar95"]),
            "Skew": fmt_num(s["skew"], 2),
            "Kurtosis": fmt_num(s["kurtosis"], 1),
            "Bêta": fmt_num(s.get("beta"), 2) if name != bench_key else "—",
            "Alpha": fmt_pct(s.get("alpha"), 2, True) if name != bench_key else "—",
            "TE": fmt_pct(s.get("tracking_error")) if name != bench_key else "—",
            "Rotation/an": f"{s['annual_turnover']*100:.0f} %",
        })
    return pd.DataFrame(rows)


def tab_frontier(res: dict):
    p = res["params"]
    if p["mu_method"] == "none":
        st.info("La frontière efficiente requiert une estimation de μ. "
                "Choisissez « Historique » ou « Black-Litterman » dans la barre latérale.")
    else:
        core = {k: v for k, v in res["weights"].items()
                if k not in ("Equal weight (1/N)", "Benchmark 60/40")}
        st.plotly_chart(chart_frontier(res["mu"], res["cov"], p["rf"], res["assets"],
                                       core, res["cloud"], res["frontier"]),
                        use_container_width=True)
        if res["frontier_failed"]:
            st.caption(f"{res['frontier_failed']} point(s) de la frontière étaient "
                       f"infaisables sous vos contraintes et ont été omis "
                       f"(l'ancienne version les supprimait sans le dire).")

    section_title("Incertitude sur les rendements espérés",
                  "La barre horizontale est l'intervalle de confiance à 95 %")
    st.plotly_chart(chart_mu_uncertainty(res["assets"], res["diag"]["mu_hist"],
                                         res["diag"]["se_mu"]),
                    use_container_width=True)
    st.caption("Quand la barre traverse le zéro, l'historique ne permet pas "
               "d'affirmer que l'actif a une espérance de rendement positive. "
               "C'est le cas le plus fréquent sur moins de dix ans de données.")

    if res["W_boot"] is not None:
        section_title("Stabilité des poids", "60 rééchantillonnages de l'historique")
        st.plotly_chart(chart_weight_uncertainty(res["assets"], res["W_boot"]),
                        use_container_width=True)
        spread = (res["W_boot"].max(axis=0) - res["W_boot"].min(axis=0)) * 100
        st.caption(f"Amplitude maximale d'un poids selon le tirage : "
                   f"{spread.max():.0f} points. Une amplitude large signifie que "
                   f"l'allocation « optimale » n'est pas identifiable à partir "
                   f"de ces données.")


def tab_allocations(res: dict):
    p, w_all = res["params"], res["weights"]
    show = [k for k in w_all if k != "Benchmark 60/40"]
    section_title("Répartition par portefeuille")
    for i in range(0, len(show), 2):
        cols = st.columns(2)
        for col, name in zip(cols, show[i:i + 2]):
            s = res["stats"][name]
            with col:
                st.plotly_chart(
                    chart_pie(w_all[name], res["assets"], name,
                              f"CAGR {fmt_pct(s['cagr'],1,True)} · vol {fmt_pct(s['vol'],1)}"),
                    use_container_width=True)

    section_title("Poids détaillés")
    df = pd.DataFrame({"Actif": res["assets"],
                       "Classe": [res["classes"][a] for a in res["assets"]]})
    for name, w in w_all.items():
        df[name] = [f"{x*100:.1f} %" for x in w]
    st.dataframe(df, use_container_width=True, hide_index=True)

    section_title("Trajectoires nettes de frais",
                  f"Rééquilibrage {p['reb_label'].lower()} — définition unique")
    navs = {k: res["stats"][k]["nav"] * 100 for k in w_all}
    st.plotly_chart(chart_oos(navs, "Performance cumulée (base 100)"),
                    use_container_width=True)
    st.caption("Attention : ces courbes sont construites sur la période qui a "
               "servi à l'optimisation. Elles ne mesurent pas une capacité "
               "prédictive — voir l'onglet Hors échantillon.")


def tab_risk(res: dict):
    p = res["params"]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_correlation(res["returns"]), use_container_width=True)
    with c2:
        core = {k: v for k, v in res["weights"].items() if k != "Benchmark 60/40"}
        st.plotly_chart(chart_risk_contrib(res["assets"], core, res["cov"]),
                        use_container_width=True)

    section_title("Risque de queue", "Ce que la volatilité ne montre pas")
    st.plotly_chart(chart_tail({k: v["net_returns"] for k, v in res["stats"].items()}),
                    use_container_width=True)

    section_title("Drawdowns")
    st.plotly_chart(chart_drawdown({k: v["nav"] for k, v in res["stats"].items()}),
                    use_container_width=True)

    section_title("Diversification réelle")
    rows = []
    for name, w in res["weights"].items():
        rows.append({"Portefeuille": name,
                     "Lignes effectives": f"{effective_n(w):.1f}",
                     "Ratio de diversification": f"{diversification_ratio(w, res['cov']):.2f}",
                     "Poids max": f"{np.max(w)*100:.1f} %",
                     "Contribution au risque max":
                         f"{np.max(risk_contributions(w, res['cov'])):.0f} %"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    section_title("Exposition factorielle",
                  "Le portefeuille est-il diversifié, ou concentré sur un facteur ?")
    if not HAS_YF:
        st.info("yfinance requis pour télécharger les proxies factoriels.")
        return
    if st.button("Lancer la régression factorielle", key="run_factor"):
        try:
            with st.spinner("Téléchargement des proxies factoriels…"):
                fp = fetch_raw_prices(tuple(FACTOR_PROXIES.values()),
                                      str(p["start"]), str(p["end"]))
                fp = fp.rename(columns={v: k for k, v in FACTOR_PROXIES.items()})
                fr = simple_returns(fp)
            main = "Max Sharpe" if "Max Sharpe" in res["stats"] else list(res["stats"])[0]
            fac = factor_regression(res["stats"][main]["net_returns"], fr, p["rf"])
            st.plotly_chart(chart_factors(fac), use_container_width=True)
            st.dataframe(fac, use_container_width=True, hide_index=True)
            st.caption("Un bêta marché proche de 1 avec un R² élevé signifie que "
                       "le portefeuille reproduit l'indice : l'optimisation n'a "
                       "alors rien apporté qu'un ETF large n'offrirait moins cher.")
        except PortfolioLabError as e:
            st.error(str(e))


def tab_oos(res: dict):
    p = res["params"]
    section_title("Backtest hors échantillon",
                  f"Fenêtre glissante de {p['lookback']:.1f} ans, ré-optimisation "
                  f"{'mensuelle' if p['oos_freq']=='M' else 'trimestrielle' if p['oos_freq']=='Q' else 'annuelle'}")
    if not p["do_oos"]:
        st.info("Activez « Backtest hors échantillon » dans la barre latérale.")
        return
    if not res["oos"]:
        st.error("Aucun backtest n'a abouti. " +
                 " ".join(v for k, v in res["errors"].items() if k.startswith("OOS")))
        return

    st.markdown(
        '<div style="background:rgba(59,130,246,0.07);border:1px solid rgba(59,130,246,0.28);'
        'border-radius:8px;padding:11px 15px;color:#93C5FD;font-size:0.8rem;">'
        'À chaque date de rééquilibrage, les poids sont recalculés en n\'utilisant '
        'que les données antérieures, puis appliqués à la période suivante. '
        'Aucune information future n\'entre dans le calcul, frais compris.</div>',
        unsafe_allow_html=True)

    navs = {k: v["nav"] for k, v in res["oos"].items()}
    st.plotly_chart(chart_oos(navs, "Performance hors échantillon (base 1)"),
                    use_container_width=True)

    rows = []
    naive = res["oos"].get("Equal weight (1/N)")
    for name, bt in res["oos"].items():
        s = full_stats(bt["nav"], bt["returns"], p["rf"])
        row = {"Stratégie": name, "CAGR": fmt_pct(s["cagr"], 2, True),
               "Volatilité": fmt_pct(s["vol"]), "Sharpe": fmt_num(s["sharpe"], 2),
               "Sortino": fmt_num(s["sortino"], 2), "DD max": fmt_pct(s["max_drawdown"]),
               "CVaR 95 %": fmt_pct(s["cvar95"]),
               "Rotation/an": f"{bt['annual_turnover']*100:.0f} %",
               "Ré-optimisations": bt["n_rebalances"],
               "Échecs solveur": bt["n_failed"]}
        if naive is not None and name != "Equal weight (1/N)":
            v = oos_verdict(bt, naive, p["rf"])
            row["Écart Sharpe vs 1/N"] = f"{v['sharpe_gap']:+.2f}"
            row["p-value"] = fmt_num(v["p_value"], 3)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption("Rappel méthodologique : DeMiguel, Garlappi et Uppal (2009) montrent "
               "que le 1/N naïf bat l'optimisation moyenne-variance hors échantillon "
               "dans la plupart des jeux de données. Si votre stratégie ne le bat pas "
               "ici, l'optimisation ne vous apporte rien sur cet univers.")

    section_title("Rotation du portefeuille", "Chaque rotation coûte des frais")
    for name, bt in res["oos"].items():
        with st.expander(f"Historique des poids — {name}", expanded=False):
            st.dataframe((bt["weights"] * 100).round(1), use_container_width=True)


def tab_scenarios(res: dict):
    p, assets = res["params"], res["assets"]
    section_title("Scénarios de stress",
                  "Comportement du portefeuille sur des périodes de crise passées")
    st.markdown(
        '<div style="background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.3);'
        'border-radius:8px;padding:11px 15px;color:#FCD34D;font-size:0.8rem;">'
        'Biais assumé : ces poids ont été optimisés sur votre période d\'analyse, '
        'pas avant la crise. Personne n\'aurait détenu ce portefeuille à l\'époque. '
        'Lisez ces résultats comme un test de sensibilité, pas comme un backtest.</div>',
        unsafe_allow_html=True)

    # P1.4 — le scénario personnalisé vit dans la session, jamais dans le global.
    if "custom_scenario" not in st.session_state:
        st.session_state.custom_scenario = None

    names = list(SCENARIO_PRESETS)
    chosen = st.multiselect("Scénarios", names, default=names[:2], key="sc_pick")

    with st.expander("Ajouter un scénario personnalisé", expanded=False):
        c1, c2, c3 = st.columns(3)
        cs = c1.date_input("Début", date(2018, 9, 1), key="cs_start")
        ce = c2.date_input("Fin", date(2018, 12, 31), key="cs_end")
        cn = c3.text_input("Libellé", "Mon scénario", key="cs_name")
        if st.button("Enregistrer le scénario", key="cs_save"):
            if cs >= ce:
                st.error("La date de début doit précéder la date de fin.")
            else:
                st.session_state.custom_scenario = {
                    "name": cn, "start": str(cs), "end": str(ce),
                    "color": GOLD, "description": "Période définie par l'utilisateur"}
                st.success(f"Scénario « {cn} » enregistré.")

    scenarios = {n: SCENARIO_PRESETS[n] for n in chosen}
    cust = st.session_state.custom_scenario
    if cust:
        scenarios[cust["name"]] = cust
    if not scenarios:
        st.info("Sélectionnez au moins un scénario.")
        return

    core = {k: v for k, v in res["weights"].items() if k != "Benchmark 60/40"}
    for name, meta in scenarios.items():
        with st.spinner(f"Téléchargement — {name}…"):
            try:
                raw = fetch_raw_prices(tuple(assets), meta["start"], meta["end"])
                ccy = {a: res["validation"][a].get("currency", "USD") for a in assets
                       if a in raw.columns}
                fx = None
                if any(c != p["base_ccy"] for c in ccy.values()):
                    try:
                        fx = fetch_fx_series(tuple(sorted(set(ccy.values()))),
                                             p["base_ccy"], meta["start"], meta["end"])
                    except FXError:
                        fx = None
                sc_prices, _ = convert_to_base(raw, ccy, p["base_ccy"], fx, True)
            except DataError as e:
                st.warning(f"**{name}** — {e}")
                continue

        sc = run_scenario(sc_prices, core, assets, name, meta,
                          rebalance=p["rebalance"], costs=p["costs"])
        if not sc or not sc.get("portfolios"):
            st.warning(
                f"**{name}** — aucune donnée exploitable entre {meta['start']} et "
                f"{meta['end']}. Causes usuelles : les actifs n'existaient pas "
                f"encore (crypto avant 2017, introductions récentes), ou la source "
                f"n'a rien renvoyé. Ajoutez un actif ancien (SPY, AAPL) pour "
                f"vérifier la disponibilité des données sur cette fenêtre.")
            continue
        if sc["missing"]:
            st.info(f"ℹ️ **{name}** : pas d'historique pour "
                    f"{', '.join(sc['missing'])}. Calcul effectué sur "
                    f"{', '.join(sc['available'])}, poids re-normalisés à 100 %. "
                    f"Les résultats ne décrivent donc pas votre portefeuille complet.")

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;'
            f'background:linear-gradient(135deg,#0D1828,#121F33);'
            f'border:1px solid {meta["color"]}33;border-radius:12px;padding:14px 18px;'
            f'margin:16px 0 10px 0;"><div style="width:4px;height:38px;border-radius:2px;'
            f'background:{meta["color"]};"></div><div>'
            f'<div style="font-size:1rem;font-weight:600;color:#EEF2F7;">{name}</div>'
            f'<div style="font-size:0.72rem;color:#64748B;margin-top:2px;">'
            f'{meta["start"]} → {meta["end"]} · {len(sc["prices"])} séances · '
            f'{meta["description"]}</div></div></div>', unsafe_allow_html=True)

        cols = st.columns(len(sc["portfolios"]))
        for col, (pname, s) in zip(cols, sc["portfolios"].items()):
            col.markdown(kpi_card(
                pname, fmt_pct(s["total_return"], 1, True),
                f"DD max {fmt_pct(s['max_drawdown'],1)} · CVaR {fmt_pct(s['cvar95'],1)}",
                GREEN if s["total_return"] >= 0 else RED), unsafe_allow_html=True)

        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(chart_scenario_cum(sc), use_container_width=True)
        c2.plotly_chart(chart_scenario_bars(sc), use_container_width=True)
        with st.expander("Performance par actif", expanded=False):
            st.dataframe(pd.DataFrame([
                {"Actif": a, "Rendement": fmt_pct(v["total_return"], 1, True),
                 "DD max": fmt_pct(v["max_drawdown"], 1)}
                for a, v in sc["assets"].items()]),
                use_container_width=True, hide_index=True)
        st.markdown("---")


def tab_projection(res: dict):
    p = res["params"]
    sym = p["sym"]
    section_title("Projection avec versements programmés",
                  "Bootstrap par blocs sur les rendements mensuels historiques")
    st.markdown(
        '<div style="background:rgba(212,175,55,0.06);border:1px solid rgba(212,175,55,0.25);'
        'border-radius:8px;padding:11px 15px;color:#E5C76B;font-size:0.8rem;">'
        'Hypothèse structurante : l\'avenir ressemble en distribution au passé '
        'observé sur votre période d\'analyse. Les blocs de 6 mois préservent '
        'l\'autocorrélation et les regroupements de volatilité, mais aucune '
        'simulation ne contient de crise plus sévère que celles déjà présentes '
        'dans l\'historique.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    pname = c1.selectbox("Portefeuille projeté", list(res["weights"]),
                         index=0, key="proj_port")
    n_sims = c2.select_slider("Simulations", [1000, 2500, 5000, 10000],
                              value=5000, key="proj_sims")
    block = c3.slider("Taille de bloc (mois)", 1, 24, 6, key="proj_block")

    monthly_r = to_monthly(res["stats"][pname]["net_returns"])
    tax_rate = TAX_REGIMES[p["tax"]]["rate"]
    try:
        sim = simulate_wealth(monthly_r, initial=p["initial"],
                              monthly_contribution=p["monthly"],
                              horizon_years=p["horizon"], n_sims=int(n_sims),
                              block=int(block), inflation=p["inflation"],
                              tax_rate=tax_rate, seed=SEED)
    except (DataError, ValueError) as e:
        st.error(str(e))
        return

    q = sim["quantiles"]
    cards = [
        ("Capital versé", f"{sym}{sim['invested']:,.0f}",
         f"{p['initial']:,.0f} + {p['monthly']:,.0f}/mois"),
        ("Médiane", f"{sym}{q[0.5]:,.0f}", f"sur {p['horizon']} ans"),
        ("Scénario défavorable", f"{sym}{q[0.05]:,.0f}", "5e centile", RED),
        ("Scénario favorable", f"{sym}{q[0.95]:,.0f}", "95e centile", GREEN),
        ("Perte en capital", f"{sim['prob_loss']*100:.0f} %",
         "probabilité de finir sous les versements",
         RED if sim["prob_loss"] > 0.15 else ORANGE),
        ("TRI médian", fmt_pct(sim["median_irr"], 2, True), "pondéré par les flux"),
    ]
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        lab, val, note = card[0], card[1], card[2]
        col.markdown(kpi_card(lab, val, note, card[3] if len(card) > 3 else GOLD),
                     unsafe_allow_html=True)

    invested_path = p["initial"] + p["monthly"] * np.arange(sim["horizon_months"] + 1)
    st.plotly_chart(chart_fan(sim["trajectories"], invested_path, sym),
                    use_container_width=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        goal = p["goal"] if p["goal"] > 0 else None
        st.plotly_chart(chart_terminal_hist(sim["terminal"], sim["invested"], goal, sym),
                        use_container_width=True)
    with c2:
        section_title("Après impôt et inflation")
        qr = sim["quantiles_real"]
        st.dataframe(pd.DataFrame({
            "Centile": ["5e", "25e", "Médiane", "75e", "95e"],
            "Brut": [f"{sym}{q[x]:,.0f}" for x in (0.05, 0.25, 0.5, 0.75, 0.95)],
            "Net d'impôt et d'inflation":
                [f"{sym}{qr[x]:,.0f}" for x in (0.05, 0.25, 0.5, 0.75, 0.95)],
        }), use_container_width=True, hide_index=True)
        st.caption(f"Régime : {p['tax']} ({tax_rate*100:.1f} %). "
                   f"{TAX_REGIMES[p['tax']]['note']} Inflation "
                   f"{p['inflation']*100:.1f} %/an. Calcul indicatif, "
                   f"à confirmer auprès d'un conseil fiscal.")

    if p["goal"] > 0:
        prob = prob_reach_goal(sim["terminal"], p["goal"])
        color = GREEN if prob > 0.7 else ORANGE if prob > 0.4 else RED
        # Les séparateurs de milliers sont formatés sur les nombres seuls :
        # appliquer un remplacement global casserait les dégradés CSS.
        goal_txt = f"{p['goal']:,.0f}".replace(",", " ")
        sims_txt = f"{int(n_sims):,}".replace(",", " ")
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#0D1828,#121F33);'
            f'border-left:4px solid {color};border-radius:12px;padding:16px 20px;'
            f'margin-top:14px;"><div style="color:{color};font-weight:700;'
            f'font-size:1rem;">Probabilité d\'atteindre {sym}{goal_txt} '
            f'en {p["horizon"]} ans : {prob*100:.0f} %</div>'
            f'<div style="color:#94A3B8;font-size:0.78rem;margin-top:5px;">'
            f'Estimée sur {sims_txt} trajectoires simulées.</div></div>',
            unsafe_allow_html=True)

    section_title("Risque de parcours",
                  "Ce qu'il faudra supporter avant d'arriver au terme")
    dd = sim["max_drawdown"]
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi_card("Baisse médiane traversée", fmt_pct(float(np.median(dd)), 1),
                         "au pire moment du parcours", RED), unsafe_allow_html=True)
    c2.markdown(kpi_card("Baisse au 5e centile", fmt_pct(float(np.quantile(dd, 0.05)), 1),
                         "1 trajectoire sur 20 fait pire", RED), unsafe_allow_html=True)
    c3.markdown(kpi_card("Trajectoires perdant > 30 %",
                         f"{float((dd < -0.30).mean())*100:.0f} %",
                         "à un moment quelconque", ORANGE), unsafe_allow_html=True)
    st.caption("Un plan d'investissement n'échoue presque jamais sur les "
               "mathématiques : il échoue quand l'investisseur vend au creux. "
               "Vérifiez que ces baisses vous sont supportables.")


def tab_execution(res: dict):
    p, assets = res["params"], res["assets"]
    sym = p["sym"]
    section_title("Plan d'exécution", "Quantités réellement passables chez un courtier")

    pname = st.selectbox("Portefeuille à exécuter", list(res["weights"]), key="exec_port")
    w = res["weights"][pname]

    prices_base = {a: float(res["prices"][a].iloc[-1]) for a in assets}
    src = "dernière clôture de la période analysée"
    if HAS_YF and st.checkbox("Utiliser les cours du jour", value=False, key="use_live"):
        live = fetch_last_prices(tuple(assets))
        if live:
            prices_base.update({a: v for a, v in live.items()})
            src = "cours du jour (devise de cotation, non convertis)"
            st.caption("⚠️ Les cours du jour ne sont pas convertis en devise de "
                       "base : à n'utiliser que si tous les actifs cotent en "
                       f"{p['base_ccy']}.")

    plan = build_execution_plan(assets, w, p["initial"], prices_base,
                                allow_fractional=p["fractional"], costs=p["costs"])
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Capital investi", f"{sym}{plan['Montant investi'].sum():,.0f}",
                         f"{plan.attrs['taux_investi']*100:.1f} % du budget"),
                unsafe_allow_html=True)
    c2.markdown(kpi_card("Cash résiduel", f"{sym}{plan.attrs['cash_residuel']:,.0f}",
                         "non investissable en lots entiers",
                         ORANGE if plan.attrs["cash_residuel"] > p["initial"] * 0.05 else GOLD),
                unsafe_allow_html=True)
    c3.markdown(kpi_card("Frais d'entrée", f"{sym}{plan.attrs['frais_entree']:,.2f}",
                         f"{p['costs'].one_way*1e4:.0f} pb"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Écart maximal aux cibles",
                         f"{plan.attrs['ecart_max_pts']:.1f} pts",
                         "dû à l'arrondi", RED if plan.attrs["ecart_max_pts"] > 5 else GOLD),
                unsafe_allow_html=True)

    disp = plan.copy()
    disp["Poids cible"] = disp["Poids cible"].map(lambda x: f"{x*100:.1f} %")
    disp["Poids réel"] = disp["Poids réel"].map(lambda x: f"{x*100:.1f} %")
    for c in ("Montant cible", "Montant investi", "Prix"):
        disp[c] = disp[c].map(lambda x: f"{sym}{x:,.2f}" if np.isfinite(x) else "—")
    disp["Quantité"] = disp["Quantité"].map(
        lambda x: (f"{x:,.4f}" if p["fractional"] else f"{x:,.0f}")
        if np.isfinite(x) else "—")
    disp["Écart (pts)"] = disp["Écart (pts)"].map(lambda x: f"{x:+.2f}")
    st.dataframe(disp.style.map(delta_color, subset=["Écart (pts)"]),
                 use_container_width=True, hide_index=True)
    st.caption(f"Prix utilisés : {src}. "
               f"{'Fractions autorisées.' if p['fractional'] else 'Arrondi à l’unité.'}")
    if plan.attrs["prix_manquants"]:
        st.warning(f"Prix indisponible pour : {', '.join(plan.attrs['prix_manquants'])}.")

    st.plotly_chart(chart_alloc_bars(plan, sym), use_container_width=True)

    section_title("Enveloppe fiscale", "Indication France — à vérifier auprès du courtier")
    rows = []
    for a in assets:
        v = res["validation"].get(a, {})
        rows.append({"Actif": a, "Classe": res["classes"][a],
                     "Devise": v.get("currency", "—"),
                     "Place": v.get("exchange", "—"),
                     "PEA (indicatif)": "✅ probable" if pea_eligible(
                         v.get("exchange", ""), v.get("currency", ""),
                         v.get("asset_type", "EQUITY")) else "❌ improbable"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("L'éligibilité PEA dépend du siège de l'émetteur et non de la seule "
               "place de cotation : cette colonne est une présomption, pas une "
               "détermination.")

    st.download_button("⬇ Télécharger le plan d'exécution (CSV)",
                       plan.to_csv(index=False).encode("utf-8"),
                       file_name=f"plan_execution_{pname.replace(' ', '_')}.csv",
                       mime="text/csv")


def tab_data(res: dict):
    p = res["params"]
    section_title("Qualité des données", "Une erreur de données produit un optimum crédible et faux")
    st.dataframe(res["quality"], use_container_width=True, hide_index=True)
    if res["dropped"]:
        st.warning(f"Actifs écartés faute d'historique suffisant : "
                   f"{', '.join(res['dropped'])}.")

    info = res["fx_info"]
    if info["converted"]:
        msg = ("séries de change quotidiennes" if info["method"] == "series"
               else "taux statiques de repli — montants approximatifs")
        (st.success if info["method"] == "series" else st.warning)(
            f"Conversion en {info['base']} appliquée avant le calcul des "
            f"rendements pour {', '.join(info['converted'])} ({msg}).")
    else:
        st.info(f"Tous les actifs cotent déjà en {p['base_ccy']} : aucune conversion.")

    section_title("Validation des symboles")
    st.dataframe(pd.DataFrame([
        {"Symbole": t, "Statut": "✅ valide" if v.get("valid") else "❌ invalide",
         "Devise": v.get("currency", "—"), "Place": v.get("exchange", "—"),
         "Dernier cours": fmt_num(v.get("price"), 2),
         "Message": v.get("error", "")}
        for t, v in res["validation"].items()]),
        use_container_width=True, hide_index=True)
    if st.button("🔄 Revalider (vide le cache)", key="revalidate"):
        if HAS_ST:
            validate_ticker.clear()
        st.rerun()

    section_title("Contraintes effectivement appliquées",
                  "Celles du calcul, pas celles affichées dans la barre latérale")
    cons = res["cons"]                                          # P1.7
    st.dataframe(pd.DataFrame({
        "Actif": res["assets"],
        "Classe": [res["classes"][a] for a in res["assets"]],
        "Poids min": [f"{x*100:.0f} %" for x in cons.min_w],
        "Poids max": [f"{x*100:.0f} %" for x in cons.max_w],
    }), use_container_width=True, hide_index=True)
    if cons.groups:
        st.dataframe(pd.DataFrame([
            {"Groupe": g, "Actifs": ", ".join(res["assets"][i] for i in idx),
             "Plafond": f"{gmax*100:.0f} %"}
            for g, (idx, _, gmax) in cons.groups.items()]),
            use_container_width=True, hide_index=True)

    section_title("Journal d'audit", "Permet de rejouer et de justifier une allocation")
    st.json(res["audit"], expanded=False)
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Journal d'audit (JSON)",
                       json.dumps(res["audit"], indent=2, ensure_ascii=False).encode(),
                       file_name=f"audit_{res['audit']['run_id']}.json",
                       mime="application/json")
    export = pd.DataFrame({"Actif": res["assets"],
                           "Classe": [res["classes"][a] for a in res["assets"]],
                           "Rendement annuel estimé": res["mu"],
                           "Erreur-type sur μ": res["diag"]["se_mu"],
                           "Volatilité annuelle": np.sqrt(np.diag(res["cov"]))})
    for name, w in res["weights"].items():
        export[f"Poids {name}"] = w
    c2.download_button("⬇ Résultats complets (CSV)",
                       export.to_csv(index=False).encode("utf-8"),
                       file_name="portfolio_lab_resultats.csv", mime="text/csv")

    st.markdown("""
<div style="margin-top:32px;padding:20px;border-top:1px solid rgba(212,175,55,0.12);
 color:#64748B;font-size:0.73rem;line-height:1.7;">
<strong style="color:#94A3B8;">Limites connues et assumées</strong><br>
• Source de données non contractuelle (yfinance) : ruptures, ajustements de
splits imparfaits, absence de titres radiés — les backtests sont donc affectés
d'un biais du survivant favorable.<br>
• Les rendements espérés estimés sur l'historique restent bruités même après
shrinkage et rééchantillonnage.<br>
• La cardinalité maximale est résolue par heuristique, non à l'optimum exact.<br>
• L'éligibilité PEA et le calcul fiscal sont indicatifs.<br>
• Aucune prise en compte des dividendes non réinvestis, des retenues à la
source étrangères, ni de la liquidité des titres.<br><br>
PORTFOLIO LAB · d'après Portfolio Optimizer (c) AEG<br>
<span style="color:rgba(212,175,55,0.45);">Simulation à but pédagogique — ne
constitue pas un conseil en investissement.</span>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def landing():
    st.markdown("""
<div style="background:linear-gradient(135deg,#0D1828,#121F33);
 border:1px solid rgba(212,175,55,0.18);border-radius:16px;padding:36px;margin-top:16px;">
 <div style="font-size:2.4rem;margin-bottom:12px;">📊</div>
 <div style="font-size:1.1rem;font-weight:600;color:#EEF2F7;margin-bottom:10px;">
  Prêt à analyser un portefeuille</div>
 <div style="color:#94A3B8;font-size:0.85rem;max-width:640px;line-height:1.75;">
  Renseignez votre profil, votre devise et vos actifs dans la barre latérale,
  acceptez l'avertissement, puis lancez l'analyse.<br><br>
  <strong style="color:#D4AF37;">Ce qui différencie cet outil d'un optimiseur classique :</strong>
  la performance est mesurée hors échantillon et comparée à un portefeuille
  équipondéré, l'incertitude d'estimation est affichée plutôt que masquée,
  et les frais, l'arrondi aux lots et la fiscalité sont intégrés au calcul.
 </div></div>""", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Portfolio Lab", page_icon="📈", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    page_header()
    legal_banner()

    p = render_sidebar()

    if p["run"]:
        status = st.status("Analyse en cours…", expanded=True)
        try:
            res = run_analysis(p, lambda m: status.write(m))
            st.session_state.res = res
            status.update(label=f"Analyse terminée · run {res['audit']['run_id']}",
                          state="complete", expanded=False)
        except PortfolioLabError as e:
            status.update(label="Analyse interrompue", state="error", expanded=False)
            st.error(f"**Analyse interrompue.** {e}")
            st.stop()
        except Exception as e:                                   # pragma: no cover
            status.update(label="Erreur inattendue", state="error", expanded=False)
            st.error(f"**Erreur inattendue :** {type(e).__name__} — {e}")
            st.stop()

    if "res" not in st.session_state:
        landing()
        return

    res = st.session_state.res
    tabs = st.tabs(["  📋  Synthèse  ", "  📊  Frontière  ", "  🥧  Allocations  ",
                    "  🔬  Risque  ", "  🎯  Hors échantillon  ", "  🌪️  Scénarios  ",
                    "  🔮  Projection  ", "  💰  Exécution  ", "  🗂️  Données & audit  "])
    for tab, fn in zip(tabs, [tab_summary, tab_frontier, tab_allocations, tab_risk,
                              tab_oos, tab_scenarios, tab_projection, tab_execution,
                              tab_data]):
        with tab:
            try:
                fn(res)
            except PortfolioLabError as e:
                st.error(str(e))
            except Exception as e:                               # pragma: no cover
                st.error(f"Erreur dans cet onglet : {type(e).__name__} — {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TESTS UNITAIRES  (P4.4)  —  python portfolio_lab.py --selftest
# ═══════════════════════════════════════════════════════════════════════════
# Sur un outil financier, c'est le point le plus critique de l'architecture :
# une erreur de signe ou d'annualisation produit un résultat parfaitement
# présentable et parfaitement faux.

def _toy_returns(n_assets=3, T=1000, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=T)
    mus = np.linspace(0.08, 0.02, n_assets) / PERIODS
    sds = np.linspace(0.20, 0.05, n_assets) / np.sqrt(PERIODS)
    data = rng.normal(mus, sds, size=(T, n_assets))
    return pd.DataFrame(data, index=idx,
                        columns=[f"A{i}" for i in range(n_assets)])


def run_self_tests(verbose: bool = True) -> int:
    checks: list[tuple[str, Callable[[], None]]] = []
    ok, failed = 0, []

    def test(name):
        def deco(fn):
            checks.append((name, fn))
            return fn
        return deco

    # ── Métriques élémentaires ───────────────────────────────────────────
    @test("port_vol avec covariance identité")
    def _():
        w = np.array([0.6, 0.4])
        cov = np.eye(2) * 0.04
        assert abs(port_vol(w, cov) - np.sqrt(0.04 * (0.36 + 0.16))) < 1e-12

    @test("max_drawdown sur série connue")
    def _():
        nav = pd.Series([100, 120, 60, 90], index=pd.bdate_range("2020-01-01", periods=4))
        assert abs(max_drawdown(nav) - (-0.5)) < 1e-12

    @test("underwater_days compte les séances sous le sommet")
    def _():
        nav = pd.Series([100, 90, 95, 105, 100],
                        index=pd.bdate_range("2020-01-01", periods=5))
        assert underwater_days(nav) == 2

    @test("CAGR d'un doublement en 2 ans ≈ 41,4 %")
    def _():
        nav = pd.Series(np.linspace(1, 2, 2 * PERIODS),
                        index=pd.bdate_range("2020-01-01", periods=2 * PERIODS))
        assert abs(cagr(nav) - (2 ** 0.5 - 1)) < 0.02

    @test("CVaR ≥ VaR (queue plus lourde que le quantile)")
    def _():
        r = pd.Series(np.random.default_rng(1).standard_t(4, 3000) / 100)
        assert historical_cvar(r, 0.95) >= historical_var(r, 0.95)

    @test("Sortino ignore la volatilité haussière")
    def _():
        rng = np.random.default_rng(3)
        base = pd.Series(rng.normal(0.0004, 0.01, 2000))
        boosted = base.copy()
        boosted[boosted > 0] *= 3.0          # plus de volatilité, mais à la hausse
        assert sortino(boosted) > sortino(base)

    @test("risk_contributions somme à 100 %")
    def _():
        rng = np.random.default_rng(5)
        A = rng.normal(size=(4, 4))
        cov = A @ A.T / 10
        w = np.array([0.4, 0.3, 0.2, 0.1])
        assert abs(risk_contributions(w, cov).sum() - 100.0) < 1e-8

    # ── Estimateurs ──────────────────────────────────────────────────────
    @test("Ledoit-Wolf : δ ∈ [0,1] et matrice définie positive")
    def _():
        cov, delta = ledoit_wolf_cov(_toy_returns(4, 300))
        assert 0.0 <= delta <= 1.0
        assert np.all(np.linalg.eigvalsh(cov) > -1e-10)
        assert np.allclose(cov, cov.T)

    @test("Ledoit-Wolf réduit la dispersion des valeurs propres")
    def _():
        r = _toy_returns(3, 120)              # T petit → shrinkage utile
        lw, _ = ledoit_wolf_cov(r)
        sm = sample_cov(r)
        assert np.linalg.cond(lw) <= np.linalg.cond(sm) + 1e-6

    @test("Estimation hebdomadaire : corrélation non biaisée par les fuseaux")
    def _():
        rng = np.random.default_rng(3)
        T = 3000
        f = rng.normal(0, 0.009, T)
        us = f + rng.normal(0, 0.005, T)
        eu = 0.55 * f + 0.45 * np.roll(f, 1) + rng.normal(0, 0.005, T)
        r = pd.DataFrame({"US": us, "EU": eu},
                         index=pd.bdate_range("2014-01-01", periods=T))
        _, cov_d, dd = estimate_moments(r, ModelSpec(cov_method="sample", est_freq="D"))
        _, cov_w, dw = estimate_moments(r, ModelSpec(cov_method="sample", est_freq="W"))
        to_corr = lambda c: c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])
        assert dd["est_freq"] == "D" and dw["est_freq"] == "W"
        assert dw["periods_per_year"] == 52
        # L'asynchronisme des clôtures écrase la corrélation quotidienne.
        assert to_corr(cov_w) > to_corr(cov_d) + 0.10
        # …et sous-estime la volatilité du marché décalé.
        assert np.sqrt(cov_w[1, 1]) > np.sqrt(cov_d[1, 1]) * 1.1

    @test("Repli automatique sur le quotidien si trop peu d'observations")
    def _():
        r = _toy_returns(2, 60)
        _, _, d = estimate_moments(r, ModelSpec(est_freq="M"))
        assert d["est_freq"] == "D"

    @test("to_monthly écarte les mois incomplets")
    def _():
        r = pd.Series(0.001, index=pd.bdate_range("2020-01-20", "2022-03-10"))
        m = to_monthly(r)
        assert len(m) == 25                      # 26 mois civils, 2 tronqués
        assert np.ptp(m) < 0.01                  # plus de mois « à 3 séances »

    @test("fetch_fx_series ne plante pas quand tout est déjà en USD")
    def _():
        import types as _t
        keep_yf, keep_has = yf, HAS_YF
        globals()["yf"] = _t.SimpleNamespace(download=lambda *a, **k: pd.DataFrame())
        globals()["HAS_YF"] = True
        try:
            out = fetch_fx_series(("USD",), "USD", "2020-01-01", "2021-01-01")
            assert out.empty
        finally:
            globals()["yf"], globals()["HAS_YF"] = keep_yf, keep_has

    @test("Correction géométrique ↔ arithmétique cohérente")
    def _():
        g, s = 0.07, 0.20
        assert abs(arithmetic_to_geometric(geometric_to_arithmetic(
            np.array([g]), np.array([s]))[0], s) - g) < 1e-12

    @test("Rendements simples, pas logarithmiques (μ non sous-estimé)")
    def _():
        prices = pd.DataFrame({"A": [100, 110, 121]},
                              index=pd.bdate_range("2020-01-01", periods=3))
        r = simple_returns(prices)["A"].values
        assert np.allclose(r, [0.10, 0.10])

    # ── Contraintes et optimisation ──────────────────────────────────────
    @test("Contraintes infaisables → InfeasibleConstraints (plus de repli 1/n)")
    def _():
        try:
            check_feasibility(Constraints(3, np.array([0.5, 0.5, 0.5]), np.ones(3)))
        except InfeasibleConstraints:
            return
        raise AssertionError("aucune erreur levée sur min_w impossible")

    @test("Plafonds insuffisants détectés")
    def _():
        try:
            check_feasibility(Constraints(3, np.zeros(3), np.full(3, 0.2)))
        except InfeasibleConstraints:
            return
        raise AssertionError("aucune erreur levée sur max_w impossible")

    @test("Contraintes de groupe incompatibles détectées")
    def _():
        cons = Constraints(3, np.zeros(3), np.ones(3),
                           {"g": ([0, 1, 2], 0.0, 0.5)})
        try:
            check_feasibility(cons)
        except InfeasibleConstraints:
            return
        raise AssertionError("aucune erreur levée sur groupe impossible")

    @test("Min variance respecte les bornes")
    def _():
        r = _toy_returns(3, 600)
        cov = sample_cov(r)
        cons = Constraints(3, np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5]))
        w = solve_min_variance(cov, cons)
        assert abs(w.sum() - 1) < 1e-6
        assert np.all(w >= 0.1 - 1e-4) and np.all(w <= 0.5 + 1e-4)

    @test("Min variance ≤ variance de tout autre portefeuille admissible")
    def _():
        r = _toy_returns(3, 600)
        cov = sample_cov(r)
        cons = Constraints(3)
        w = solve_min_variance(cov, cons)
        assert port_vol(w, cov) <= port_vol(np.ones(3) / 3, cov) + 1e-9

    @test("Risk parity égalise les contributions (actifs indépendants)")
    def _():
        cov = np.diag([0.04, 0.09, 0.01])
        w = solve_risk_parity(cov, Constraints(3))
        rc = risk_contributions(w, cov)
        assert rc.max() - rc.min() < 1.0     # écart < 1 point de pourcentage

    @test("Max Sharpe bat 1/N sur l'échantillon d'estimation (in-sample)")
    def _():
        r = _toy_returns(3, 800)
        mu, cov, _ = estimate_moments(r, ModelSpec(cov_method="sample"))
        cons = Constraints(3)
        w = solve_max_sharpe(mu, cov, cons, 0.0)
        assert sharpe(w, mu, cov, 0.0) >= sharpe(np.ones(3) / 3, mu, cov, 0.0) - 1e-6

    @test("Max Sharpe refuse de tourner sans μ")
    def _():
        try:
            solve_max_sharpe(np.zeros(3), np.eye(3) * 0.04, Constraints(3), 0.0)
        except OptimizationError:
            return
        raise AssertionError("aurait dû refuser un μ nul")

    @test("Bornes de rendement faisables sous plafonds")
    def _():
        mu = np.array([0.10, 0.05, 0.02])
        cons = Constraints(3, np.zeros(3), np.full(3, 0.4))
        lo, hi = feasible_return_range(mu, cons)
        assert hi < mu.max()                 # le plafond mord réellement
        assert abs(hi - (0.4 * 0.10 + 0.4 * 0.05 + 0.2 * 0.02)) < 1e-9
        assert lo < hi

    @test("Frontière efficiente croissante en rendement")
    def _():
        r = _toy_returns(3, 600)
        mu, cov, _ = estimate_moments(r, ModelSpec(cov_method="sample"))
        fv, fr, failed = efficient_frontier(mu, cov, Constraints(3), 12)
        assert len(fv) >= 5 and np.all(np.diff(fr) > -1e-9)

    @test("Min CVaR admissible et non dominé en CVaR")
    def _():
        r = _toy_returns(3, 400)
        cons = Constraints(3)
        w = solve_min_cvar(r, cons, 0.95)
        assert abs(w.sum() - 1) < 1e-6 and np.all(w >= -1e-8)
        pr = pd.Series(r.values @ w, index=r.index)
        eq = pd.Series(r.values @ (np.ones(3) / 3), index=r.index)
        assert historical_cvar(pr, 0.95) <= historical_cvar(eq, 0.95) + 1e-6

    @test("Black-Litterman sans vue redonne l'équilibre implicite")
    def _():
        cov = np.diag([0.04, 0.09])
        spec = ModelSpec(mu_method="black_litterman", bl_prior_w=np.array([0.6, 0.4]))
        mu = black_litterman_mu(cov, spec, 0.02)
        assert np.allclose(mu, 2.5 * (cov @ np.array([0.6, 0.4])) + 0.02)

    @test("Black-Litterman déplace μ vers la vue exprimée")
    def _():
        cov = np.diag([0.04, 0.09])
        base = ModelSpec(mu_method="black_litterman", bl_prior_w=np.array([0.5, 0.5]))
        pi = black_litterman_mu(cov, base, 0.0)
        viewed = ModelSpec(mu_method="black_litterman", bl_prior_w=np.array([0.5, 0.5]),
                           bl_views=np.array([0.25, np.nan]),
                           bl_confidence=np.array([0.9, 0.0]))
        post = black_litterman_mu(cov, viewed, 0.0)
        assert post[0] > pi[0]

    # ── Trajectoires, frais, backtest ────────────────────────────────────
    @test("Rendements nuls et frais nuls → NAV constante")
    def _():
        r = pd.DataFrame(0.0, index=pd.bdate_range("2020-01-01", periods=100),
                         columns=["A", "B"])
        path = portfolio_path(r, np.array([0.5, 0.5]),
                              costs=CostModel(0, 0, 0), charge_initial=False)
        assert abs(path["nav"].iloc[-1] - 1.0) < 1e-12

    @test("Les frais réduisent la NAV, jamais l'inverse")
    def _():
        r = _toy_returns(2, 500)
        free = portfolio_path(r, np.array([0.5, 0.5]), rebalance="M",
                              costs=CostModel(0, 0, 0))["nav"].iloc[-1]
        costly = portfolio_path(r, np.array([0.5, 0.5]), rebalance="M",
                                costs=CostModel(20, 10, 50))["nav"].iloc[-1]
        assert costly < free

    @test("Buy-and-hold ≠ rééquilibré : une seule définition doit être utilisée")
    def _():
        r = _toy_returns(2, 500)
        w = np.array([0.5, 0.5])
        bh = portfolio_path(r, w, rebalance="none", costs=CostModel(0, 0, 0))
        rb = portfolio_path(r, w, rebalance="D", costs=CostModel(0, 0, 0))
        assert abs(bh["nav"].iloc[-1] - rb["nav"].iloc[-1]) > 1e-6
        assert bh["annual_turnover"] == 0.0

    @test("Rendement total et drawdown proviennent de la même trajectoire")
    def _():
        r = _toy_returns(3, 400)
        w = np.array([0.5, 0.3, 0.2])
        path = portfolio_path(r, w, rebalance="M", costs=CostModel(0, 0, 0))
        s = full_stats(path["nav"], path["returns"], 0.0)
        assert abs(s["total_return"] - (path["nav"].iloc[-1] - 1)) < 1e-9

    @test("rebalance_flags : un seul déclenchement par mois")
    def _():
        idx = pd.bdate_range("2020-01-01", "2020-12-31")
        assert rebalance_flags(idx, "M").sum() == 11    # 11 fins de mois internes
        assert rebalance_flags(idx, "none").sum() == 0
        assert rebalance_flags(idx, "D").sum() == len(idx)

    @test("Walk-forward n'utilise aucune donnée future")
    def _():
        r = _toy_returns(3, 1400)
        cons = Constraints(3)
        bt = walk_forward_backtest(r, method="min_variance", cons=cons, rf=0.0,
                                   spec=ModelSpec(cov_method="sample"),
                                   lookback_years=2.0, reb_freq="Q",
                                   costs=CostModel(0, 0, 0))
        first_w_date = bt["weights"].index[0]
        # Les premiers poids ne peuvent pas être fixés avant d'avoir la
        # fenêtre d'estimation complète…
        assert first_w_date >= r.index[int(2.0 * PERIODS) - 1]
        # …et aucun rendement n'est comptabilisé avant leur fixation.
        assert bt["nav"].index[0] == first_w_date
        assert bt["returns"].index[0] > first_w_date

    @test("Walk-forward refuse un historique trop court")
    def _():
        try:
            walk_forward_backtest(_toy_returns(3, 200), method="min_variance",
                                  cons=Constraints(3), rf=0.0, spec=ModelSpec(),
                                  lookback_years=3.0)
        except DataError:
            return
        raise AssertionError("aurait dû refuser 200 séances")

    # ── Projection ───────────────────────────────────────────────────────
    @test("Bootstrap par blocs : dimensions et support corrects")
    def _():
        hist = np.array([0.01, -0.02, 0.03] * 20)
        paths = block_bootstrap_paths(hist, 60, n_sims=200, block=6)
        assert paths.shape == (200, 60)
        assert np.all(np.isin(np.round(paths, 10), np.round(hist, 10)))

    @test("Sans versement ni frais, le capital final suit la composition")
    def _():
        hist = np.full(60, 0.01)             # rendement mensuel déterministe
        sim = simulate_wealth(hist, initial=1000, monthly_contribution=0,
                              horizon_years=1, n_sims=50, block=3,
                              inflation=0.0, tax_rate=0.0)
        assert abs(np.median(sim["terminal"]) - 1000 * 1.01 ** 12) < 1e-6

    @test("TRI d'un versement unique doublé en 1 an = 100 %")
    def _():
        cf = np.concatenate([[-100.0], np.zeros(12)])
        assert abs(money_weighted_return(cf, 200.0) - 1.0) < 1e-6

    @test("Les versements mensuels augmentent le capital final")
    def _():
        hist = np.full(120, 0.005)
        a = simulate_wealth(hist, initial=1000, monthly_contribution=0,
                            horizon_years=5, n_sims=50, block=6)
        b = simulate_wealth(hist, initial=1000, monthly_contribution=100,
                            horizon_years=5, n_sims=50, block=6)
        assert np.median(b["terminal"]) > np.median(a["terminal"])

    @test("La fiscalité ne s'applique qu'aux plus-values")
    def _():
        hist = np.full(24, 0.0)
        sim = simulate_wealth(hist, initial=1000, monthly_contribution=0,
                              horizon_years=2, n_sims=20, block=6,
                              inflation=0.0, tax_rate=0.30)
        assert abs(np.median(sim["after_tax"]) - 1000) < 1e-6

    # ── Exécution ────────────────────────────────────────────────────────
    @test("Plan d'exécution : lots entiers et cash résiduel cohérent")
    def _():
        plan = build_execution_plan(["A", "B"], np.array([0.5, 0.5]), 1000,
                                    {"A": 300.0, "B": 70.0},
                                    allow_fractional=False, costs=CostModel(0, 0, 0))
        assert list(plan["Quantité"]) == [1.0, 7.0]
        assert abs(plan.attrs["cash_residuel"] - (1000 - 300 - 490)) < 1e-9

    @test("Fractions autorisées → poids cible atteint exactement")
    def _():
        plan = build_execution_plan(["A", "B"], np.array([0.7, 0.3]), 1000,
                                    {"A": 137.0, "B": 41.0},
                                    allow_fractional=True, costs=CostModel(0, 0, 0))
        assert plan.attrs["ecart_max_pts"] < 1e-6

    # ── Données ──────────────────────────────────────────────────────────
    @test("Conversion FX appliquée avant le calcul des rendements")
    def _():
        idx = pd.bdate_range("2020-01-01", periods=50)
        prices = pd.DataFrame({"US": np.linspace(100, 110, 50),
                               "EU": np.linspace(50, 55, 50)}, index=idx)
        fx = pd.DataFrame({"USD": np.linspace(0.90, 1.00, 50),
                           "EUR": np.ones(50)}, index=idx)
        conv, info = convert_to_base(prices, {"US": "USD", "EU": "EUR"}, "EUR", fx)
        assert info["method"] == "series"
        assert abs(conv["US"].iloc[0] - 90.0) < 1e-9
        assert np.allclose(conv["EU"].values, prices["EU"].values)
        # le rendement converti diffère du rendement local : c'est tout l'enjeu
        assert abs(conv["US"].pct_change().iloc[1] - prices["US"].pct_change().iloc[1]) > 1e-6

    @test("FX manquant → FXError, pas de repli silencieux")
    def _():
        idx = pd.bdate_range("2020-01-01", periods=10)
        prices = pd.DataFrame({"X": np.ones(10)}, index=idx)
        try:
            convert_to_base(prices, {"X": "JPY"}, "EUR", None, allow_static=False)
        except FXError:
            return
        raise AssertionError("aucune erreur levée sur FX manquant")

    @test("align_common_history tronque et signale les exclusions")
    def _():
        idx = pd.bdate_range("2018-01-01", periods=800)
        df = pd.DataFrame({"long": np.linspace(1, 2, 800),
                           "long2": np.linspace(1, 3, 800),
                           "court": [np.nan] * 700 + list(np.linspace(1, 1.1, 100))},
                          index=idx)
        sub, dropped = align_common_history(df, 252)
        assert "court" in dropped and len(sub.columns) == 2

    @test("Le rapport qualité détecte les prix figés")
    def _():
        idx = pd.bdate_range("2020-01-01", periods=400)
        df = pd.DataFrame({"figé": np.ones(400)}, index=idx)
        rep = data_quality_report(df)
        assert "figé" in rep.iloc[0]["Alertes"] or "variance" in rep.iloc[0]["Alertes"]

    @test("SCENARIO_PRESETS n'est jamais muté (isolation entre sessions)")
    def _():
        before = json.dumps(SCENARIO_PRESETS, sort_keys=True)
        _ = {k: v for k, v in SCENARIO_PRESETS.items()}
        assert json.dumps(SCENARIO_PRESETS, sort_keys=True) == before
        assert all(v["start"] is not None for v in SCENARIO_PRESETS.values())

    @test("Empreinte des données stable et sensible")
    def _():
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]},
                          index=pd.bdate_range("2020-01-01", periods=3))
        h1 = data_fingerprint(df)
        assert h1 == data_fingerprint(df.copy())
        df2 = df.copy(); df2.iloc[0, 0] = 1.0001
        assert h1 != data_fingerprint(df2)

    @test("Nuage aléatoire vectorisé : poids admissibles")
    def _():
        cons = Constraints(4)
        r, v, s = random_portfolio_cloud(np.array([0.08, 0.06, 0.04, 0.02]),
                                         np.eye(4) * 0.04, cons, n=500, rf=0.01)
        assert len(r) == len(v) == len(s) and np.all(v > 0)

    @test("classify_asset reconnaît les grandes familles")
    def _():
        assert classify_asset("AGG") == "Bond"
        assert classify_asset("GLD") == "Commodity"
        assert classify_asset("BTC-USD") == "Crypto"
        assert classify_asset("AAPL", "EQUITY") == "Equity"

    @test("60/40 construit uniquement si actions ET obligations présentes")
    def _():
        assets = ["SPY", "EFA", "AGG"]
        cls = {"SPY": "Equity", "EFA": "Equity", "AGG": "Bond"}
        w = sixty_forty_weights(assets, cls)
        assert abs(w[0] - 0.30) < 1e-12 and abs(w[2] - 0.40) < 1e-12
        assert sixty_forty_weights(["SPY"], {"SPY": "Equity"}) is None

    @test("Régression factorielle : bêta ≈ 1 sur soi-même")
    def _():
        r = _toy_returns(2, 500)
        fac = factor_regression(r["A0"], r[["A0"]].rename(columns={"A0": "Marché"}))
        beta = float(fac.loc[fac["Facteur"] == "Marché", "Coefficient"].iloc[0])
        assert abs(beta - 1.0) < 1e-6

    @test("Verdict hors échantillon : détecte une stratégie supérieure")
    def _():
        idx = pd.bdate_range("2020-01-01", periods=800)
        rng = np.random.default_rng(11)
        good = pd.Series(rng.normal(0.0008, 0.008, 800), index=idx)
        bad = pd.Series(rng.normal(0.0001, 0.012, 800), index=idx)
        v = oos_verdict({"nav": (1 + good).cumprod(), "returns": good},
                        {"nav": (1 + bad).cumprod(), "returns": bad}, 0.0)
        assert v["beats_naive"] and v["sharpe_gap"] > 0

    # ── Exécution des tests ──────────────────────────────────────────────
    for name, fn in checks:
        try:
            fn()
            ok += 1
            if verbose:
                print(f"  ✅  {name}")
        except Exception as e:
            failed.append((name, e))
            if verbose:
                print(f"  ❌  {name}\n        → {type(e).__name__}: {e}")

    if verbose:
        print(f"\n{ok}/{len(checks)} tests réussis"
              + (f", {len(failed)} en échec" if failed else ""))
    return 0 if not failed else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_self_tests())
    if not HAS_ST:
        print("Streamlit n'est pas installé.\n"
              "  pip install streamlit yfinance plotly\n"
              "  streamlit run portfolio_lab.py")
        sys.exit(1)
    main()
