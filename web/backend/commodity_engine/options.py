"""Black-76 European options on futures."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


@dataclass
class Black76:
    """European option on a futures contract priced under Black's 1976 model."""

    F: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: str = "call"

    def _d1_d2(self) -> Tuple[float, float]:
        if self.sigma <= 0 or self.T <= 0:
            return float("nan"), float("nan")
        d1 = (np.log(self.F / self.K) + 0.5 * self.sigma ** 2 * self.T) / (
            self.sigma * np.sqrt(self.T))
        return d1, d1 - self.sigma * np.sqrt(self.T)

    def price(self) -> float:
        d1, d2 = self._d1_d2()
        disc = np.exp(-self.r * self.T)
        if self.option_type == "call":
            return float(disc * (self.F * norm.cdf(d1) - self.K * norm.cdf(d2)))
        return float(disc * (self.K * norm.cdf(-d2) - self.F * norm.cdf(-d1)))

    def delta(self) -> float:
        d1, _ = self._d1_d2()
        disc = np.exp(-self.r * self.T)
        return float(disc * (norm.cdf(d1) if self.option_type == "call"
                              else norm.cdf(d1) - 1))

    def gamma(self) -> float:
        d1, _ = self._d1_d2()
        return float(np.exp(-self.r * self.T) * norm.pdf(d1)
                      / (self.F * self.sigma * np.sqrt(self.T)))

    def vega(self) -> float:
        """Per 1 vol point (i.e. 0.01)."""
        d1, _ = self._d1_d2()
        return float(np.exp(-self.r * self.T) * self.F * norm.pdf(d1)
                      * np.sqrt(self.T) / 100)

    def theta(self) -> float:
        """Per day."""
        d1, d2 = self._d1_d2()
        disc = np.exp(-self.r * self.T)
        t1 = -disc * self.F * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))
        if self.option_type == "call":
            t2 = -self.r * disc * (self.F * norm.cdf(d1) - self.K * norm.cdf(d2))
        else:
            t2 = self.r * disc * (self.K * norm.cdf(-d2) - self.F * norm.cdf(-d1))
        return float((t1 + t2) / 365)

    def rho(self) -> float:
        return float(-self.T * self.price())

    def greeks(self) -> Dict[str, float]:
        return {
            "price": self.price(), "delta": self.delta(),
            "gamma": self.gamma(), "vega": self.vega(),
            "theta": self.theta(), "rho": self.rho(),
        }

    def implied_vol(self, market_price: float,
                    lo: float = 0.001, hi: float = 5.0) -> float:
        try:
            return float(brentq(
                lambda s: Black76(self.F, self.K, self.T, self.r, s,
                                   self.option_type).price() - market_price,
                lo, hi, maxiter=200,
            ))
        except Exception:
            return float("nan")
