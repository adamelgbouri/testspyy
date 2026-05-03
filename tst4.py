import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")
# 1. CAPM and Efficient Frontier
def capm_and_frontier(expected_returns, cov_matrix, n_portfolios=1000):
    """
    Generate random portfolios and compute the efficient frontier.
    """
    np.random.seed(42)
    n       = len(expected_returns)
    results = []
    for _ in range(n_portfolios):
        w   = np.random.dirichlet(np.ones(n))
        ret = w @ expected_returns
        vol = np.sqrt(w @ cov_matrix @ w)
        sharpe = ret / vol
        results.append({"return": ret, "vol": vol, "sharpe": sharpe,
                         "weights": w.tolist()})
    df = pd.DataFrame(results)
    best = df.loc[df["sharpe"].idxmax()]
    # TODO: add mean-variance optimisation (scipy.minimize)
    # TODO: add risk-parity portfolio
    # TODO: add Black-Litterman views
    return df, best
 
rets = np.array([0.08, 0.12, 0.10, 0.07])
cov  = np.array([[0.04, 0.01, 0.02, 0.00],
                  [0.01, 0.09, 0.03, 0.01],
                  [0.02, 0.03, 0.06, 0.01],
                  [0.00, 0.01, 0.01, 0.02]])
frontier, best_port = capm_and_frontier(rets, cov)
print("\n8. CAPM / Efficient Frontier")
print(f"   Max Sharpe ratio:  {best_port['sharpe']:.3f}")
print(f"   Expected return:   {best_port['return']:.1%}")
print(f"   Portfolio vol:     {best_port['vol']:.1%}")
 
 
# 2. Implied Volatility Surface
def implied_vol_surface(S, r, T_list, K_list, market_prices, option="call"):
    """
    Build an implied vol surface from market option prices.
    Returns a 2D array of implied vols indexed by (T, K).
    """
    def bs_price(sigma, S, K, T, r, option):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option == "call":
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
 
    surface = {}
    for T in T_list:
        surface[T] = {}
        for K in K_list:
            mkt = market_prices.get((T, K), None)
            if mkt is None:
                surface[T][K] = None
                continue
            try:
                iv = brentq(lambda s: bs_price(s, S, K, T, r, option) - mkt,
                            1e-4, 10.0, xtol=1e-8)
                surface[T][K] = round(iv * 100, 2)
            except ValueError:
                surface[T][K] = None
    # TODO: add SABR model fit
    # TODO: add SVI parametrisation
    # TODO: add arbitrage-free interpolation
    return surface
 
T_list = [0.25, 0.5, 1.0]
K_list = [90, 95, 100, 105, 110]
mkt_prices = {
    (0.25, 90): 11.2, (0.25, 95): 6.8,  (0.25,100): 3.5,
    (0.25,105): 1.5,  (0.25,110): 0.5,
    (0.5,  90): 13.1, (0.5,  95): 9.2,  (0.5, 100): 6.1,
    (0.5, 105): 3.8,  (0.5, 110): 2.2,
    (1.0,  90): 16.0, (1.0,  95): 12.5, (1.0, 100): 9.5,
    (1.0, 105): 7.1,  (1.0, 110): 5.2,
}
surf = implied_vol_surface(100, 0.04, T_list, K_list, mkt_prices)
print("\n9. Implied Volatility Surface")
print(f"   ATM 3M implied vol: {surf[0.25][100]}%")
print(f"   ATM 6M implied vol: {surf[0.5][100]}%")
print(f"   ATM 1Y implied vol: {surf[1.0][100]}%")
 
