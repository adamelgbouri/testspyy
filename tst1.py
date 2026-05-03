import 

def portfolio_var(returns, weights, confidence=0.95, n_simulations=10000):
    """
    Monte Carlo VaR for a multi-asset portfolio.
    returns: pd.DataFrame of daily returns
    weights: np.array of portfolio weights
    """
    np.random.seed(42)
    mean   = returns.mean().values
    cov    = returns.cov().values
    sim    = np.random.multivariate_normal(mean, cov, n_simulations)
    pnl    = sim @ weights
    var    = np.percentile(pnl, (1 - confidence) * 100)
    # add CVaR stress testng & historical cmprsn
    return var
 
np.random.seed(1)
fake_returns = pd.DataFrame(np.random.normal(0.001, 0.02, (252, 3)),
                             columns=["WTI", "Gold", "NatGas"])
weights = np.array([0.5, 0.3, 0.2])
print("\n2. Monte Carlo Portfolio VaR")
print(f"   95% 1-day VaR: {portfolio_var(fake_returns, weights)*100:.2f}%")
