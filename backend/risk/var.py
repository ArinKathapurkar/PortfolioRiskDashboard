#used to calculate value at risk / conditional value at risk
import numpy as np
from scipy.stats import norm

def historical_var(port_returns, confidence):
    return np.percentile(port_returns, 100 - (confidence * 100))


def parametric_var(port_returns, confidence):
    mean = port_returns.mean()
    std = port_returns.std()
    z = norm.ppf(1 - confidence)
    return mean + z * std

def simulate_port_returns(holdings, weights, simulations=10000, cov=None, seed=None):
    rng = np.random.default_rng(seed)
    mean_vector = holdings.mean().values
    cov_matrix = holdings.cov().values if cov is None else cov.values
    w = weights.reindex(holdings.columns).values
    sims = rng.multivariate_normal(mean_vector, cov_matrix, size=simulations)
    return sims @ w

def monte_carlo_var(holdings, weights, confidence, simulations=10000, cov=None, seed=None):
    sims = simulate_port_returns(holdings, weights, simulations, cov=cov, seed=seed)
    return historical_var(sims, confidence)

def cvar(returns, confidence=.95):
    var = historical_var(returns, confidence)
    return returns[returns <= var].mean()


def parametric_cvar(port_returns, confidence=.95):
    z = norm.ppf(1 - confidence)
    return port_returns.mean() - port_returns.std() * norm.pdf(z) / (1 - confidence)
