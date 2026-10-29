import pandas as pd
import numpy as np

def sector_weight(weights, sector_map):
    return weights.groupby(sector_map).sum()


def risk_contributions(cov, weights):
    sigma = np.array(cov)
    w = np.array(weights.reindex(cov.columns))

    port_volatility = np.sqrt(w @ sigma @ w)

    marginal_contributions = pd.Series(np.dot(sigma, w) / port_volatility, index = cov.columns)

    risk_contributions = pd.Series(w * marginal_contributions, index = cov.columns)

    pct_contributions = pd.Series(risk_contributions / port_volatility, index = cov.columns)


    risk_df = pd.concat([weights, marginal_contributions, risk_contributions, pct_contributions], axis = 1, keys = ["weights", "marginal_contributions", "risk_contributions", "percentage_contributions"])


    return risk_df


def sector_risk_contributions(cov, weights, sector_map):
    risk_df = risk_contributions(cov, weights)
    rc = risk_df["risk_contributions"]
    return rc.groupby(sector_map).sum()


def beta_exposures(stock_betas, weights, benchmark_col = "SPY"):
    stock_betas = stock_betas.drop(benchmark_col)
    w = weights.reindex(stock_betas.index)
    return w * stock_betas


def market_r_squared(port_returns, benchmark):
    return port_returns.corr(benchmark) ** 2
