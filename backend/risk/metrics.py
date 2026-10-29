import pandas as pd
from config import RISK_FREE_RATE


def annualized_volatility(returns):
    return returns.std() * (252 ** .5)


def stock_betas(returns, benchmark_col="SPY"):
    market_var = returns[benchmark_col].var()
    return returns.cov()[benchmark_col] / market_var


def stock_sharpe(returns, benchmark_col="SPY"):
    holdings = returns.drop(columns=benchmark_col)
    annualized_returns = holdings.mean() * 252
    return (annualized_returns - RISK_FREE_RATE) / annualized_volatility(holdings)


def stock_max_drawdown(returns, benchmark_col="SPY"):
    holdings = returns.drop(columns=benchmark_col)
    cumulative_returns = (1 + holdings).cumprod()  # for simple returns, additive for log returns
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    return drawdowns.min()  # will return in negative values unless wrapped in abs
