from backend.risk import returns
import numpy as np
import pandas as pd



def portfolio_returns(returns, weights, benchmark_col="SPY"): 
    holdings = returns.drop(columns = benchmark_col)
    return (holdings * weights).sum(axis = 1)

def annualized_volatility(returns):
    return returns.std() * (252 **.5)

def stock_betas(returns, benchmark_col="SPY"):
    market_var = returns[benchmark_col].var()
    return returns.cov()[benchmark_col] / market_var

def portfolio_beta(returns, weights,benchmark_col="SPY"):
    betas = stock_betas(returns, benchmark_col).drop(benchmark_col)
    return (betas * weights).sum()

def stock_sharpe(returns, benchmark_col="SPY"):
    annual_rf = .046
    holdings = returns.drop(columns=benchmark_col)
    annualized_returns = holdings.mean() * 252
    return (annualized_returns - annual_rf) / annualized_volatility(holdings)

def portfolio_sharpe(returns, weights, benchmark_col="SPY"):
    annual_rf = .046
    port = portfolio_returns(returns, weights, benchmark_col="SPY")
    return (port.mean()*252 - annual_rf) / (port.std() * (252**.5))

def stock_max_drawdown(returns, benchmark_col="SPY"):
    holdings = returns.drop(columns = benchmark_col)
    cumulative_returns = (1 + holdings).cumprod() #for simple returns, additive for log returns
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    return drawdowns.min()    #will return in negative values unless wrapped in abs

def portfolio_max_drawdown(returns, weights, benchmark_col="SPY"):
    port = portfolio_returns(returns, weights, benchmark_col= "SPY")
    cumulative_returns = (1+ port).cumprod()
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    return drawdowns.min()    #will return in negative values unless wrapped in abs 

