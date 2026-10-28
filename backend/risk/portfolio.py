from backend.risk import metrics
import numpy as np

class Portfolio:
    def __init__(self, returns, weights, benchmark_col="SPY"):
        self.returns = returns
        self.benchmark_col = benchmark_col
        self.weights = weights
        self.benchmark = returns[benchmark_col]
        self.holdings = returns.drop(columns=benchmark_col)   # computed once
        self.port_returns = (self.holdings * weights).sum(axis=1)  # shared, once

    def volatility(self):
        return metrics.annualized_volatility(self.port_returns)

    def beta(self):
        stock_betas = metrics.stock_betas(self.returns, self.benchmark_col)
        stock_betas = stock_betas.drop(self.benchmark_col)
        return (stock_betas * self.weights).sum()

    def sharpe(self):
        annual_rf = .046
        annualized_returns = self.port_returns.mean() * 252
        return (annualized_returns - annual_rf) / self.volatility()

    def max_drawdown(self):
        cumulative_returns = (1 + self.port_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        return drawdowns.min()
    
    def volatility_from_cov(self):
        cov_matrix = self.holdings.cov()
        w = self.weights.reindex(cov_matrix.columns)
        daily_port_var = (w @ cov_matrix @ w)
        daily_vol = np.sqrt(daily_port_var)
        return daily_vol * (252 ** .5)
    
    def diversification_ratio(self):
        individualized_volatility = metrics.annualized_volatility(self.holdings) * self.weights
        weighted_avg_vols = individualized_volatility.sum()
        portfolio_vol = self.volatility_from_cov()
        return weighted_avg_vols / portfolio_vol
    
    def var_historical(self, confidence=0.95):
        return var.historical_var(self.port_returns, confidence)

    def var_parametric(self, confidence=0.95):
        return var.parametric_var(self.port_returns, confidence)

    def var_monte_carlo(self, confidence=0.95, n_sims=10000):
        return var.monte_carlo_var(self.holdings, self.weights, confidence, n_sims)

    def cvar(self, confidence=0.95):
        return var.cvar(self.port_returns, confidence)