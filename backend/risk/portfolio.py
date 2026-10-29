from functools import cached_property
from backend.risk import metrics
import numpy as np
from backend.risk import var
from backend.risk import covariance
from config import RISK_FREE_RATE

class Portfolio:
    def __init__(self, returns, weights, benchmark_col="SPY"):
        self.returns = returns
        self.benchmark_col = benchmark_col
        self.weights = weights
        self.benchmark = returns[benchmark_col]
        self.holdings = returns.drop(columns=benchmark_col)
        self.port_returns = (self.holdings * weights).sum(axis=1)

        #port_returns --> series

    @cached_property
    def cov_matrix(self):
        # Ledoit-Wolf fit is the expensive step; compute once and share across methods
        return covariance.lw_cov(self.holdings)

    def volatility(self):
        return metrics.annualized_volatility(self.port_returns)

    def beta(self):
        stock_betas = metrics.stock_betas(self.returns, self.benchmark_col)
        stock_betas = stock_betas.drop(self.benchmark_col)
        return (stock_betas * self.weights).sum()

    def sharpe(self):
        annualized_returns = self.port_returns.mean() * 252
        return (annualized_returns - RISK_FREE_RATE) / self.volatility()

    def max_drawdown(self):
        cumulative_returns = (1 + self.port_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        return drawdowns.min()

    def volatility_from_cov(self):
        w = self.weights.reindex(self.cov_matrix.columns)
        daily_port_var = (w @ self.cov_matrix @ w)
        daily_vol = np.sqrt(daily_port_var)
        return daily_vol * (252 ** .5)

    def diversification_ratio(self):
        # individual vols from the same LW matrix as the portfolio vol, so numerator and denominator match
        individual_vols = np.sqrt(np.diag(self.cov_matrix)) * (252 ** .5)
        w = self.weights.reindex(self.cov_matrix.columns)
        weighted_avg_vols = (individual_vols * w).sum()
        portfolio_vol = self.volatility_from_cov()
        return weighted_avg_vols / portfolio_vol

    def var_historical(self, confidence=0.95):
        return var.historical_var(self.port_returns, confidence)

    def var_parametric(self, confidence=0.95):
        return var.parametric_var(self.port_returns, confidence)

    def var_monte_carlo(self, confidence=0.95, n_sims=10000, seed=42):
        return var.monte_carlo_var(self.holdings, self.weights, confidence, n_sims, cov=self.cov_matrix, seed=seed)

    def cvar_historical(self, confidence=0.95):
        return var.cvar(self.port_returns, confidence)

    def cvar_monte_carlo(self, confidence=0.95, n_sims=10000, seed=42):
        sims = var.simulate_port_returns(self.holdings, self.weights, n_sims, cov=self.cov_matrix, seed=seed)
        return var.cvar(sims, confidence)

    def cvar_parametric(self, confidence=0.95):
        return var.parametric_cvar(self.port_returns, confidence)
