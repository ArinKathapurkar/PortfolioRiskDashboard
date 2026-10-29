import numpy as np
import pytest

from backend.risk import covariance, exposures, metrics
from backend.risk.portfolio import Portfolio


@pytest.fixture
def cov(returns):
    return covariance.lw_cov(returns.drop(columns="SPY"))


def test_pct_contributions_sum_to_one(cov, weights):
    risk_df = exposures.risk_contributions(cov, weights)
    assert risk_df["percentage_contributions"].sum() == pytest.approx(1.0)


def test_risk_contributions_sum_to_port_vol(cov, weights):
    # Euler decomposition: sum of w_i * MCR_i equals sqrt(w' Σ w)
    risk_df = exposures.risk_contributions(cov, weights)
    w = weights.reindex(cov.columns)
    assert risk_df["risk_contributions"].sum() == pytest.approx(np.sqrt(w @ cov @ w))


def test_sector_risk_contributions_preserve_total(cov, weights, sector_map):
    sector_rc = exposures.sector_risk_contributions(cov, weights, sector_map)
    total_rc = exposures.risk_contributions(cov, weights)["risk_contributions"].sum()
    assert set(sector_rc.index) == {"Tech", "Energy", "Bonds"}
    assert sector_rc.sum() == pytest.approx(total_rc)


def test_sector_weight(weights, sector_map):
    sw = exposures.sector_weight(weights, sector_map)
    assert sw["Tech"] == pytest.approx(0.7)
    assert sw.sum() == pytest.approx(1.0)


def test_beta_exposures_sum_to_portfolio_beta(returns, weights):
    be = exposures.beta_exposures(metrics.stock_betas(returns), weights)
    assert "SPY" not in be.index
    assert be.sum() == pytest.approx(Portfolio(returns, weights).beta())


def test_r_squared_matches_variance_decomposition(returns, weights):
    # R² = corr² = β²σ_m² / σ_p² for a single-factor regression
    p = Portfolio(returns, weights)
    port_beta = p.port_returns.cov(p.benchmark) / p.benchmark.var()
    expected = port_beta ** 2 * p.benchmark.var() / p.port_returns.var()
    r2 = exposures.market_r_squared(p.port_returns, p.benchmark)
    assert r2 == pytest.approx(expected)
    assert 0 <= r2 <= 1
