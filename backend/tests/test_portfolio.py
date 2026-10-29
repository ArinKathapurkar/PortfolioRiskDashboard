import pytest

from backend.risk import metrics
from backend.risk.portfolio import Portfolio


@pytest.fixture
def p(returns, weights):
    return Portfolio(returns, weights)


def test_benchmark_beta_is_one(returns):
    assert metrics.stock_betas(returns)["SPY"] == pytest.approx(1.0)


def test_volatility_matches_direct_std(p):
    assert p.volatility() == pytest.approx(p.port_returns.std() * 252 ** .5)


def test_cov_vol_close_to_empirical_vol(p):
    # Ledoit-Wolf shrinkage nudges the estimate, but it should stay close
    assert p.volatility_from_cov() == pytest.approx(p.volatility(), rel=0.05)


def test_max_drawdown_in_range(p):
    assert -1 < p.max_drawdown() <= 0


def test_diversification_ratio_above_one(p):
    assert p.diversification_ratio() > 1


@pytest.mark.parametrize("kind", ["historical", "parametric", "monte_carlo"])
def test_cvar_at_least_as_bad_as_var(p, kind):
    v = getattr(p, f"var_{kind}")(0.95)
    cv = getattr(p, f"cvar_{kind}")(0.95)
    assert v < 0
    assert cv <= v


def test_var_methods_roughly_agree(p):
    # synthetic returns are normal, so all three estimators should land close together
    h, para, mc = p.var_historical(), p.var_parametric(), p.var_monte_carlo()
    assert para == pytest.approx(h, rel=0.15)
    assert mc == pytest.approx(h, rel=0.15)
