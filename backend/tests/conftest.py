import numpy as np
import pandas as pd
import pytest

TICKERS = ["AAA", "BBB", "CCC", "DDD"]
SECTORS = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy", "DDD": "Bonds"}


@pytest.fixture
def returns():
    """Synthetic daily simple returns: each stock = beta * market + noise."""
    rng = np.random.default_rng(0)
    n = 750
    market = rng.normal(0.0004, 0.01, n)
    betas = [1.5, 1.0, 0.6, 0.1]
    data = {t: b * market + rng.normal(0, 0.008, n) for t, b in zip(TICKERS, betas)}
    data["SPY"] = market
    return pd.DataFrame(data, index=pd.bdate_range("2022-01-03", periods=n))


@pytest.fixture
def weights():
    return pd.Series([0.4, 0.3, 0.2, 0.1], index=TICKERS)


@pytest.fixture
def sector_map():
    return SECTORS
