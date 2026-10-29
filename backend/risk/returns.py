from backend.data import fetch
import numpy as np
import pandas as pd

# Default is simple returns: Portfolio aggregates holdings as a weighted sum
# and compounds with (1 + r).cumprod(), both of which only hold for simple returns.
def get_returns(kind = "simple"):
    prices = fetch.get_prices()
    if kind == "simple":
        return prices.pct_change(fill_method=None).dropna()
    if kind == "log":
        return np.log(prices).diff().dropna()
    raise ValueError(f"kind must be 'simple' or 'log', got {kind!r}")
