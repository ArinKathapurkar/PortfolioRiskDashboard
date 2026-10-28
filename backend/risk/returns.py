from backend.data import fetch
import numpy as np 
import pandas as pd

def get_returns(kind = "log"):
    prices = fetch.get_prices()
    if kind == "simple":
        return prices.pct_change(fill_method=None).dropna()
    if kind == "log":
        return np.log(prices).diff().dropna()
    raise ValueError(f"kind must be 'simple' or 'log', got {kind!r}")



         




    