import pandas as pd 
import yfinance as yf 
import universe
import datetime as dt
from cache import load_or_fetch    

START = "2021-01-01"
END = dt.date.today().isoformat()



def fetch_prices(portfolio, start = START, end = END): 
    data = yf.download(portfolio, start=start, end=end, auto_adjust = True)
    
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = portfolio if isinstance(portfolio, list) else [portfolio]
    return prices.dropna(how="all")


def get_prices(force: bool = False):
    portfolio = universe.tickers + [universe.benchmark]
    return load_or_fetch(lambda: fetch_prices(portfolio), force=force)  


if __name__ == "__main__":
    prices = get_prices(force = True)
    print(prices)
    print(prices.shape)
    print(prices.index.max())
    returns = prices.pct_change()
    print(returns)

