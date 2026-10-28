import sys, pathlib
# Allow running this file directly (python scratch.py); puts the repo root
# on sys.path so the `backend` package and `config` resolve.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from backend.risk import returns as r, metrics as m
import config   # your weights

rets = r.get_returns(kind="simple")   # simple, since drawdown uses (1+r).cumprod()
w = config.weights                    # Series indexed by ticker, summing to 1

print(m.annualized_volatility(rets.drop(columns="SPY")))
print(m.stock_betas(rets))            # SPY should be 1.0
print(m.portfolio_beta(rets, w))
print(m.stock_sharpe(rets))
print(m.portfolio_sharpe(rets, w))
print(m.stock_max_drawdown(rets))
print(m.portfolio_max_drawdown(rets, w))