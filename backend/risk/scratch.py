import sys, pathlib
# Allow running this file directly (python scratch.py); puts the repo root
# on sys.path so the `backend` package and `config` resolve.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from backend.risk.returns import get_returns
from backend.risk.portfolio import Portfolio
from backend.risk import metrics
from config import weights

rets = get_returns(kind="simple")  # simple, since drawdown uses (1+r).cumprod()
p = Portfolio(rets, weights)

print("=== Portfolio metrics ===")
print(f"volatility:   {p.volatility():.4f}   # expect ~0.15-0.35")
print(f"beta:         {p.beta():.4f}   # expect ~0.7-1.3")
print(f"sharpe:       {p.sharpe():.4f}   # expect ~0.3-1.5")
print(f"max_drawdown: {p.max_drawdown():.4f}   # expect negative, ~ -0.2 to -0.5")

print("\n=== Per-stock metrics (from metrics.py) ===")
print("annualized_volatility:  # expect ~0.15-0.35 per stock")
print(metrics.annualized_volatility(rets))
print("\nstock_betas:  # expect ~0.7-1.3; SPY's own beta should be exactly 1.0")
print(metrics.stock_betas(rets))
print("\nstock_sharpe:  # expect ~0.3-1.5")
print(metrics.stock_sharpe(rets))
print("\nstock_max_drawdown:  # expect negative, ~ -0.2 to -0.5")
print(metrics.stock_max_drawdown(rets))

print("\n=== Two-way volatility check ===")
vol_from_class = p.volatility()
vol_direct = p.port_returns.std() * (252 ** .5)
print(f"p.volatility():            {vol_from_class:.10f}")
print(f"direct std * sqrt(252):    {vol_direct:.10f}")
print(f"difference:                {vol_from_class - vol_direct:.2e}   # expect ~0")

print("empirical vol: ", p.volatility())
print("from cov vol:  ", p.volatility_from_cov())
print("difference:    ", abs(p.volatility() - p.volatility_from_cov()))