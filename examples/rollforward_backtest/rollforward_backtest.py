# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# packages
from portfolio_optimization import data 
from portfolio_optimization import portfolio_optimizer 
from portfolio_optimization import max_sharpe
from portfolio_optimization import expected_return
from portfolio_optimization import calculate_metrics

# ==========================================
# Settings and data
# ==========================================
TICKERS_INDEX = "s&p500"
MARKET_TICKER = "^GSPC"
START_DATE = "2019-01-01"  
END_DATE = "2026-01-01"
LAMBDA = 5
A = 0.8
B = 0.05
K_TURNOVER = 0.3
RISK_FREE_RATE = 0.04 / 252

# Settings for roll-forward window
WINDOW_SIZE = 252 
STEP_SIZE = 21    

# downloading data or load data.tsv file
returns, tickers, market_data = data(
    source="yahoo",
    index_name=TICKERS_INDEX,
    market=MARKET_TICKER,
    start_date=START_DATE,
    end_date=END_DATE,
    return_type="log",
    method="mean"
)

# ==========================================
# Inputs for roll-forward
# ==========================================

# Strategies for testing
strategies = ["Variance", "Semivariance", "CVaR", "EVaR", "Ave-DD", "CVaR-DD", "Max Sharpe"]

# Variables for saving actual weights and returns
rolling_returns = {name: [] for name in strategies}
num_assets = len(tickers)
current_weights = {name: np.zeros(num_assets) for name in strategies}

# ==========================================
# ROLLING WINDOW LOOP
# ==========================================
# new iteration every 21 days (approximately one month)
for i in range(WINDOW_SIZE, len(returns) - STEP_SIZE, STEP_SIZE):
    # pick train and test data
    train_returns = returns.iloc[i - WINDOW_SIZE : i].copy()
    test_returns_lin = np.exp(returns.iloc[i : i + STEP_SIZE]) - 1
    
    # expected returns
    mu_rolling = expected_return(
        method="mean",
        returns=train_returns
    )

    # if this is first step whan k_turnover has to be 1, so it takes no effect 
    first_step = (i == WINDOW_SIZE)
    K_constraint = 1 if first_step else K_TURNOVER
    
    # --- VARIANCE ---
    w_var = portfolio_optimizer(mu=mu_rolling, returns=train_returns, risk_measure="variance", 
                                      Lambda=LAMBDA, w_0=current_weights["Variance"], k_turnover=K_constraint, A=A, B=B, 
                                      Solver="mosek", timer = 600, points=0)[2]
    # extract value if it exist for numpy
    w_v = w_var.value if hasattr(w_var, "value") else w_var
    rolling_returns["Variance"].append(test_returns_lin @ w_v)
    current_weights["Variance"] = w_v

    # --- SEMIVARIANCE ---
    w_semi = portfolio_optimizer(mu=mu_rolling, returns=train_returns, risk_measure="semivariance", 
                                       Lambda=LAMBDA, w_0=current_weights["Semivariance"], k_turnover=K_constraint, A=A, B=B,  
                                       Solver="mosek", timer = 600, points=0)[2]
    w_s = w_semi.value if hasattr(w_semi, "value") else w_semi
    rolling_returns["Semivariance"].append(test_returns_lin @ w_s)
    current_weights["Semivariance"] = w_s

    # --- CVaR ---
    w_cvar = portfolio_optimizer(mu=mu_rolling, returns=train_returns, risk_measure="cvar", 
                                       Lambda=LAMBDA, w_0=current_weights["CVaR"], k_turnover=K_constraint, A=A, B=B,  
                                       Solver="mosek", timer = 600, points=0)[2]
    w_c = w_cvar.value if hasattr(w_cvar, "value") else w_cvar
    rolling_returns["CVaR"].append(test_returns_lin @ w_c)
    current_weights["CVaR"] = w_c

    # --- EVaR ---
    w_evar = portfolio_optimizer(mu=mu_rolling, returns=train_returns, risk_measure="evar", 
                                       Lambda=LAMBDA, w_0=current_weights["EVaR"], k_turnover=K_constraint, A=A, B=B, 
                                       Solver="mosek", timer = 600, points=0)[2]
    w_e = w_evar.value if hasattr(w_evar, "value") else w_evar
    rolling_returns["EVaR"].append(test_returns_lin @ w_e)
    current_weights["EVaR"] = w_e

    # --- AVE-DD ---
    w_ave = portfolio_optimizer(mu=mu_rolling, returns=train_returns, risk_measure="ave-dd",  drawdown_return_type="log", 
                                      Lambda=LAMBDA, w_0=current_weights["Ave-DD"], k_turnover=K_constraint, A=A, B=B, 
                                      Solver="mosek", timer = 600, points=0)[2]
    w_a = w_ave.value if hasattr(w_ave, "value") else w_ave
    rolling_returns["Ave-DD"].append(test_returns_lin @ w_a)
    current_weights["Ave-DD"] = w_a
    
    # --- CVaR-DD ---
    w_cdd = portfolio_optimizer(mu=mu_rolling, returns=train_returns, risk_measure="cvar-dd",  drawdown_return_type="log", 
                                      Lambda=LAMBDA, w_0=current_weights["CVaR-DD"], k_turnover=K_constraint, A=A, B=B, 
                                      Solver="mosek", timer = 600, points=0)[2]
    w_cd = w_cdd.value if hasattr(w_cdd, "value") else w_cdd
    rolling_returns["CVaR-DD"].append(test_returns_lin @ w_cd)
    current_weights["CVaR-DD"] = w_cd

    # --- MAX SHARPE ---
    w_sh = max_sharpe(mu=mu_rolling, returns=train_returns, r_f=RISK_FREE_RATE, 
                               w_0=current_weights["Max Sharpe"], k_turnover=K_constraint, gamma=1,
                               Solver="mosek", timer = 600)[3]
    w_sh_val = w_sh.value if hasattr(w_sh, "value") else w_sh
    rolling_returns["Max Sharpe"].append(test_returns_lin @ w_sh_val)
    current_weights["Max Sharpe"] = w_sh_val


# ==========================================
# Benchmarks
# ==========================================
final_returns = pd.DataFrame()
for name in strategies:
    if rolling_returns[name]: 
        final_returns[name] = pd.concat(rolling_returns[name])

common_index = final_returns.index

# Equal Weight 
test_returns = np.exp(returns.loc[common_index]) - 1
benchmark_ew_returns = test_returns.mean(axis=1)

# Index benchmark
test_market= np.exp(market_data["daily"].loc[common_index]) - 1

# Add benchmarks to metrics
final_returns["Equal Weight"] = benchmark_ew_returns
final_returns["S&P 500 Index"] = test_market

# ==========================================
# Graph of cumulative returns
# ==========================================
plt.figure(figsize=(14, 7), dpi=120)
for col in final_returns.columns:
    if col == "S&P 500 Index":
        plt.plot((1 + final_returns[col]).cumprod(), label=col, color='black', linestyle=':')
    else:
        plt.plot((1 + final_returns[col]).cumprod(), label=col, alpha=0.6)

plt.title(f"Roll-Forward Backtest (Rebalance: 1M, Window: 1Y)")
plt.legend()
plt.xlabel("Date", fontsize=10)
plt.ylabel("Cumulative returns", fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# Table of metrics
# ==========================================
all_metrics = {}
# Calculate metrics
for col in final_returns.columns:
    all_metrics[col] = calculate_metrics(
        returns=final_returns[col], 
        r_f=RISK_FREE_RATE, 
        V=RISK_FREE_RATE
    )

comparison = pd.DataFrame(all_metrics).T

print("\n Risk metrics")
print(comparison.to_string(formatters={
    'Return': '{:,.2%}'.format, 'Volatility': '{:,.2%}'.format, 'Semivolatility': '{:,.2%}'.format,
    'Sharpe': '{:,.2f}'.format, 'Sortino': '{:,.2f}'.format, 'CVaR 95%': '{:,.2%}'.format,
    'Max Drawdown': '{:,.2%}'.format, 'Average Drawdown': '{:,.2%}'.format
}))
