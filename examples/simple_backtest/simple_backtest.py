# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# packages
from portfolio_optimization import data 
from portfolio_optimization import portfolio_optimizer 
from portfolio_optimization import max_sharpe
from portfolio_optimization import expected_return
from portfolio_optimization import plot_portfolio_composition 
from portfolio_optimization import calculate_metrics

# ==========================================
# Settings and data
# ==========================================
TICKERS_INDEX = "nasdaq100"
MARKET_TICKER = "^GSPC"
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
BIG_M = 20
RISK_FREE_RATE = 0.01732 / 252 
U = 0.2
L = 0
LAMBDA = 5

# download data from yahoo or use data.tsv file
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
# Split data (70/30)
# ==========================================
# sample data (training data)
split_idx = int(len(returns) * 0.7)
train_returns = returns.iloc[:split_idx]
train_market = {
    "daily": market_data["daily"].iloc[:split_idx],
    "r_m": market_data["r_m"]
}

# out of sample data (test data)
test_returns_log = returns.iloc[split_idx:]
test_returns_lin = np.exp(test_returns_log) - 1

# ==========================================
# Optimisation
# ==========================================
# calculate expected returns
mu_train = expected_return(
    method="CAPM",
    returns=train_returns,
    r_f=RISK_FREE_RATE,
    market_returns=train_market
)

# for saving results
all_portfolio_returns = {}
all_weights = {}

# module for saving weights and returns
def process_result(name, w):
    # extract value if it exist for numpy
    weights = w.value if hasattr(w, "value") else w
    all_weights[name] = weights
    all_portfolio_returns[name] = test_returns_lin @ weights

# --- VARIANCE ---
w_var = portfolio_optimizer(mu=mu_train, returns=train_returns, risk_measure="variance", 
                                  Lambda=LAMBDA, M=BIG_M, l=L, u=U, 
                                  points=0, Solver="mosek", timer=600)[2]
process_result("Variance", w_var)

# --- SEMIVARIANCE ---
w_semi = portfolio_optimizer(mu=mu_train, returns=train_returns, risk_measure="semivariance", 
                                   Lambda=LAMBDA, M=BIG_M, l=L, u=U, 
                                   points=0, Solver="mosek", timer=600)[2]
process_result("Semivariance", w_semi)

# --- CVaR ---
w_cvar = portfolio_optimizer(mu=mu_train, returns=train_returns, risk_measure="cvar", 
                                   Lambda=LAMBDA, M=BIG_M, l=L, u=U, 
                                   points=0, Solver="mosek", timer=600)[2]
process_result("CVaR", w_cvar)

# --- EVaR ---
w_evar = portfolio_optimizer(mu=mu_train, returns=train_returns, risk_measure="evar", 
                                   Lambda=LAMBDA, M=BIG_M, l=L, u=U,  
                                   points=0, Solver="mosek", timer=600)[2]
process_result("EVaR", w_evar)

# --- AVE_DD ---
w_ave_dd = portfolio_optimizer(mu=mu_train, returns=train_returns, risk_measure="ave-dd", drawdown_return_type="log", 
                                     Lambda=LAMBDA, M=BIG_M, l=L, u=U, 
                                     points=0, Solver="mosek", timer=600)[2]
process_result("Ave-DD", w_ave_dd)

# --- CVAR_DD ---
w_cvar_dd = portfolio_optimizer(mu=mu_train, returns=train_returns, risk_measure="cvar-dd", drawdown_return_type="log", 
                                      Lambda=LAMBDA, M=BIG_M, l=L, u=U, 
                                      points=0, Solver="mosek", timer=600)[2]
process_result("CVaR-DD", w_cvar_dd)

# --- MAX SHARPE ---
# high K for numerical stability and timer=6000 for finding optimal solution
w_sh = max_sharpe(mu=mu_train, returns=train_returns, r_f = RISK_FREE_RATE, 
                                M=BIG_M, l=L, u=U, K=3000, 
                                Solver="mosek", timer=6000)[3]
process_result("Max Sharpe", w_sh)

# ==========================================
# Benchmarks
# ==========================================
benchmark_returns = test_returns_lin.mean(axis=1)
test_market_log = market_data["daily"].iloc[split_idx:]
test_market_lin = np.exp(test_market_log) - 1

# ==========================================
# Cumulative returns
# ==========================================
plt.figure(figsize=(14, 7), dpi=120)

for name, p_ret in all_portfolio_returns.items():
    plt.plot((1 + p_ret).cumprod().index, (1 + p_ret).cumprod(), label=name, linewidth=1.2, alpha=0.9)

plt.plot((1 + benchmark_returns).cumprod().index, (1 + benchmark_returns).cumprod(), label="Nasdaq 100 (Equal Weight)", linestyle="--", linewidth=1.0, alpha=0.6)
plt.plot((1 + test_market_lin).cumprod().index, (1 + test_market_lin).cumprod(), label="Nasdaq 100 Index", color='black', linestyle=":", linewidth=1.0, alpha=0.6)

plt.title("Backtest")
plt.legend()
plt.xlabel("Date", fontsize=10)
plt.ylabel("Cumulative returns", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# Table of metrics
# ==========================================
all_metrics = {}
for name, p_ret in all_portfolio_returns.items():
    all_metrics[name] = calculate_metrics(returns=p_ret, r_f=RISK_FREE_RATE, V=RISK_FREE_RATE)

all_metrics["Equal Weight"] = calculate_metrics(returns=benchmark_returns, r_f=RISK_FREE_RATE, V=RISK_FREE_RATE)
all_metrics["Nasdaq Index"] = calculate_metrics(returns=test_market_lin, r_f=RISK_FREE_RATE, V=RISK_FREE_RATE)

comparison = pd.DataFrame(all_metrics).T
print("\n--- POROVNANIE RIZIKOVÝCH METRÍK ---")
print(comparison.to_string(formatters={
    'Return': '{:,.2%}'.format, 'Volatility': '{:,.2%}'.format, 'Semivolatility': '{:,.2%}'.format,
    'Sharpe': '{:,.2f}'.format, 'Sortino': '{:,.2f}'.format, 'CVaR 95%': '{:,.2%}'.format,
    'Max Drawdown': '{:,.2%}'.format, 'Average Drawdown': '{:,.2%}'.format
}))

# ==========================================
# Composition of portfolios
# ==========================================
for name, weights in all_weights.items():
    plot_portfolio_composition(weights = weights, tickers = tickers, portfolio_name = name)

# ==========================================
# Heatmap of correlation between  returns of portfolios and benchmarks
# ==========================================
# create dataframe for models and benchmarks
all_returns = pd.DataFrame(all_portfolio_returns) 
all_returns["Equal Weight"] = benchmark_returns
all_returns["Nasdaq Index"] = test_market_lin

# calculate correlarion matrix
corr_matrix = all_returns.corr()

plt.figure(figsize=(10, 8), dpi=120)
# masking upper triangle for better visualisation
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, 
            mask=mask, 
            annot=True, 
            fmt=".3f", 
            cmap='RdBu_r', 
            center=0.9,    
            linewidths=0.5, 
            square=True)

plt.title("Correlation matrix of out-of-sample returns", fontsize=14, pad=20)
plt.tight_layout()
plt.show()