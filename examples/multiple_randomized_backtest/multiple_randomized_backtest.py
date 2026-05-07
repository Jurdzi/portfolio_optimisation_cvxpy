# libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
import yfinance as yf

# packages
from portfolio_optimization import data 
from portfolio_optimization import portfolio_optimizer 
from portfolio_optimization import max_sharpe
from portfolio_optimization import expected_return
from portfolio_optimization import calculate_metrics
from portfolio_optimization import ensure_solution

# ==========================================
# Settings and data
# ==========================================
TICKERS_INDEX = "s&p500"
MARKET_TICKER = "^GSPC" 
START_DATE = "2005-01-01" 
END_DATE = "2026-01-01"

# settings for roll-forward window
N_SIMULATIONS = 1000   
N_ASSETS = 100         
TOTAL_DAYS = 252 * 2     
WINDOW_SIZE = 252     
STEP_SIZE = 21  

# download interest rate of 10 year US bond or load rf_data.tsv file
rf_data = yf.download("^TNX", start="2005-01-01", end="2026-01-01")['Close']
# coversion to daily rate
rf_daily = (rf_data / 100) / 252

# download data load data.tsv file
all_returns, all_tickers, market_data = data(
    source="yahoo",
    index_name=TICKERS_INDEX,
    market=MARKET_TICKER,
    start_date=START_DATE,
    end_date=END_DATE,
    return_type="log",
    method="mean"
)

strategies = ["Variance", "Semivariance", "CVaR", "EVaR", "Ave-DD", "CVaR-DD", "Max Sharpe", "EW"]
sim_results = {name: [] for name in strategies}
sim_returns_for_boxplot = {name: [] for name in sim_results.keys()}

# ==========================================
# Randomized rolling window backtest
# ==========================================
for sim in range(N_SIMULATIONS):
    
    # randomly select 100 assets
    valid_tickers = all_returns.dropna(axis=1).columns
    selected_tickers = random.sample(list(valid_tickers), N_ASSETS)
    
    # randomly select 2 year interval
    max_start = len(all_returns) - TOTAL_DAYS - 1
    start_point = random.randint(0, max_start)
    sim_subset = all_returns.iloc[start_point : start_point + TOTAL_DAYS][selected_tickers]
    
    # initialisation of variables for one simulation
    rolling_returns = {name: [] for name in sim_results.keys()}
    current_weights = {name: None for name in sim_results.keys()}

    # pick random lambda
    LAMBDA = random.uniform(1, 10)

    # roll-forward for simulation
    for i in range(WINDOW_SIZE, len(sim_subset) - STEP_SIZE, STEP_SIZE):
        # pick train and test data
        train_returns = sim_subset.iloc[i - WINDOW_SIZE : i]
        test_returns_lin = np.exp(sim_subset.iloc[i : i + STEP_SIZE]) - 1

        # pick risk free rate for last day in training data
        RISK_FREE_RATE = rf_daily.asof(train_returns.index[-1]).item() 

        # expected returns
        mu_rolling = mu_rolling = expected_return(
            method="mean",
            returns=train_returns
        )

        # inputs for ensure_soution
        num_assets = train_returns.shape[1]
        first_step = len(rolling_returns["EW"]) == 0

        # --- VARIANCE ---
        w_v = ensure_solution(
            solve_func=lambda: portfolio_optimizer(mu=mu_rolling, returns=train_returns, risk_measure="variance", 
                                                   Lambda=LAMBDA, 
                                                   Solver="mosek", timer = 600, points=0)[2],
            prev_weights=current_weights["Variance"],
            first_step=first_step, num_assets=num_assets
        )
        rolling_returns["Variance"].append(test_returns_lin @ w_v)
        current_weights["Variance"] = w_v

        # --- SEMIVARIANCE ---
        w_s = ensure_solution(
            solve_func=lambda: portfolio_optimizer(mu=mu_rolling, returns=train_returns, risk_measure="semivariance", 
                                                   Lambda=LAMBDA,
                                                   Solver="mosek", timer = 600, points=0)[2],
            prev_weights=current_weights["Semivariance"],
            first_step=first_step, num_assets=num_assets
        )
        rolling_returns["Semivariance"].append(test_returns_lin @ w_s)
        current_weights["Semivariance"] = w_s

        # --- CVaR ---
        w_c = ensure_solution(
            solve_func=lambda: portfolio_optimizer(mu=mu_rolling, returns=train_returns, risk_measure="cvar", 
                                                   Lambda=LAMBDA, 
                                                   Solver="mosek", timer = 600, points=0)[2],
            prev_weights=current_weights["CVaR"],
            first_step=first_step, num_assets=num_assets
        )
        rolling_returns["CVaR"].append(test_returns_lin @ w_c)
        current_weights["CVaR"] = w_c

        # --- EVaR ---
        w_e = ensure_solution(
            solve_func=lambda: portfolio_optimizer(mu=mu_rolling, returns=train_returns, risk_measure="evar", 
                                                   Lambda=LAMBDA, 
                                                   Solver="mosek", timer = 600, points=0)[2],
            prev_weights=current_weights["EVaR"],
            first_step=first_step, num_assets=num_assets
        )
        rolling_returns["EVaR"].append(test_returns_lin @ w_e)
        current_weights["EVaR"] = w_e

        # --- AVE-DD ---
        w_a = ensure_solution(
            solve_func=lambda: portfolio_optimizer(mu=mu_rolling, returns=train_returns, risk_measure="ave-dd", drawdown_return_type="log", 
                                                   Lambda=LAMBDA, 
                                                   Solver="mosek", timer = 600, points=0)[2],
            prev_weights=current_weights["Ave-DD"],
            first_step=first_step, num_assets=num_assets
        )
        rolling_returns["Ave-DD"].append(test_returns_lin @ w_a)
        current_weights["Ave-DD"] = w_a

        # --- CVaR-DD ---
        w_cd = ensure_solution(
            solve_func=lambda: portfolio_optimizer(mu=mu_rolling, returns=train_returns, risk_measure="cvar-dd", drawdown_return_type="log", 
                                                   Lambda=LAMBDA, 
                                                   Solver="mosek", timer = 600, points=0)[2],
            prev_weights=current_weights["CVaR-DD"],
            first_step=first_step, num_assets=num_assets
        )
        rolling_returns["CVaR-DD"].append(test_returns_lin @ w_cd)
        current_weights["CVaR-DD"] = w_cd

        # --- MAX SHARPE ---
        w_sh = ensure_solution(
            solve_func=lambda: max_sharpe(mu=mu_rolling, returns=train_returns, r_f=RISK_FREE_RATE, 
                                          Solver="mosek", timer=600)[3],
            prev_weights=current_weights["Max Sharpe"],
            first_step=first_step, num_assets=num_assets
        )
        rolling_returns["Max Sharpe"].append(test_returns_lin @ w_sh)
        current_weights["Max Sharpe"] = w_sh

        # --- EW ---
        w_ew = np.ones(num_assets) / num_assets
        rolling_returns["EW"].append(test_returns_lin @ w_ew)
        
    # proces results of simulation
    for name in sim_results.keys():
        if rolling_returns[name]:
            sim_series = pd.concat(rolling_returns[name])
            # calculate metrics
            metrics = calculate_metrics(returns=sim_series, r_f=RISK_FREE_RATE, V=RISK_FREE_RATE)
            sim_results[name].append(metrics)
            # save return
            sim_returns_for_boxplot[name].append(metrics['Return'])

# ==========================================
# Distribution of returns
# ==========================================
plt.figure(figsize=(12, 6))
returns_distribution = pd.DataFrame(sim_returns_for_boxplot)
flier_style = {
    'marker': 'o', 
    'markerfacecolor': 'black', 
    'markersize': 3,
    'markeredgecolor': 'none', 
    'alpha': 0.5
}
returns_distribution.boxplot(flierprops=flier_style)
plt.title(f"Distribution of cumulative returns")
plt.ylabel("Cumulative returns")
plt.grid(alpha=0.3)
plt.show()

# ==========================================
# Table of average returns
# ==========================================
avg_results = {}
for name in sim_results.keys():
    if sim_results[name]:
        avg_results[name] = pd.DataFrame(sim_results[name]).mean()

comparison = pd.DataFrame(avg_results).T

print("\n Average risk metrics")
print(comparison.to_string(formatters={
    'Return': '{:,.2%}'.format, 'Volatility': '{:,.2%}'.format, 'Semivolatility': '{:,.2%}'.format,
    'Sharpe': '{:,.2f}'.format, 'Sortino': '{:,.2f}'.format, 'CVaR 95%': '{:,.2%}'.format,
    'Max Drawdown': '{:,.2%}'.format, 'Average Drawdown': '{:,.2%}'.format
}))
