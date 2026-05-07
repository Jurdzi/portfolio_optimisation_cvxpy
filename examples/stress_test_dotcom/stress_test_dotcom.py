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
# tickers from https://sites.google.com/site/unofficialnasdaq100site/yearly-year-end-rosters/1998
HISTORICAL_TICKERS_1998 = [
    "COMS", "ADPT", "ADCT", "ADBE", "ALTR", "AMZN", "APCC", "AMGN", "ANDW", "APOL",
    "AAPL", "AMAT", "ASND", "ATML", "ADSK", "BBBY", "BGEN", "BMET", "BMCS", "CATP",
    "CBRL", "CNTO", "AMFM", "CHIR", "CTAS", "CSCO", "CTXS", "COMR", "CMCSK", "CPWR",
    "CEFT", "CEXP", "COST", "DELL", "DLTR", "ERTS", "EFII", "FAST", "FHCC", "FISV",
    "FDLNB", "FORE", "GENZ", "HBOC", "MLHR", "IMNX", "INTC", "INTU", "JCOR", "KLAC",
    "LVLT", "LNCR", "LLTC", "ERICY", "MXIM", "MCCRK", "WCOM", "MCLD", "MCHP", "MUEI",
    "MSFT", "MOLX", "NSCP", "NETA", "NXTL", "NOBE", "NWAC", "NOVL", "NTLI", "ORCL",
    "PCAR", "PHSYB", "SPOT", "PMTC", "PAYX", "PSFT", "QCOM", "QNTM", "QTRN", "QWST",
    "RTRSY", "RXSD", "ROST", "SANM", "SIAL", "SSCC", "SPLS", "SBUX", "STEW", "SUNW",
    "SNPS", "TECD", "TCOMA", "TLAB", "USAI", "VRTS", "VTSS", "WTHG", "XLNX", "YHOO"
]

# Nasdaq Composite for more data for benchmark
MARKET_TICKER = "^IXIC"  
START_DATE = "1997-01-01"
STRESS_START = "2000-03-10" 
END_DATE = "2002-10-01"    
RISK_FREE_RATE = 0.05 / 252 
D = 0.1
LAMBDA = 5

# # download data from yahoo or load data.tsv file

returns, tickers, market_data = data(
    source="yahoo",
    tickers=HISTORICAL_TICKERS_1998,  
    market=MARKET_TICKER,                  
    start_date=START_DATE,
    end_date=END_DATE,
    return_type="log",
    method="mean", 
    gamma = 1
)

# ==========================================
# Split data
# ==========================================
# day before burst of bubble
train_returns = returns.loc[:STRESS_START]

# out of sample dat
test_returns_log = returns.loc[STRESS_START:]
test_returns_lin = np.exp(test_returns_log) - 1

# ==========================================
# Optimization
# ==========================================
# calculate expected return
mu_train = expected_return(
    method="mean",
    returns=train_returns
)

# for saving results
all_portfolio_returns = {}
all_weights = {}

# module for saving weights and returns
def process_result(name, w):
    weights = w.value if hasattr(w, "value") else w
    all_weights[name] = weights
    # Aplikácia váh na lineárne výnosy počas krízy
    all_portfolio_returns[name] = test_returns_lin @ weights

# --- VARIANCE ---
w_var = portfolio_optimizer(mu=mu_train, returns=train_returns, risk_measure="variance", 
                                  Lambda=LAMBDA, D=D,
                                  points=0, Solver="mosek", timer=600)[2]
process_result("Variance", w_var)

# --- SEMIVARIANCE ---
w_semi = portfolio_optimizer(mu=mu_train, returns=train_returns, risk_measure="semivariance", 
                                   Lambda=LAMBDA, D=D,
                                   points=0, Solver="mosek", timer=600)[2]
process_result("Semivariance", w_semi)

# --- CVaR ---
w_cvar = portfolio_optimizer(mu=mu_train, returns=train_returns, risk_measure="cvar", 
                                   Lambda=LAMBDA, D=D,
                                   points=0, Solver="mosek", timer=600)[2]
process_result("CVaR", w_cvar)

# --- EVaR ---
w_evar = portfolio_optimizer(mu=mu_train, returns=train_returns, risk_measure="evar", 
                                   Lambda=LAMBDA, D=D,
                                   points=0, Solver="mosek", timer=600)[2]
process_result("EVaR", w_evar)

# --- AVE_DD ---
w_ave_dd = portfolio_optimizer(mu=mu_train, returns=train_returns, risk_measure="ave-dd", drawdown_return_type="log", 
                                     Lambda=LAMBDA, D=D,
                                     points=0, Solver="mosek", timer=600)[2]
process_result("Ave-DD", w_ave_dd)


# --- CVAR_DD ---
w_cvar_dd = portfolio_optimizer(mu=mu_train, returns=train_returns, risk_measure="cvar-dd", drawdown_return_type="log", 
                                  Lambda=LAMBDA, D=D,
                                  points=0, Solver="mosek", timer=600)[2]
process_result("CVaR-DD", w_cvar_dd)

# --- MAX SHARPE ---
# we did not choose here D = 0.1, because solver had trouble finding solution with numerically small numbers. However max_sharpe is usually diversifed enough. 
# This issue can be solved by using annualized teruns.
w_sh = max_sharpe(mu=mu_train, returns=train_returns, r_f = RISK_FREE_RATE, 
                                Solver="mosek", timer=600)[3]
process_result("Max Sharpe", w_sh)

# ==========================================
# Benchmarks
# ==========================================
benchmark_returns = test_returns_lin.mean(axis=1) 
test_market_log = market_data["daily"].loc[STRESS_START:]
test_market_lin = np.exp(test_market_log) - 1

# ==========================================
# Cumulative returns
# ==========================================
plt.figure(figsize=(14, 7), dpi=120)

for name, p_ret in all_portfolio_returns.items():
    plt.plot((1 + p_ret).cumprod().index, (1 + p_ret).cumprod(), label=name, linewidth=1.2)

plt.plot((1 + benchmark_returns).cumprod().index, (1 + benchmark_returns).cumprod(), label="Equal Weight", linestyle="--", alpha=0.6)
plt.plot((1 + test_market_lin).cumprod().index, (1 + test_market_lin).cumprod(), label="Nasdaq 100 Index", color="black", linestyle=":", alpha=0.8)

plt.title("Stress Test: Dot-com Bubble Crash")
plt.legend()
plt.xlabel("Date", fontsize=10)
plt.ylabel("Cumulative returns", fontsize=10)
plt.grid(True, alpha=0.3)
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

plt.title("Correlation matrix of Out-of-Sample returns (Dot-com Stress Test)", fontsize=14, pad=20)
plt.tight_layout()
plt.show()
