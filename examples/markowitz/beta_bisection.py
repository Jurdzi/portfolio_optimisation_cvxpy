# packages
from portfolio_optimization import data 
from portfolio_optimization import portfolio_optimizer

# download data from yahoo or load data.tsv file
returns, tickers = data(
    source = "yahoo", 
    index_name="S&P500",
    start_date="2005-01-01",
    end_date="2025-12-31",
    return_type= "lin"
) 

# optimization
obj_return, obj_risk, w = portfolio_optimizer(returns = returns, return_method = "geometric_mean", risk_measure = "variance",
                    beta = 0.25, D = 0.05, bisection = "Y", gamma = 252,
                    eff_frontier = "Y")

