# packages
import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import datetime as dt
import os
import urllib.request
import matplotlib.pyplot as plt 
# for bloomberg
# from xbbg import blp

# ==========================================
# Fetch tickers
# ==========================================
def index_tickers(index_name: str):
    """
    Module used to fetch current stock tickers from Wikipedia for a given index. 

    Inputs:
        - index_name (str): supported indices S&P500, nasdaq100, dowjones, ftse100, dax, cac40 are supported

    Returns:
        - tickers (list): a cleaned list of tickers formatted for Yahoo Finance

    """

    # take lower if some letters are big
    index_name = index_name.lower()

    # choose site by index_name
    if index_name in ["s&p500"]:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        symbol_column = "Symbol"
    elif index_name in ["nasdaq100"]:
        url = "https://en.wikipedia.org/wiki/NASDAQ-100"
        symbol_column = "Ticker"
    elif index_name in ["dowjones"]:
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        symbol_column = "Symbol"
    elif index_name in ["ftse100"]:
        url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
        symbol_column = "Ticker"
    elif index_name in ["dax"]:
        url = "https://en.wikipedia.org/wiki/DAX"
        symbol_column = "Ticker"
    elif index_name in ["cac40"]:
        url = "https://en.wikipedia.org/wiki/CAC_40"
        symbol_column = "Ticker"
    else:
        raise ValueError("Unsupported index_name." )

    # pretend to be a browser
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        html = response.read()

    # parse all tables on the page
    tables = pd.read_html(html)

    # try to find the table that contains the expected symbol column
    tickers = None
    for table in tables:
        if symbol_column in table.columns:
            tickers = table[symbol_column].dropna().tolist()
            break

    if tickers is None:
        raise ValueError(f"Could not find column '{symbol_column}' in any table on {url}")

    # clean tickers (remove non-strings and extra spaces)
    if index_name == "ftse100": 
        tickers = [str(t).strip() + ".L" for t in tickers] 
    else: 
        tickers = [str(t).strip() for t in tickers]

    print(f"Loaded {len(tickers)} tickers from {index_name.upper()}")
    return tickers

# ==========================================
# Download data from Yahoo or load them from local file
# ==========================================
def data(source = None, file_path = None, tickers = None, start_date = None, end_date = None, sheet_name = None,
         w_0 = None,
         index_name = None, market = None, method = None, alpha = None, gamma = None,
         return_type = None):
    """
    Data acquisition module for financial assets and market benchmarks.
    
    Inputs:
        - source (str): Data origin (yahoo, csv, xlsx, txt, tsv).
        - file_path (str): Path to the local file.
        - tickers (list): List of specific asset symbols to download from Yahoo.
        - start_date / end_date (str): Time horizon for historical data (e.g., 2010-01-01).
        - sheet_name (str): Target sheet for Excel files. If file has only one sheet this input can be empty. 
        - w_0 (str): Path to a file containing initial portfolio weights (csv, xlsx, txt, tsv).
        - index_name (str): Predefined index to fetch tickers automatically ('s&p500', 'nasdaq100', 'dowjones', 'ftse100', 'dax', 'cac40').
        - market (str): Ticker of the market proxy (e.g., '^GSPC') or name in local file for CAPM calculations.
        - method (str): Calculation of expected market return ('mean', 'geometric_mean', 'exponential_mean').
        - alpha (float): Parameter for the exponential recursive weighted mean.
        - gamma (float): Scalar for returns (trading days per year, default is 1).
        - return_type (str): Returns ('lin' for linear or 'log' for logarithmic).

    Returns:
        - returns (DataFrame): Historical assets returns used for parameter estimation.
        - tickers (list): Final list of processed asset identifiers.
        - market_data (dict): Dictionary containing daily and expected market returns for CAPM.
        - w0 (ndarray): Flattened array of initial portfolio weights.

    """
    
    if source is None:
        raise ValueError("Provide source for data.")
    
    # --- Yahoo ---
    if source.lower() == "yahoo":
        # check tickers
        if tickers is not None:
            tickers = tickers
        # chceck index_name and if necessary fetch tickers from Wikipedia
        elif index_name is not None:
            tickers = index_tickers(index_name)
        # if tickers and check_index are epmty there are no specifics for downloading data
        else:
            raise ValueError("Provide tickers name.")

        # check star_date for downloading data from Yahoo
        if start_date is None:
            raise ValueError("Provide a start_date.")
        
        # check end_date for downloading data from Yahoo, if it is None choose current date
        if end_date is None:
            end_date = dt.date.today().isoformat()

        # Fix Yahoo Finance tickers 
        if index_name in ["nasdaq100", "s&p500"]:
            tickers = [t.replace('.', '-') for t in tickers]
            
        print(f"Downloading data from Yahoo Finance ({start_date} -> {end_date})")
        # download data from yahoo
        data =  yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
        # extract adjusted close prices
        if isinstance(data.columns, pd.MultiIndex):
            prices = data.xs('Adj Close', axis=1, level=1)
        else:
            prices = data[['Adj Close']]
        
        # filters assets with missing datas 
        prices = prices.dropna(axis = 1, how = "any")

        # actualize tickers
        tickers = prices.columns.tolist()

    
    # --- CSV ---
    elif source.lower() == "csv":
        # chceck file path
        if file_path is None:
            raise ValueError("Provide a file_path when loading CSV data.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # load data from csv file
        print(f"Loading data from CSV: {file_path}")
        prices = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # --- EXCEL ---
    elif source.lower() == "excel":
        # chceck file path
        if file_path is None:
            raise ValueError("Provide a file_path when loading Excel data.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel file not found: {file_path}")

        # load data from excel file
        print(f"Loading data from Excel: {file_path}")
        prices = pd.read_excel(file_path, index_col=0, parse_dates=True, sheet_name=sheet_name)

    # --- TXT ---
    elif source.lower() == "txt":
        # chceck file path
        if file_path is None:
            raise ValueError("Provide a file_path when loading TXT data.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TXT file not found: {file_path}")

        # load data from txt file
        print(f"Loading data from TXT: {file_path}")
        prices = pd.read_csv(file_path, index_col=0, parse_dates=True, delim_whitespace=True)

    # --- TSV ---
    elif source.lower() == "tsv":
        # chceck file path
        if file_path is None:
            raise ValueError("Provide a file_path when loading TSV data.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TSV file not found: {file_path}")

        # load data from tsv file
        print(f"Loading data from TSV: {file_path}")
        prices = pd.read_csv(file_path, index_col=0, parse_dates=True, sep="\t")
    # error if source = None
    else:
        raise ValueError("Invalid source. Choose from 'yahoo', 'csv', 'excel', 'txt' or 'tsv'.") 
    
    # --- load weights input ---
    if w_0 is not None:
        # load w_0 depending on type of file
        if w_0.endswith(".csv"):
            w0_df = pd.read_csv(w_0, header=None)
        elif w_0.endswith(".xlsx"):
            w0_df = pd.read_excel(w_0, header=None)
        elif w_0.endswith(".txt") or w_0.endswith(".tsv"):
            w0_df = pd.read_csv(w_0, header=None, sep=r"\s+")
        else:
            raise ValueError("Unsupported file type for w_0. Use .csv, .xlsx, .txt, or .tsv.")

        w0 = w0_df.values.flatten()

    # --- expected return of the market ---
    r_m = None
    if market is not None:
        # download market data from yahoo
        if source.lower() == "yahoo": 
            # Download market prices directly from Yahoo
            market_download = yf.download(market, start=start_date, end=end_date, auto_adjust=False)

            if 'Adj Close' in market_download.columns:
                p_m = market_download['Adj Close']
            else:
                p_m = market_download.xs('Adj Close', axis=1, level=1) if isinstance(market_download.columns, pd.MultiIndex) else market_download['Close']
        else: 
            # expect that market data are in prices data
            if market in prices.columns:
                p_m = prices[market]
                # if market is found in prices, remove it from prices
                prices = prices.drop(columns=[market])
                # actualize tickers
                tickers = prices.columns.tolist()
            else:
                raise ValueError(f"Market ticker '{market}' not found in the provided file. "
                                 f"Ensure the column exists or use source='yahoo'.")
        
        p_m.name = "Market_Index"
            
        # Ensure Series 
        if isinstance(p_m, pd.DataFrame):
            p_m = p_m.iloc[:, 0]
        # Compute daily market returns
        if return_type.lower() == "lin":
            daily_market = p_m.pct_change().dropna()
        elif return_type.lower() == "log":
            daily_market = np.log(p_m / p_m.shift(1)).dropna()
        else:
            raise ValueError("return_type must be 'lin' or 'log'.")
        
        # set gamma for 1 if it is None
        if gamma is None:
            gamma = 1 

        # calculate expected market return 
        if method == "geometric_mean":
            geo_r_m = (1 + daily_market).prod()**(1 / len(daily_market)) - 1
            r_m = (1 + geo_r_m)**gamma - 1
        elif method == "exponential_mean":
            if alpha is None:
                raise ValueError("Provide alpha for exponential average.")
            # Recursive 
            mu = daily_market.iloc[0]
            for r in daily_market.iloc[1:]:
                mu = (1 - alpha) * r + alpha * mu
            r_m = mu * gamma
        elif method == "mean":
            r_m = daily_market.mean() * gamma
        else:
            raise ValueError("Unidentified method for calculating retuns of market. Specify geometric_mean, exponential_mean or mean.")
        # r_m can be annualized by gamma 
        market_data = {"daily": daily_market, "r_m": r_m}  
    else:
        market_data = None

    # linear or log returns of assets
    if return_type == "lin":
        returns = prices.pct_change().dropna()
    elif return_type == "log":
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("return_type must be 'lin' or 'log'.")
        
    # aligment of assets data with market
    if market_data is not None:
        combined = pd.concat([returns, market_data['daily']], axis=1).dropna()
        returns = combined.drop(columns=[market_data['daily'].name])
        market_data['daily'] = combined[market_data['daily'].name]

    # return data
    if w_0 is not None and market_data is not None:
        return returns, tickers, market_data, w0
    elif w_0 is not None:
        return returns, tickers, w0
    elif market_data is not None:
        return returns, tickers, market_data
    else:
        return returns, tickers
    

# ==========================================
# Calculate expected returns of assets
# ==========================================
def expected_return(method = None, returns = None, alpha = None, r_f = None, market_returns = None, gamma = None):
    """
    Module for estimating expected asset returns.

    Inputs:
        - method (str): Calculation of expected market return ('mean', 'geometric_mean', 'exponential_mean').
        - returns (DataFrame): Historical assets returns used for parameter estimation.
        - alpha (float): Parameter for the exponential recursive weighted mean.
        - r_f (float): Risk-free rate required for the CAPM method.
        - market_returns (dict): Dictionary containing daily and expected market returns for CAPM.
        - gamma (float): Scalar for returns (trading days per year, default is 1).

    Returns:
        - mu (ndarray): Vector of expected returns for all assets in the portfolio.

    """

    # check returns
    if returns is None:
        raise Exception("Provide returns.")

    # set gamma if necessary
    if gamma is None:
        gamma = 1

    # calculate expected returns of assets
    if method == "geometric_mean":
        geo_mean_daily = (1 + returns).prod()**(1 / len(returns)) - 1
        mu = (1 + geo_mean_daily)**gamma - 1

    elif method == "exponential_mean":
        if alpha is None:
            raise ValueError("Provide alpha for exponential_mean.")
        # Recursive exponentional mean
        mu = returns.iloc[0]
        for r in returns.iloc[1:]:
            mu = (1 - alpha) * r + alpha * mu

    elif method == "CAPM":
        if market_returns is None or r_f is None:
            raise Exception("Provide both risk-free rate and market returns for CAPM")
        
        if isinstance(market_returns, dict):
            daily_market = market_returns["daily"]
            r_m = market_returns["r_m"]
        else:
            raise Exception("market_returns should be a dict with keys 'daily' and 'r_m'")
        
        # Align the two time series
        common_idx = returns.index.intersection(daily_market.index)
        returns = returns.loc[common_idx]
        daily_market = daily_market.loc[common_idx]
        # Compute covariances and betas
        cov_matrix = returns.apply(lambda x: np.cov(x, daily_market)[0, 1])
        var_m = np.var(daily_market, ddof=1)
        betas = cov_matrix / var_m
        mu = r_f + betas * (r_m - r_f)

    elif method == "mean":
        mu = returns.mean().values * gamma
    else:
        raise ValueError("Unidentifed method for calculation of return. Input mean, geometric_mean, exponentional_mean or CAPM.")

    return mu 

# ==========================================
# Calculate expected risk (variance or semivariance)
# ==========================================
def expected_risk(risk_measure = None, returns = None, tau = None, gamma = None):
    """
    Module for estimating variance or semivariance.

    Inputs:
        - risk_measure (str): Chosen metric ('variance' for total risk, 'semivariance' for downside risk).
        - returns (DataFrame): Historical asset returns.
        - tau (float/array): Minimum acceptable return. If None, asset mean is used.
        - gamma (int): Scalar for possible annualization (default is 1).

    Returns:
        - risk (ndarray): Covariance or semi-covariance matrix.
    """
    
    # check returns
    if returns is None:
        raise Exception("Provide assets returns.")
    
    # set gamma if it is None
    if gamma is None:
        gamma = 1

    # calculate variance or semivariance
    if risk_measure == "variance":
        risk = returns.cov().values * gamma
    elif risk_measure == "semivariance":
        risk = semivariance(returns, tau = tau, gamma = gamma)
    else:
        raise ValueError("Unknown risk measure")
        
    return risk

# ==========================================
# Calculate semivariance metrix
# ==========================================
def semivariance(returns = None, tau = None, gamma = None):
    """
    Computes the Semi-covariance matrix for Downside Risk analysis.
    
    Inputs:
        - returns (DataFrame): Historical asset returns.
        - tau (float/array): Minimum acceptable return. If None, asset mean is used.
          Values below tau are treated as risk, values above tau as 0.
        - gamma (float): Scalar for possible annualization (default is 1).

    Returns:
        - M (ndarray): Semi-covariance matrix (T x N).
    """

    # check returns
    if returns is None:
        raise ValueError("Provide returns")
    
    T, N = returns.shape
    # calculate tau as mean of returns if it is not set
    if tau is None:
        tau = returns.mean(axis = 0)
    elif np.isscalar(tau):
        tau = np.full(N, tau)

    # calculate semivariance
    R = np.maximum(tau - returns, 0)
    M = (R.T @ R) / T

    # can be annualized with gamma
    return M * gamma 

# ==========================================
# Calculate alternative risk measures and define contraints (CVaR, EVaR, CVaR-DD or EVaR-DD)
# ==========================================
def alternative_risk_measures(risk_measure = None, w = None, returns = None, alpha_tail = None, alpha_drawdown = None, drawdown_return_type = None):
    """
    Module for alternative risk measures.
    
    Inputs:
        - risk_measure (str): Risk metric to minimize (cvar, evar, ave-dd, cvar-dd).
        - w: weight vetor 
        - returns (DataFrame): Historical daily returns of the assets.
        - alpha_tail (float): Confidence level for CVaR/EVaR tail risk (default: 0.95).
        - alpha_drawdown (float): Confidence level for Drawdown-based risk (default: 0.95).
        - drawdown_return_type (str): Cumulative return method (linear_compound, linear_simple, log).

    Returns: 
        - risk_expression 
        - extra_constraints 
    """
    r_t = returns.values
    T, n = r_t.shape
    constraints = []

    # --- Tail based ---
    if risk_measure in ["cvar", "evar"]:
        # set alpha as 0.95% if it is None
        if alpha_tail is None:
            alpha_tail = 0.95
        
        # chceck if alpha is not out of bounds
        if alpha_tail > 1 or alpha_tail < 0:
            raise ValueError("alpha_tail is out of bounds. Set it in iterval [0,1]")

        # --- CVaR ---
        if risk_measure == "cvar":
            tau = cp.Variable()
            u_t = cp.Variable(T)
            constraints += [
                u_t >= -r_t @ w - tau,
                u_t >= 0
            ]
            risk_expr = tau + (1 / ((1 - alpha_tail) * T)) * cp.sum(u_t)

        # --- EVaR ---
        else:
            t_var = cp.Variable(pos=True)
            s = cp.Variable()
            u_exp = cp.Variable(T, nonneg=True)
            for i in range(T):
                constraints.append(cp.ExpCone(-r_t[i, :] @ w - s, t_var, u_exp[i]))
            constraints.append(t_var >= cp.sum(u_exp))
            risk_expr = s - t_var * np.log((1 - alpha_tail) * T)

    # --- Drawdown ---
    elif risk_measure in ["ave-dd", "cvar-dd"]:
        # set alpha as 0.95% if it is None
        if alpha_drawdown is None:
                alpha_drawdown = 0.95 

        # chceck if alpha is not out of bounds
        if alpha_drawdown > 1 or alpha_drawdown < 0:
            raise ValueError("alpha_drawdown is out of bounds. Set it in iterval [0,1]")

        # cumulative returns
        if drawdown_return_type == "linear_compound":
            r_t_cum = np.cumprod(1 + r_t, axis=0) - 1
        elif drawdown_return_type in ["linear_simple", "log"]:
            r_t_cum = np.cumsum(r_t, axis=0)
        else:
            raise ValueError("Invalid drawdown_type.")
            
        u_t = cp.Variable(T)
        s = cp.Variable()
        
        # --- Ave-DD ---
        if risk_measure == "ave-dd":
            constraints += [
                cp.sum(u_t) / T <= cp.sum(r_t_cum @ w) / T + s,
                r_t_cum @ w <= u_t,
                u_t[:-1] <= u_t[1:]
            ]
            risk_expr = s

        # --- CVaR-DD ---
        else: 
            tau = cp.Variable()
            z_t = cp.Variable(T, nonneg=True)
            constraints += [
                s >= tau + cp.sum(z_t) / (T * (1 - alpha_drawdown)),
                z_t >= u_t - r_t_cum @ w - tau,
                r_t_cum @ w <= u_t,
                u_t[:-1] <= u_t[1:]
            ]
            risk_expr = s
    else:
        raise ValueError("Provide risk_measure.")
            
    return risk_expr, constraints

# ==========================================
# Portfolio optimisation based on tradeoff between return and risk
# ==========================================
def portfolio_optimizer(mu = None, returns = None, return_method = None, r_f = None, market_returns = None, gamma = None,
                        risk_measure = None, tau = None, alpha_tail = None, drawdown_return_type = None, alpha_drawdown = None,
                        Lambda = None, alpha = None, beta = None, bisection = None, Lambda_max = None, 
                        k = None, u = None, l = None, 
                        u_long = None, u_short = None, l_long = None, l_short = None, 
                        k_turnover = None, w_0 = None, D = None, D_long = None, D_short = None, L_max = None, M = None, 
                        A = None, B = None, 
                        points = None, highlight_choice = None, eff_frontier = None,
                        Solver = None, timer = None):
    """
    Portfolio Optimization module based on Convex Programming.

    This module constructs the objective function and constraints for various 
    risk-return frameworks. It supports efficient frontier generation and 
    specific parameter calibration.

    Inputs:
        - mu (DataFrame): Expected returns of assets.
        - returns (DataFrame): Historical daily returns of the assets.
        - return_method (str): Methodology for expected returns (mean, geometric_mean, exponential_mean, CAPM).
        - r_f (float): Risk-free rate (annualized), used for CAPM and Sharpe ratio calculations.
        - market_returns (dict): Market benchmark data (daily/annual) for equilibrium models.
        - gamma (int): Scalar for possible annualization (default is 1).
        
        - risk_measure (str): Risk metric to minimize (variance, semivariance, cvar, evar, ave-dd, cvar-dd).
        - tau (float): Minimal Acceptable Return (MAR) for downside risk measures.
        - alpha_tail (float): Confidence level for CVaR/EVaR tail risk (default: 0.95).
        - drawdown_return_type (str): Cumulative return method (linear_compound, linear_simple, log).
        - alpha_drawdown (float): Confidence level for Drawdown-based risk (default: 0.95).

        - Lamnbda (float): Specific risk aversion parameter (lambda) to solve and highlight.  av
        - alpha (float): Target maximum volatility (used to back-calculate lambda).
        - beta (float): Target minimum return (used to back-calculate lambda).
        - bisection (str): Choose "Y" if you want to use bisection for alpha or beta.
        - av_max (float): Choose upper value for bisection in first iteration (default is 100).

        - u / l (float): Upper and lower bounds for individual asset weights (long position only).
        - u_long / l_long (float): Bounds for the positive part of weights (default for l_long is 0.
        - u_short / l_short (float): Bounds for the negative part of weights (default for l_short is 0).
        - k (float): Proportion of total long positions.
        
        - k_turnover (float): Maximum allowed L1-norm change from the initial portfolio w_0.
        - w_0 (array): Initial portfolio weights for turnover and rebalancing analysis.
        - D, D_long, D_short (float): Diversification limit.
        - L_max (float): Maximum portfolio leverage.
        - M (int): Cardinality constraint (limit on the number of non-zero weights).
        - K (int): Big-M constant used for linearization of cardinality constraints.
        - A / B (float): Concentration bounds (A% of wealth must be in at least B% of assets).

        - points (int): Number of lambda values to solve for the Efficient Frontier.
        - highlight_choice (list): Specific lambda values to be labeled on the plot.
        - eff_frontier (str): Choose "Y" if you want graph of efficient frontier for Markowitz formulation. 
        - Solver (str): Solver selection (e.g., 'MOSEK', 'GUROBI', 'CPLEX').
        - timer (int): Time limit in seconds for the optimization solver.

    Returns:
        - obj_return (float): Expected return of portfolio.
        - obj_risk (float): Risk of portfolio.
        - w (ndarray): Optimal weight vector.

    """

    # calculate expected returns (mu) if not provided, using the specified return_method
    if mu is None:
        if returns is None:
            raise ValueError("Provide returns.")

        if return_method is None:
            raise ValueError("Provide return_method.")
        else:
            if return_method == "CAPM" and (r_f is None or market_returns is None):
                raise ValueError("Provide risk free rate and market returns for CAPM.")

        mu = expected_return(method = return_method, returns = returns, r_f = r_f, market_returns = market_returns, gamma = gamma)

    # check if risk measure is set 
    if risk_measure is None:
        raise ValueError("Provide risk measure.")

    n = len(mu)
    one = np.ones(n)

    # ==========================================
    # Formulate problem
    # ==========================================
    # enable shorting
    if k is not None:

        if u_long is None or l_long is None or u_short is None or l_short is None:
            raise ValueError("You need to provide constraints for short position.")

        # split weights into long and short componets
        w_long = cp.Variable(n)
        w_short = cp.Variable(n)
        w = w_long - w_short

        constraints += [
            cp.sum(w_long) - cp.sum(w_short) == 1,
            cp.sum(w_short) <= k * cp.sum(w_long)
        ]

        # cardinality constraints for short position
        if M is not None:

            if u_long is None or u_short is None or l_long is None or l_short is None:
                raise ValueError("You need to set constraints for cardinality with short position.")

            z_long = cp.Variable(n, boolean=True)
            z_short = cp.Variable(n, boolean=True)

            if u_long < 1/n:
                raise ValueError("Constraint is out of bound, u_long is smaller than 1/N.")
            elif u_long > 1:
                raise ValueError("Constraint is out of bound, u_long is bigger than 1.")
            else:
                constraints += [
                    w_long >= l_long * z_long,
                    w_long <= u_long * z_long,
                    w_short >= l_short * z_short,
                    w_short <= u_short * z_short,
                    cp.sum(z_long + z_short) <= M
                ]

        # short position without cardinality constraints
        else:
            if l_long is not None:
                constraints.append(w_long >= l_long)
            else:
                constraints.append(w_long >= 0)

            if l_short is not None:
                constraints.append(w_short >= l_short)
            else:
                constraints.append(w_short >= 0)

            if u_long is not None:
                if u_long < 1/n:
                    raise ValueError("Constraints is out of bound, u_long is smaller than 1/N.")
                elif u_long > 1:
                    raise ValueError("Constraints is out of bound, u_long is bigger than 1.")
                else:
                    constraints.append(w_long <= u_long)

            if u_short is not None:
                constraints.append(w_short <= u_short)

        if L_max is not None:
            constraints.append(cp.norm1(w) <= L_max)

        # diversification constraint for long and short position separately
        if D_long is not None or D_short is not None:
            if D_long < 1/n:
                raise ValueError("Constraints is out of bound, D_long is smaller than 1/N.")
            elif D_long > 1:
                raise ValueError("Constraints is out of bound, D_long is bigger than 1.")
            else:
                constraints.append(cp.sum_squares(w_long) <= D_long)

            if D_short < 1/n:
                raise ValueError("Constraints is out of bound, D_short is smaller than 1/N.")
            elif D_short > 1:
                raise ValueError("Constraints is out of bound, D_short is bigger than 1.")
            else:
                constraints.append(cp.sum_squares(w_short) <= D_short)

    # long position only
    else:
        w = cp.Variable(len(mu))
        constraints = [cp.sum(w) == 1]

        # cardinality constraints for long position only
        if M is not None:

            if u is None or l is None:
                raise ValueError("You need to set constraints for cardinality.")

            z = cp.Variable(n, boolean=True)

            if u is not None:
                if u < 1/n:
                    raise ValueError("Constraints is out of bound, u is smaller than 1/N.")
                elif u > 1:
                    raise ValueError("Constraints is out of bound, u is bigger than 1.")
                else:
                    constraints += [
                        w >= l * z,
                        w <= u * z,
                        cp.sum(z) <= M
                    ]
        else:
            if l is not None:
                constraints.append(w >= l)
            else:
                constraints.append(w >= 0)

            if u is not None:
                if u < 1/n:
                    raise ValueError("Constraints is out of bound, u is smaller than 1/N.")
                elif u > 1:
                    raise ValueError("Constraints is out of bound, u is bigger than 1.")
                else:
                    constraints.append(w <= u)

        # diversification constraint
        if D is not None:
            if D < 1/n:
                raise ValueError("Constraints is out of bound, D is smaller than 1/N.")
            elif D > 1:
                raise ValueError("Constraints is out of bound, D is bigger than 1.")
            else:
                constraints.append(cp.sum_squares(w) <= D)

    # turnover constraint
    if k_turnover is not None:
        if w_0 is None:
            raise ValueError("You need to privode vector w_0.")
        constraints.append(cp.norm1(w - w_0) <= k_turnover)

    # concentreition constraint
    if A is not None and B is not None:
        u_ = cp.Variable(len(mu))
        u_0 = cp.Variable(1)
        constraints += [
            cp.sum(u_) + B * n * u_0 <= A,
            u_ + u_0 * one >= w,
            u_ >= 0
        ]

    # ==========================================
    # Efficient frontier
    # ==========================================
    # using logarithmic scale for effective selection of points from wide interval
    if points is not None:
        Lambda_par_values = np.logspace(np.log10(1e-2), np.log10(1e6), num = points)
    else:
        Lambda_par_values = np.logspace(np.log10(1e-2), np.log10(1e6), num = 200)

    # choose points which will be highlighted on efficient frontier
    if risk_measure in ["variance"] and eff_frontier in ["Y"]:
        if highlight_choice is None:
            highlight_values = [1, 5, 10, 20, 100]
        else:
            highlight_values = highlight_choice

        highlight_indices = [
            np.argmin(np.abs(Lambda_par_values - hv))
            for hv in highlight_values
        ]

    # ==========================================
    # Assign risk measure and costraints
    # ==========================================
    # assign risk measure
    if risk_measure in ["variance", "semivariance"]:
        matrix = expected_risk(risk_measure = risk_measure, returns = returns, tau = tau, gamma = gamma)
        risk_expr = cp.quad_form(w, matrix)

    elif risk_measure in ["cvar", "evar", "ave-dd", "cvar-dd"]:
        risk_expr, extra_constraints = alternative_risk_measures(
        risk_measure = risk_measure, w = w, returns = returns, 
        alpha_tail = alpha_tail, alpha_drawdown = alpha_drawdown, drawdown_return_type = drawdown_return_type
        )
        constraints += extra_constraints

    else:
        raise ValueError("Provide risk_measure.")

    # define optimization problem
    lambda_param = cp.Parameter(nonneg=True)
    objective_eff = cp.Maximize(w @ mu - lambda_param * risk_expr)
    problem_eff = cp.Problem(objective_eff, constraints)

    frontier_returns = np.zeros(len(Lambda_par_values))
    frontier_risks = np.zeros(len(Lambda_par_values))
    frontier_all_weights = []  

    # ==========================================
    # Solve efficient frontier
    # ==========================================
    for i, Lambda_value in enumerate(Lambda_par_values):

        lambda_param.value = Lambda_value

        if Solver is not None:
            name = Solver.upper()

            if timer is None:
                timer = 60

            if name == "MOSEK":
                problem_eff.solve(
                    solver = cp.MOSEK,
                    mosek_params = {"MSK_DPAR_OPTIMIZER_MAX_TIME": timer}
                )
            elif name == "GUROBI":
                problem_eff.solve(solver = cp.GUROBI, TimeLimit = timer)
            elif name == "CPLEX":
                problem_eff.solve(solver = cp.CPLEX, timelimit = timer)
            else:
                problem_eff.solve()
        else:
            problem_eff.solve()

        if w.value is None:
            frontier_returns[i] = np.nan
            frontier_risks[i] = np.nan
            print(f"No solution for λ = {Lambda_value:.4f}")
            continue

        obj_return = w.value @ mu
        obj_risk = risk_expr.value

        frontier_returns[i] = obj_return
        frontier_risks[i] = obj_risk
        frontier_all_weights.append(w.value.copy()) 

        print(f"λ = {Lambda_value:.4f} | Return = {obj_return:.8f} | Risk = {obj_risk:.8f}")
        print("Weighted vector:")
        print(np.round(w.value, 4))

    # ==========================================
    # Find lambda in Markowitz formulation for specific alpha or beta 
    # ==========================================
    # bisection
    if alpha is not None or beta is not None:
        if bisection == "Y":

            if Lambda_max is None:
                Lambda_max = 1e2

            lambda_min, lambda_max = 0, Lambda_max
            tolerance = 1e-5
            
            for i in range(50):
                mid = (lambda_min + lambda_max) / 2
                lambda_param.value = mid
                
                if Solver is not None:
                    name = Solver.upper()
                    if timer is None:
                        timer = 60
                    if name == "MOSEK":
                        problem_eff.solve(
                            solver = cp.MOSEK,
                            mosek_params = {"MSK_DPAR_OPTIMIZER_MAX_TIME": timer}
                        )
                    elif name == "GUROBI":
                        problem_eff.solve(solver = cp.GUROBI, TimeLimit = timer)
                    elif name == "CPLEX":
                        problem_eff.solve(solver = cp.CPLEX, timelimit = timer)
                    else:
                        problem_eff.solve()
                else:
                    problem_eff.solve()

                if w.value is None:
                    lambda_max = mid 
                    continue
                
                current_val = risk_expr.value if alpha is not None else w.value @ mu
                target = alpha if alpha is not None else beta

                if current_val > target:
                    lambda_min = mid
                else:
                    lambda_max = mid

                if abs(current_val - target) < tolerance:
                    break

            Lambda = mid
            print(f"Bisection finished: λ = {Lambda:.4f}")

        # analytical method for long and short position without additional constraints
        else:         
            if risk_measure != "variance":
                raise ValueError("Analytical method works only for variance. Otherwise use bisection.")
            
            matrix_inv = np.linalg.inv(matrix)
            a = one @ matrix_inv @ mu
            b = one @ matrix_inv @ one
            c = mu @ matrix_inv @ mu

            if alpha is not None:
                if alpha <= 1 / b:
                    raise ValueError("α is too small, real λ can't be found.")

                Lambda = np.sqrt((c * b - a**2) / (4 * b * (alpha - 1 / b)))
                print(f"For α = {alpha:.4f}, λ = {Lambda:.4f}")

            elif beta is not None:
                if beta <= a / b:
                    raise ValueError("β is too small, real λ can't be found.")

                Lambda = (c * b - a**2) / (2 * b * (beta - a / b))
                print(f"For β = {beta:.4f}, λ = {Lambda:.4f}")

    # ==========================================
    # Solve for specific lambda 
    # ==========================================
    if Lambda is not None:

        # find solution if bisection was not used
        if w.value is None or not np.isclose(lambda_param.value, Lambda):
            lambda_param.value = Lambda

            # use specific solver or cvxpy default solver
            if Solver is not None:
                name = Solver.upper()
                if timer is None:
                    timer = 60
                if name == "MOSEK":
                    problem_eff.solve(
                        solver = cp.MOSEK,
                        mosek_params = {"MSK_DPAR_OPTIMIZER_MAX_TIME": timer}
                    )
                elif name == "GUROBI":
                    problem_eff.solve(solver = cp.GUROBI, TimeLimit = timer)
                elif name == "CPLEX":
                    problem_eff.solve(solver = cp.CPLEX, timelimit = timer)
                else:
                    problem_eff.solve()
            else:
                problem_eff.solve()

        if w.value is not None:

            obj_return = w.value @ mu
            obj_risk = risk_expr.value

            print(f"λ = {Lambda:.4f} | Return = {obj_return:.8f} | Risk = {obj_risk:.8f}")
            print("Weighted vector:")
            print(np.round(w.value, 4))

            if risk_measure in ["variance"]:
                obj_vol = np.sqrt(obj_risk)

    # ==========================================
    # Generating random portfolios
    # ==========================================
    if risk_measure == "variance" and len(frontier_all_weights) > 10 and eff_frontier == "Y":
        all_fill_weights = []
        
        # prepare data for sampling  
        frontier_vols = np.sqrt(frontier_risks)
        f_vols = np.array(frontier_vols)
        f_rets = np.array(frontier_returns)
        f_weights = np.array(frontier_all_weights)

        # create a uniform subset of points along the frontier to avoid clustering near the tails
        n_resampled = 50
        indices = [0]
        
        # calculate target distance between points on the frontier curve
        target_dist = np.sum(np.sqrt(np.diff(f_vols)**2 + np.diff(f_rets)**2)) / n_resampled
        
        current_dist = 0
        for i in range(1, len(f_vols)):
            d = np.sqrt((f_vols[i] - f_vols[i-1])**2 + (f_rets[i] - f_rets[i-1])**2)
            current_dist += d
            if current_dist >= target_dist:
                indices.append(i)
                current_dist = 0
        if (len(f_vols) - 1) not in indices:
            indices.append(len(f_vols) - 1)
            
        pts_uniform = f_weights[indices]

        # generating points randomly
        max_attempts = 40000  # safety break for the loop
        target_count = 5000  # number of random portfolios to generate
        attempts = 0

        while len(all_fill_weights) < target_count and attempts < max_attempts:
            attempts += 1
            
            # select two random points from the frontier to create a base linear combination
            idx = np.random.choice(len(pts_uniform), 2, replace=False)
            w1, w2 = pts_uniform[idx[0]], pts_uniform[idx[1]]
            
            alpha = np.random.rand()
            w_base = alpha * w1 + (1 - alpha) * w2
            
            # filter points by cardinality constraints
            if M is not None:
                noise = np.zeros(n)
                indices_k = np.random.choice(n, int(M), replace=False)
                noise[indices_k] = np.random.dirichlet(np.ones(int(M)) * 0.5)
            else:
                noise = np.random.dirichlet(np.ones(n) * 0.5)
            
            r_param = np.random.beta(1, 3) 
            w_rand = (1 - r_param) * w_base + r_param * noise
            w_rand = np.maximum(w_rand, 0)
            w_rand /= np.sum(w_rand)

            # filter points by additional contraints
            valid = True
            
            if u is not None:
                if np.any(w_rand > u):
                    valid = False

            if valid and l is not None:
                if np.any(w_rand < l):
                    valid = False

            if L_max is not None and np.sum(np.abs(w_rand)) > L_max:
                valid = False

            if valid and k_turnover is not None and w_0 is not None:
                if np.sum(np.abs(w_rand - w_0)) > k_turnover:
                    valid = False

            if valid and D is not None:
                if np.sum(w_rand**2) > D:
                    valid = False

            if valid and A is not None and B is not None:
                n_B = int(np.ceil(B * n))
                top_weights = np.sort(w_rand)[-n_B:]
                if np.sum(top_weights) > A + 1e-4:
                    valid = False

            if valid:
                all_fill_weights.append(w_rand)

        random_weights = np.vstack(all_fill_weights)
        port_returns = random_weights @ mu
        port_vols = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, matrix, random_weights))

    # ==========================================
    # Plot effective frontier with random portfolios
    # ==========================================
    if risk_measure in ["variance"] and eff_frontier in ["Y"]:   # pridat podmienku na vykreslenie grafu 
        frontier_vols = np.sqrt(frontier_risks)

        plt.figure(figsize=(11, 7))
        plt.grid(True, linestyle='--', color='lightgray', alpha=0.8, zorder=0)

        # plot efficient frontier and random portfolios
        plt.scatter(port_vols, port_returns, s=3, c='royalblue', alpha=0.5, marker='o', edgecolors='none', label='Random portfolios')
        plt.plot(frontier_vols, frontier_returns, color='black', linewidth=1.2, label='Efficient frontier', zorder=2)

        # plot highlighted values
        for i, (x, y) in enumerate(zip(frontier_vols, frontier_returns)):
            if i in highlight_indices:
                plt.scatter(x, y, color='black', edgecolors='white', s=10, zorder=3)
                plt.annotate(f"λ={Lambda_par_values[i]:.0f}", (x, y),
                             xytext=(-8, -2), textcoords='offset points', 
                             fontsize=9, fontweight='bold', ha='right')

        # plot specific lambda for av, aplha or beta
        if Lambda is not None and w.value is not None:
            plt.scatter(obj_vol, obj_return, color='red', marker='*', s=250, 
                        label='Optimal portfolio (Target)', edgecolors='black', zorder=4)
            plt.annotate(f"λ={Lambda:.2f}", (obj_vol, obj_return),
                         xytext=(-8, -2), textcoords='offset points', 
                         color='red', fontsize=10, fontweight='bold', ha='right', va='center')

        plt.xlabel('Annual volatility')
        plt.ylabel('Return')
        plt.legend(loc='best', frameon=True, shadow=True)
        plt.tight_layout()
        plt.show()

    # return values for specific lambda
    if Lambda is not None:
        return obj_return, obj_risk, w
        
##################################################################################################################
# SHARPE RATIO
##################################################################################################################
def max_sharpe(mu = None, returns = None, return_method = None, market_returns = None, r_f = None, gamma = None,
              k = None, u = None, l = None, u_long = None, u_short = None, l_long = None, l_short = None,
              k_turnover = None, w_0 = None, D = None, D_long = None, D_short = None, L_max = None, M = None, K = None, 
              Solver = None, timer = None):
    """
    Maximum Sharpe Ratio.

    Inputs:
        - mu (DataFrame): Expected returns of assets.
        - returns (DataFrame): Historical asset returns.
        - return_method (str): Methodology for expected returns (mean, geometric, exponential, CAPM).
        - r_f (float): Risk-free rate (benchmark for the excess return).
        - market_returns (dict): Market data for equilibrium models.
        - gamma (int): Annualization factor (default is 1).

        - u / l (float): Upper and lower bounds for individual asset weights (long position only).
        - u_long / l_long (float): Bounds for the positive part of weights (default for l_long is 0.
        - u_short / l_short (float): Bounds for the negative part of weights (default for l_short is 0).
        - k (float): Proportion of total long positions.
        
        - k_turnover (float): Maximum deviation from the initial portfolio w_0.
        - w_0 (array): Reference weights for turnover constraints.
        - D, D_long, D_short (float): Diversification threshold (Herfindahl-Hirschman index).
        - L_max (float): Maximum gross leverage allowed.
        - M (int): Cardinality constraint (limit on the number of non-zero weights).
        - K (int): Big-M constant used for linearization of cardinality constraints.
        
        - Solver (str): Optimization backend (MOSEK, GUROBI, CPLEX).
        - timer (int): Maximum CPU time for the solver in seconds.

    Returns:
        - obj_return (float): Expected return of portfolio.
        - obj_risk (float): Risk of portfolio.
        - sharpe (float): Sharpe ratio of portfolio.
        - w (ndarray): Optimal weight vector.

    """

    if mu is None:
        if returns is None:
            raise ValueError("Provide returns.")

        if return_method is None:
            raise ValueError("Provide return_method.")
        else:
            if return_method == "CAPM" and (r_f is None or market_returns is None):
                raise ValueError("Provide risk free rate and market returns for CAPM.")

        mu = expected_return(method = return_method, returns = returns, r_f = r_f, market_returns = market_returns, gamma = gamma)
    
    # calculation of Sigma 
    Sigma = expected_risk(risk_measure = "variance", returns = returns, gamma = gamma)
    
    n = len(mu)
    ones = np.ones(n)

    # check if risk free rate is not None
    if r_f is None:
        raise ValueError("You have to provide risk free rate.")
    
    # ==========================================
    # Formulation of constraints
    # ==========================================
    # short position
    if k is not None:
        # variables
        y_long  = cp.Variable(n)
        y_short = cp.Variable(n)
        y = y_long - y_short  
        
        constraints = [
            y @ (mu - r_f * ones) == 1,    
            cp.sum(y_long) - cp.sum(y_short) >= 0,   
            cp.sum(y_short) <= k * cp.sum(y_long),  
            y_long >= 0,
            y_short >= 0  
        ]

        # cardinality constraints
        if M is not None:
            if u_long is None or u_short is None or l_long is None or l_short is None:
                raise ValueError("You have to provide constraints for cardinality with short position.")
            # new variables for linearization
            z_long  = cp.Variable(n, boolean=True)
            z_short = cp.Variable(n, boolean=True)
            t = cp.Variable(nonneg=True)
            s_long  = cp.Variable(n, nonneg=True)
            s_short = cp.Variable(n, nonneg=True)
            constraints += [
                cp.sum(y_long) - cp.sum(y_short) == t
            ]

            # linearization
            if K is not None:
                c = K
            else:
                c = 5

            constraints += [
                s_long <= t,
                s_long <= c * z_long,
                s_long >= t - c * (1 - z_long),
                s_short <= t,
                s_short <= c * z_short,
                s_short >= t - c * (1 - z_short),  
            ]
            # ohranicenie kardinality
            if u_long < 1/n:
                raise ValueError("Constraints is out of bound, u_long is smaller than 1/N.")
            elif u_long > 1:
                raise ValueError("Constraints is out of bound, u_long is bigger than 1.")
            else:
                constraints += [
                    y_long >= l_long * s_long,
                    y_long <= u_long * s_long,
                    y_short >= l_short * s_short,
                    y_short <= u_short * s_short,
                    cp.sum(z_long) + cp.sum(z_short) <= M
                ]
        # without cardnality constraints
        else:
            if l_long is not None:
                constraints.append(y_long >= l_long * cp.sum(y))
            else:
                constraints.append(y_long >= 0)
            if l_short is not None:
                constraints.append(y_short >= l_short * cp.sum(y))
            else:
                constraints.append(y_short >= 0)
            if u_long is not None:
                if u_long < 1/n:
                    raise ValueError("Constraints is out of bound, u_long is smaller than 1/N.")
                elif u_long > 1:
                    raise ValueError("Constraints is out of bound, u_long is bigger than 1.")
                else:
                    constraints.append(y_long <= u_long * cp.sum(y))
            if u_short is not None:
                constraints.append(y_short <= u_short * cp.sum(y))

        # leverage contraint
        if L_max is not None:
            constraints.append(cp.norm1(y) <= L_max * cp.sum(y))
        
        # diversification contraint for long and short position separately 
        if D_long is not None or D_short is not None:
            if D_long < 1/n:
                raise ValueError("Constraints is out of bound, D_long is smaller than 1/N.")
            elif D_long > 1:
                raise ValueError("Constraints is out of bound, D_long is bigger than 1.")
            else:
                constraints.append(cp.sum_squares(y_long) <= D_long * cp.sum(y_long)^2)
            
            if D_short < 1/n:
                raise ValueError("Constraints is out of bound, D_short is smaller than 1/N.")
            elif D_short > 1:
                raise ValueError("Constraints is out of bound, D_short is bigger than 1.")
            else:
                constraints.append(cp.sum_squares(y_short) <= D_short * cp.sum(y_short)^2)


    # long position only
    else:
        y = cp.Variable(n)
        constraints = [
            y @ (mu - r_f * ones) == 1
        ]

        # cardinality constraints
        if M is not None:
            if u is None or l is None:
                raise ValueError("You have to choose constraints for cardinality.")
            # new values for linearization
            z = cp.Variable(n, boolean=True)
            t = cp.Variable(nonneg=True)      
            s = cp.Variable(n, nonneg=True)  
            constraints += [cp.sum(y) == t]
            # linearizacia 
            constraints += [
                s <= t,
                s <= K * z,
                s >= t - K * (1 - z),
            ]
            # cardinality constraints
            if u < 1/n:
                raise ValueError("Constraints is out of bound, u is smaller than 1/N.")
            elif u > 1:
                raise ValueError("Constraints is out of bound, u is bigger than 1.")
            else:
                constraints += [
                    y >= l * s,
                    y <= u * s,
                    cp.sum(z) <= M
                ]

        # without cardinality constraints
        else:
            if l is not None:
                constraints.append(y >= l * cp.sum(y))
            else:
                constraints.append(y >= 0)
            if u is not None:
                if u < 1/n:
                    raise ValueError("Constraints is out of bound, u is smaller than 1/N.")
                elif u > 1:
                    raise ValueError("Constraints is out of bound, u is bigger than 1.")
                else:
                    constraints.append(y <= u * cp.sum(y))

        # diversification contraint for long only 
        if D is not None:
            if D < 1/n:
                raise ValueError("Constraints is out of bound, D is smaller than 1/N.")
            elif D > 1:
                raise ValueError("Constraints is out of bound, D is bigger than 1.")
            else:
                constraints.append(cp.sum_squares(y) <= D * cp.sum(y)^2)

    # turnover contraints
    if k_turnover is not None:
        if w_0 is None:
            raise ValueError("You need to privode vector w0.")
        constraints.append(cp.norm1(y - w_0 * cp.sum(y)) <= k_turnover * cp.sum(y))   

    # ==========================================
    # Solve problem
    # ==========================================
    # formulate problem
    objective = cp.Minimize(cp.quad_form(y, Sigma))
    problem = cp.Problem(objective, constraints)

    # choose solver or use default cvxpy solver
    if Solver is not None:
        name = Solver.upper()
        if timer is None:
            timer = 60
        if name == "MOSEK":
            problem.solve(solver = cp.MOSEK, mosek_params = {"MSK_DPAR_OPTIMIZER_MAX_TIME": timer})
        elif name == "GUROBI":
            problem.solve(solver = cp.GUROBI, TimeLimit = {"TimeLimit": timer})
        elif name == "CPLEX":
            problem.solve(solver = cp.CPLEX, timelimit = {"timelimit": timer})
        else:
            problem.solve()
    else:
        problem.solve()

    y_opt = y.value
    w = y_opt / (np.sum(y_opt))
    sharpe = (w @ mu - r_f ) / np.sqrt(w  @ Sigma @ w )
    obj_return = w @ mu
    obj_risk = np.sqrt(w @ Sigma @ w)

    print(f"Return = {obj_return:.8f} | Volatility = {obj_risk:.8f} | Sharpe ratio = {sharpe:.4f}")
    print("Weighted vector =")
    print(np.round(w, 4))

    # returns values
    return obj_return, obj_risk, sharpe, w

# ==========================================
# Module used to ensure that backtest will not stop on error
# ==========================================
def ensure_solution(solve_func = None, prev_weights = None, first_step = None, num_assets = None):
    """
    Ensures the robustness of the portfolio optimization process in case of solver failure.

    Inputs:
        - solve_func (callable): A lambda function calling the specific solver (portfolio_optimizer or max_sharpe).
        - prev_weights (array-like): Portfolio weights from the previous period.
        - first_step (bool): Indicator if this is the initial step of the simulation.
        - num_assets (int): Total number of assets to initialize Equal Weight if needed.

    Returns:
        - weights (array): A vector of optimal weights, previous weights, or equal weights as a final fallback.

    """
    
    # make sure that that necessery inputs are there
    if solve_func is None or first_step is None or num_assets is None:
        raise ValueError("Missing required arguments in ensure_solution.")

    try:
        # start optimization
        weights = solve_func()
        
        # extract value if it exist for numpy
        if hasattr(weights, 'value'):
            weights = weights.value
        
        if weights is None:
            raise ValueError("Solver failed to find a solution.")
        
        return weights 

    except Exception as e:
        print(f"DEBUG ERROR v ensure_solution: {type(e).__name__} - {e}")
        if not first_step and prev_weights is not None:
            # if there are previous weights we will use them
            return prev_weights
        else:
            # if this is first iteration use equal wieghts
            return np.ones(num_assets) / num_assets

# ==========================================
# Porfolio composition
# ==========================================
def plot_portfolio_composition(weights = None, tickers = None, portfolio_name = None):
    """

    Create pie chart from portfolio composition

    Inputs:
        - weights (array/list): Final weiths from optimization. 
        - tickers (list): Name of assets. 
        - portfolio_name (str): Name of x axis for graph. 

    Returns:
        - fig: Plotly figure.

    """
    
    if weights is None or tickers is None:
        raise ValueError("Musíte poskytnúť váhy (weights) a názvy aktív (tickers).")
    
    if portfolio_name is None:
        portfolio_name = "Portfolio"

    # prepare data
    assets = pd.DataFrame({'Asset': tickers, 'Weight': weights})
    
    # filter assets with weights smaller or equal than 0.01 for better visualisation
    assets = assets[assets['Weight'] > 0.01].sort_values(by='Weight', ascending=False)
    
    # create pie chart
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    # define color scheme
    colors = plt.get_cmap('tab20')(range(len(assets)))

    _, texts, autotexts = ax.pie(
        assets['Weight'], 
        labels=assets['Asset'], 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        pctdistance=0.85, 
        wedgeprops={'edgecolor': 'white', 'linewidth': 1} 
    )

    # change writing
    plt.setp(autotexts, size=9, weight="bold", color="black")
    plt.setp(texts, size=10)

    ax.set_title(f"{portfolio_name}", fontsize=14, pad=20)
    
    # make sure pie chart is circle
    ax.axis('equal')  

    plt.tight_layout()
    plt.show()
    
    return fig

# ==========================================
# Table with metric for backtest
# ==========================================
def calculate_metrics(returns = None, r_f = None, V = None):
    """
    Create pie chart from portfolio composition

    Inputs:
        - returns (DataFrame): Historical daily returns of the assets.
        - r_f (float): Risk free rate. 
        - V (float): Minimal required return in Sortino ratio. 

    Returns:
        - fig: Plotly figure.

    """

    # Total return
    total_return = (returns + 1).prod() - 1
    
    # Volatility
    volatility = returns.std() * np.sqrt(252)

    # Semivariance
    downside_returns = returns[returns < 0]
    semi_deviation = downside_returns.std() * np.sqrt(252)

    # Sharpe Ratio
    sharpe = (returns.mean() - r_f) / returns.std() * np.sqrt(252)
    
    # Sortino Ratio 
    downside_diffs = np.where(returns < V, returns - V, 0)
    ttd = np.sqrt(np.mean(downside_diffs**2))
    sortino = (returns.mean() - V) / ttd * np.sqrt(252)

    # CVaR 95%
    alpha = 0.95
    sorted_returns = np.sort(returns)
    cvar_95 = sorted_returns[:int( (1 - alpha) * len(sorted_returns))].mean()
    
    # Maximum Drawdown 
    cum_rets = (1 + returns).cumprod()
    running_max = cum_rets.cummax()
    drawdown_series = (cum_rets / running_max - 1)
    max_drawdown = drawdown_series.min()

    # Average Drawdown 
    avg_drawdown = drawdown_series[drawdown_series < 0].mean()

    # return risk metrics
    return {
        "Return": total_return,
        "Volatility": volatility,
        "Semivolatility": semi_deviation,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "CVaR 95%": cvar_95,
        "Max Drawdown": max_drawdown,
        "Average Drawdown": avg_drawdown
    }

