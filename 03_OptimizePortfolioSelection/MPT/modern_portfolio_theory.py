#%%
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import pickle
path_ts_data = r"D:\Py_Prjs\OPA_repo\data\50yr_timeSeries_data.pkl"
path_selected_tickers = r'D:\Py_Prjs\OPA_repo\data\alternate_port.pkl'
# Define the list of stocks in the portfolio
selected_cluster = 5
alternate_port = pickle.load(open(path_selected_tickers,'rb'))
ts_data = pickle.load(open(path_ts_data,'rb'))


# Define the list of stocks in the portfolio
stocks = list(alternate_port[f'cluster_{selected_cluster}'].index)

# Retrieve the historical price data for each stock using a data source such as Yahoo Finance
prices = pd.DataFrame()
for stock in stocks:
    prices[stock] = ts_data[stocks].loc[:, stock].loc[:, 'Adj Close']

# Calculate the daily returns of each stock
returns = prices.pct_change().dropna()

# Calculate the expected returns and covariance matrix of the returns
expected_returns = returns.mean()
cov_matrix = returns.cov()

# Define the objective function for the portfolio optimization problem
def objective_function(weights, expected_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.sum(expected_returns * weights)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
    return -sharpe_ratio

# Set the risk-free rate
risk_free_rate = 0.00001

# Define the bounds for the portfolio weights
bounds = tuple((0, 1) for i in range(len(stocks)))

# Define the constraints for the portfolio weights
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Use the SciPy library to optimize the portfolio weights
result = minimize(objective_function, len(stocks) * [1 / len(stocks)], args=(expected_returns, cov_matrix, risk_free_rate), bounds=bounds, constraints=constraints)

# Print the optimal portfolio weights
print("Optimal weights:", result.x)

# %%
