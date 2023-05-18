#%% determine the expected return and covariance matrix 
import pandas as pd
import numpy as np
import pickle
path_ts_data = r"D:\Py_Prjs\OPA_repo\data\50yr_timeSeries_data.pkl"
path_selected_tickers = r'D:\Py_Prjs\OPA_repo\data\alternate_port.pkl'
# Define the list of stocks in the portfolio
selected_cluster = 5
alternate_port = pickle.load(open(path_selected_tickers,'rb'))
ts_data = pickle.load(open(path_ts_data,'rb'))
stocks = list(alternate_port[f'cluster_{selected_cluster}'].index)

#%% Retrieve the historical price data for each stock using a data source such as Yahoo Finance
prices = pd.DataFrame()
for stock in stocks:
    prices[stock] = ts_data[stocks].loc[:, stock].loc[:, 'Adj Close']

# Calculate the daily returns of each stock
returns = prices.pct_change().dropna()

# Calculate the expected returns for each stock
expected_returns = returns.mean()

# Calculate the covariance matrix of the daily returns
cov_matrix = returns.cov()

print("Expected returns:\n", expected_returns)
print("\nCovariance matrix:\n", cov_matrix)

# %% find the weights
import cvxpy as cp
import numpy as np

# Define the number of stocks and the target return
n_stocks = len(expected_returns)
target_return = 0.0011

# Define the optimization variables
weights = cp.Variable(n_stocks)

# Define the objective function to minimize portfolio variance
portfolio_variance = cp.quad_form(weights, cov_matrix)
objective = cp.Minimize(portfolio_variance)

# Define the constraints
constraints = [cp.sum(weights) == 1, expected_returns.values @ weights >= target_return, weights >= 1/n_stocks]

# Solve the optimization problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Print the optimal weights
print("Optimal weights:", weights.value)