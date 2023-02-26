#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
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

# Normalize the returns using a MinMaxScaler
scaler = MinMaxScaler()
scaled_returns = scaler.fit_transform(returns)

# Split the data into training and testing sets
train_size = int(len(scaled_returns) * 0.8)
train_data, test_data = scaled_returns[:train_size], scaled_returns[train_size:]

# Define the input and output data for the neural network
X_train, y_train = train_data[:-1], train_data[1:]
X_test, y_test = test_data[:-1], test_data[1:]

# Define and train the neural network
model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', random_state=42)
model.fit(X_train, y_train)

# Use the trained neural network to predict the optimal weights for the testing data
predicted_weights = model.predict(X_test)
predicted_weights = predicted_weights[-1] / np.sum(predicted_weights[-1])

print("Predicted weights:", predicted_weights)
