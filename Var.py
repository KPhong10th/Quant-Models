import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import datetime


def download_data(stock, start_date, end_date):
    # Download the stock data
    ticker_data = yf.download(stock, start_date, end_date)
    print("Downloaded Data Columns:", ticker_data.columns)  # Inspect the data columns

    # Check for relevant column under MultiIndex and return the data
    if ('Adj Close', stock) in ticker_data.columns:
        return ticker_data[('Adj Close', stock)]
    elif ('Close', stock) in ticker_data.columns:
        return ticker_data[('Close', stock)]
    else:
        raise KeyError(f"'Adj Close' or 'Close' column not found for {stock} in the downloaded data")


def calculate_var(position, confidence, mu, sigma):
    """Calculate Value at Risk (VaR) for one day."""
    var = position * (mu - sigma * norm.ppf(1 - confidence))
    return var


def calculate_var_n(position, confidence, mu, sigma, n):
    """Calculate Value at Risk (VaR) for multiple days."""
    var = position * (mu * n - sigma * np.sqrt(n) * norm.ppf(1 - confidence))
    return var


if __name__ == '__main__':
    # Define the date range
    start_date = datetime.datetime(2014, 1, 1)
    end_date = datetime.datetime(2018, 1, 1)

    # Download stock data
    stock_data = download_data('C', start_date, end_date)

    # Convert to DataFrame if necessary and calculate daily returns
    stock_data = stock_data.to_frame(name='Price')  # Ensure it's a DataFrame
    stock_data['returns'] = np.log(stock_data['Price'] / stock_data['Price'].shift(1))
    stock_data = stock_data.dropna()  # Remove NaN values caused by shift
    print(stock_data.head())

    # Parameters for VaR calculation
    S = 1e6  # Investment amount
    confidence = 0.99  # Confidence level
    mu = np.mean(stock_data['returns'])  # Mean of daily returns
    sigma = np.std(stock_data['returns'])  # Standard deviation of daily returns

    # Calculate and display VaR
    var = calculate_var_n(S, confidence, mu, sigma, 1)
    print('Value at Risk (1 day, 99% confidence): ${:,.2f}'.format(var))
