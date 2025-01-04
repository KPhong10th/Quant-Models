import numpy as np
import yfinance as yf
import datetime
import pandas as pd


def download_data(stock, start, end):
    # Download stock data
    ticker = yf.download(stock, start, end)
    print("Data Columns:", ticker.columns)  # Inspect the data columns

    # Check if it's a MultiIndex and handle accordingly
    if isinstance(ticker.columns, pd.MultiIndex):
        # Access the 'Adj Close' column if it exists
        if ('Adj Close', stock) in ticker.columns:
            return ticker[('Adj Close', stock)]
        elif ('Close', stock) in ticker.columns:
            return ticker[('Close', stock)]
        else:
            raise KeyError(f"Neither 'Adj Close' nor 'Close' found for {stock} in MultiIndex data.")
    else:
        # For single-level columns
        if 'Adj Close' in ticker.columns:
            return ticker['Adj Close']
        elif 'Close' in ticker.columns:
            return ticker['Close']
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' found in the downloaded data.")


class ValueAtRiskMonteCarlo:
    def __init__(self, S, mu, sigma, c, n, iterations):
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n
        self.iterations = iterations

    def simulation(self):
        # Generate random normal values
        rand = np.random.normal(0, 1, [1, self.iterations])

        # Simulate stock prices using the Geometric Brownian Motion model
        stock_price = self.S * np.exp(self.n * (self.mu - 0.5 * self.sigma ** 2) + self.sigma * np.sqrt(self.n) * rand)

        # Sort the simulated prices and determine the percentile
        stock_price = np.sort(stock_price)
        percentile = np.percentile(stock_price, (1 - self.c) * 100)

        # VaR is the difference between initial investment and the percentile value
        return self.S - percentile


if __name__ == "__main__":
    S = 1e6  # Investment amount in USD
    c = 0.95  # Confidence level (95%)
    n = 1  # Time horizon: 1 day
    iterations = 100000  # Number of simulation paths

    # Historical data range
    start_date = datetime.datetime(2014, 1, 1)
    end_date = datetime.datetime(2017, 10, 15)

    # Download stock data
    citi = download_data('C', start_date, end_date)

    # Calculate daily returns
    citi = citi.dropna()  # Drop any NaN values
    citi['returns'] = citi.pct_change()

    # Estimate mean and standard deviation of daily returns
    mu = np.mean(citi['returns'])
    sigma = np.std(citi['returns'])

    # Instantiate the Monte Carlo model and calculate VaR
    model = ValueAtRiskMonteCarlo(S, mu, sigma, c, n, iterations)
    var = model.simulation()

    print('Value at Risk with Monte Carlo simulation: ${:,.2f}'.format(var))
