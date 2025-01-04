import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

# Constants
NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000

# Major biotech and healthcare companies
stocks = ['PFE', 'JNJ', 'MRNA', 'AMGN', 'BIIB', 'REGN', 'GILD', 'VRTX']

# Historical data range
start_date = '2021-01-01'
end_date = '2024-01-01'

def download_data():
    """Downloads historical stock data for the specified tickers."""
    stock_data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']
    return pd.DataFrame(stock_data).dropna()

def show_data(data):
    """Plots the historical stock prices."""
    data.plot(figsize=(10, 5))
    plt.title("Historical Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

def calculate_return(data):
    """Calculates logarithmic daily returns."""
    log_return = np.log(data / data.shift(1))
    return log_return[1:]

def generate_portfolios(returns):
    """Generates random portfolios and calculates their statistics."""
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []
    for _ in range(NUM_PORTFOLIOS):
        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)
        portfolio_weights.append(weights)
        portfolio_means.append(np.sum(returns.mean() * weights) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights))))
    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)

def statistics(weights, returns):
    """Calculates return, volatility, and Sharpe ratio for a portfolio."""
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])

def min_function_sharpe(weights, returns):
    """Objective function to minimize (negative Sharpe ratio)."""
    return -statistics(weights, returns)[2]

def optimize_portfolio(initial_weights, returns):
    """Finds the optimal portfolio weights by maximizing Sharpe ratio."""
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(stocks)))  # Allow full range [0, 1] for weights
    return optimization.minimize(fun=min_function_sharpe, x0=initial_weights, args=returns,
                                 method='SLSQP', bounds=bounds, constraints=constraints)

def print_optimal_portfolio(optimum, returns):
    """Prints the optimal portfolio weights and corresponding statistics."""
    print("Optimal Portfolio Allocation:")
    for stock, weight in zip(stocks, optimum['x'].round(3)):
        print(f"{stock}: {weight:.3f}")
    stats = statistics(optimum['x'], returns)
    print("\nPortfolio Statistics:")
    print(f"Expected Return: {stats[0]:.3f}")
    print(f"Expected Volatility: {stats[1]:.3f}")
    print(f"Sharpe Ratio: {stats[2]:.3f}")

def show_optimal_portfolio(opt, returns, portfolio_means, portfolio_risks):
    """Visualizes the portfolios and highlights the optimal portfolio."""
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_risks, portfolio_means, c=portfolio_means / portfolio_risks, marker='o')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.title('Portfolio Optimization')
    plt.grid(True)
    optimal_stats = statistics(opt['x'], returns)
    plt.plot(optimal_stats[1], optimal_stats[0], 'g*', markersize=20.0, label='Optimal Portfolio')
    plt.legend()
    plt.show()

# Include the missing show_portfolios function
def show_portfolios(returns, volatilities):
    """Visualizes portfolio returns and risks on a scatterplot."""
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.title('Portfolio Risk vs Return')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Step 1: Download and show raw data
    dataset = download_data()
    show_data(dataset)

    # Step 2: Calculate log daily returns
    log_daily_returns = calculate_return(dataset)

    # Step 3: Generate random portfolios
    portfolio_weights, portfolio_means, portfolio_risks = generate_portfolios(log_daily_returns)

    # Step 4: Visualize the generated portfolios
    show_portfolios(portfolio_means, portfolio_risks)

    # Step 5: Optimize portfolio
    initial_weights = portfolio_weights[0]
    optimum = optimize_portfolio(initial_weights, log_daily_returns)

    # Step 6: Print optimal portfolio and statistics
    print_optimal_portfolio(optimum, log_daily_returns)

    # Step 7: Highlight the optimal portfolio on the scatterplot
    show_optimal_portfolio(optimum, log_daily_returns, portfolio_means, portfolio_risks)
