'''
Author: Khandoker Tanjim Ahammad
Date: 07/09/2023
Purpose: use monte carlo simulation over stock price.

What is Daily Adjusted closing price?
Its the closing price where dividend and stock splis were taken into consideration to
get more accurate price value.
Dividend stocks are companies that pay out a portion of their profits to shareholders.
A stock split is when a company divides and increases the number of shares available 
to buy and sell on an exchange. 

what is daily log return?
For example, if a stock is priced at 3.570 USD per share at the close on one day, 
and at 3.575 USD per share at the close the next day, then the logarithmic return
is: ln(3.575/3.570) = 0.0014, or 0.14%.

what is portfolio?
A portfolio is a collection of financial investments like stocks, bonds, commodities,
cash, and cash equivalents, including closed-end funds and exchange traded funds (ETFs).
People generally believe that stocks, bonds, and cash comprise the core of a portfolio.

what is portfolio expected return?
An expected return of a portfolio is the weighted average rate of return for all the assets
in the portfolio. The weights represent the proportion invested in each asset in the entire 
investment portfolio and can be found by simply multiplying each asset's rate of return with
its corresponding percentage.

A portfolio's return on investment (ROI) can be calculated as follows:
(Closing value - Starting balance )/ Starting Value.

what is covariance matrix in stocks?
In the context of investment-related risk calculations, a variance-covariance
matrix is a rectangular matrix that contains the variances and covariances of 
the stocks in a portfolio. 
'''
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

### Set time from to a certain number of years
years = 15

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days = 365*years)

### Create a list of company to pull stock price 
companies = ['SPY','BND','GLD','QQQ','VTI']

### Download the daily adjusted close prices for the tickers
adj_close_price_df = pd.DataFrame()

for company in companies:
    data = yf.download(company, start = startDate, end = endDate)
    adj_close_price_df[company] = data['Adj Close']

print(adj_close_price_df)

### Calculate the daily log returns and drop any NAs
### Shift one is shifting to the previous value. 
### adjusted price daily return in log scale.
log_returns = np.log(adj_close_price_df/adj_close_price_df.shift(1))
### dorping Na value 
log_returns  = log_returns.dropna()

# print(log_returns)

### Create a function that will be used to calculate portfolio expected return
#*We are assuming that future returns are based on past returns, which is not a 
# reliable assumption.
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights)

### Create a function that will be used to calculate portfolio standard deviation
def standard_deviation (weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

### Create a covariance matrix for all the securities
cov_matrix = log_returns.cov()
print(cov_matrix)

### Create an equally weighted portfolio and find total portfolio expected return and standard deviation
portfolio_value = 1000000
weights = np.array([1/len(companies)]*len(companies))
portfolio_expected_return = expected_return(weights, log_returns)
portfolio_std_dev = standard_deviation (weights, cov_matrix)


def random_z_score():
    return np.random.normal(0, 1)

### Create a function to calculate scenarioGainLoss
days = 20

def scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days):
    return portfolio_value * portfolio_expected_return * days + portfolio_value * portfolio_std_dev * z_score * np.sqrt(days)

### Run 10000 simulations
simulations = 10000
scenarioReturn = []

for i in range(simulations):
    z_score = random_z_score()
    scenarioReturn.append(scenario_gain_loss(portfolio_value, portfolio_std_dev, z_score, days))

### Specify a confidence interval and calculate the Value at Risk (VaR)
confidence_interval = 0.99
VaR = -np.percentile(scenarioReturn, 100 * (1 - confidence_interval))
print(VaR)

### Plot the results of all 10000 scenarios
plt.hist(scenarioReturn, bins=50, density=True)
plt.xlabel('Scenario Gain/Loss ($)')
plt.ylabel('Frequency')
plt.title(f'Distribution of Portfolio Gain/Loss Over {days} Days')
plt.axvline(-VaR, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {confidence_interval:.0%} confidence level')
plt.legend()
plt.show()