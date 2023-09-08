'''
Author: Khandoker Tanjim Ahammad
Date 08.09.2023
Purpose: stock portfolio caluculation
'''
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt



years = 15

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days = 365*years)

adj_close_price_df = pd.DataFrame()
companylist=['CBA','BHP','TLS.AX','NAB.AX','WBC.AX','STO.AX']

for company in companylist:
    data = yf.download(company, start = startDate, end = endDate)
    adj_close_price_df[company] = data['Adj Close']

print(adj_close_price_df)
log_returns = np.log(adj_close_price_df/adj_close_price_df.shift(1))
meanlogreturn=log_returns.mean()
print(meanlogreturn)
covMatrix = log_returns.cov()
print(covMatrix)


# weight for porfolio randomly
weights=np.random.random(len(meanlogreturn))
weights /= np.sum(weights)

print(weights)


# # Monte Carlo Method
mc_sims = 10000 # number of simulations
T = 100 #timeframe in days
meanM = np.full(shape=(T, len(weights)), fill_value=meanlogreturn)
meanM = meanM.T
portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
initialPortfolio = 10000
for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))#uncorrelated RV's
    L = np.linalg.cholesky(covMatrix) #Cholesky decomposition to Lower Triangular Matrix
    dailyReturns = meanM + np.inner(L, Z) #Correlated daily returns for individual stocks
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio
plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
plt.show()


def mcVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: percentile on return distribution to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")
def mcCVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: CVaR or Expected Shortfall to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a pandas data series.")
portResults = pd.Series(portfolio_sims[-1,:])
VaR = initialPortfolio - mcVaR(portResults, alpha=5)
CVaR = initialPortfolio - mcCVaR(portResults, alpha=5)
print('VaR ${}'.format(round(VaR,2)))
print('CVaR ${}'.format(round(CVaR,2)))