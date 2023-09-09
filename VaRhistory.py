'''
Author: Khandoker Tanjim Ahammad
Date: 09.09.2023
Purpose: Value at risk calculation based on history data.
generate requirement file: pipreqs --encoding=utf8 ./
'''
import streamlit as st
import yfinance as yf
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm
import re

# Streamlit app layout
def VaRwithoutmc():
    st.title("Value at Risk calculation app-v0.1")
    st.markdown('###### This is a simple app to calculate value at risk  for a number of companies Based on historical data )
    st.markdown("###### Tips: Use at least more that one company to create portfolio otherwise this code will not work")
    st.markdown("###### Tips: We have 3 inputs to run this app. Frist we need at least two companies  stock symbol(stock symbol can be found in yahoo finance website)")
    st.markdown("###### Tips: Next we need the desired weigts to calculates portfolio.")
    st.markdown("###### Example value: AAPL,META,C,AMZN")
    c=st.text_input("Enter Stock symbol:")
    st.markdown("###### Tips: Make sure the number of company stock symbol and number of weight should be same.Total number of weight should be equal to one")
    st.markdown("###### Example value: 0.1,0.4,0.1,0.4")
    n=st.text_input("Enter desired weights:")
    d=st.number_input("Number of days to calculate VaR:", step=1, format="%d")
    if st.button("Calculate"):
        companies = []
        weights=[]
        input_list = [s.strip() for s in re.split(r'[,\s]+', c)]
        companies.extend(input_list)
        numbers = [float(x.strip()) for x in n.split(",")]
        weights.extend(numbers)
        weights=np.array(weights)
        print('weight')
        start=dt.datetime(2020,1,1)
        end=dt.datetime.now()
        df=yf.download(companies,start,end)['Adj Close']
        #print(df)
        # daily return of adjusted close 
        daily_return_adj_close=df.pct_change()
        #print(daily_return_adj_close)
        cov_mat=daily_return_adj_close.cov()
        print(cov_mat)
        avg_returns=daily_return_adj_close.mean()
        print('avg_returns')
        print(avg_returns)
        portfolio_mean=np.matmul(avg_returns,weights)
        print('weight')
        print(weights)
        print(portfolio_mean)
        portfolio_std=np.sqrt(weights.T@cov_mat@weights)
        #portfolio_std=np.sqrt(np.matmul(weights.transpose(),cov_mat,weights))
        print(f'protfolio mean: {portfolio_mean} and portfolio standard deviation: {portfolio_std}') 
        confidence_level=0.05
        VaR=norm.ppf(confidence_level,portfolio_mean,portfolio_std)
        print(f'if we have cofidence level:{confidence_level} and then we have the posility to lose the amount of money in one days is VaR')
        days=[]
        days.append(d)
        days_VaR=VaR*np.sqrt(days)
        st.write(f'if we have cofidence level: {confidence_level} and then we have the posility to lose the amount of money in : {days} days is: {days_VaR}')
        st.write(f'The value at risk in last {days} days is {days_VaR}')

if __name__ == "__main__":
          VaRwithoutmc()
