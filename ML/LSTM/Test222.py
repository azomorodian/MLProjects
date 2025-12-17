import numpy as np
import pandas as pd
import yfinance as yf
import datetime
ticker = 'AAPL'
#Download Historical Data from yahoo finance
data = yf.download(ticker,start= '2020-01-01',end=datetime.datetime.today().strftime('%Y-%m-%d'))
data = data[['Close']]

forecast_dates = pd.date_range(start=data.index[-1]+pd.Timedelta(days=1), periods=30,freq='D')
print(forecast_dates)
a = np.linspace(num=30,start=3,stop=90)
for _ in range(30):
    print(a[_])