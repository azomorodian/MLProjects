import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('tkAgg')
from pathlib import Path
from sklearn.linear_model import LinearRegression

data_dir = Path('../input/ts-course-data/')
comp_dir = Path('../input/store-sales-time-series-forecasting')
book_sales = pd.read_csv(data_dir / 'book_sales.csv',
                         index_col= 'Date',
                         parse_dates= ['Date']
                         ).drop('Paperback',axis=1)
book_sales['Time'] = np.arange(len(book_sales.index))
book_sales['Lag_1'] = book_sales['Hardcover'].shift(1)
book_sales = book_sales.reindex(columns=['Hardcover', 'Time', 'Lag_1'])
ar = pd.read_csv(data_dir / 'ar.csv')

dtype = {
    'store_nbr': 'category',
    'family' : 'category',
    'sales' : 'float32',
    'onpromotion':'uint64'
}
store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    dtype=dtype,
    parse_dates= ['date'],
    infer_datetime_format=True
    )
store_sales = store_sales.set_index('date').to_period('D')
store_sales = store_sales.set_index(['store_nbr','family'], append=True)
average_sales = store_sales.groupby('date').mean()['sales']
fig, ax = plt.subplots()

ax.plot('Time', 'Hardcover', data=book_sales, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=book_sales, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales');
plt.show()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
ax1.plot(ar['ar1'])
ax1.set_title('Series 1')
ax2.plot(ar['ar2'])
ax2.set_title('Series 2');
plt.show()
df = average_sales.to_frame()
print(df.head())
time = np.arange(len(df.index))
print(time)
from sklearn.linear_model import LinearRegression
df['time'] = time

print(df['time'])

# YOUR CODE HERE: Create training data
X = df.loc[:,['time']]  # features
y = df.loc[:,'sales']  # target
print(X)
print(y)
df = average_sales.to_frame()
lag_1 = df['sales'].shift(1)
df['lag_1'] = lag_1  # add to dataframe

X = df.loc[:, ['lag_1']].dropna()  # features
y = df.loc[:, 'sales']  # target
y, X = y.align(X, join='inner')  # drop corresponding values in target

# YOUR CODE HERE: Create a LinearRegression instance and fit it to X and y.
model = LinearRegression()

# YOUR CODE HERE: Create Store the fitted values as a time series with
# the same time index as the training data
model.fit(X,y)
y_pred = pd.Series(model.predict(X), index=X.index)
fig, ax = plt.subplots()
ax.plot(X['lag_1'], y, '.', color='0.25')
ax.plot(X['lag_1'], y_pred)
ax.set(aspect='equal', ylabel='sales', xlabel='lag_1', title='Lag Plot of Average Sales')
plt.show()