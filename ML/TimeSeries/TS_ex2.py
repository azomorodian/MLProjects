import matplotlib
matplotlib.use('tkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rc('figure',autolayout=True,figsize=(11,4),titlesize=18,titleweight='bold')
plt.rc('axes',labelsize='large',titleweight='bold',labelweight='bold',titlesize=16,titlepad = 10)
#%config InlineBackend.figure_format = 'retina'

df = pd.read_csv("../input/ts-course-data/book_sales.csv",index_col = 'Date',parse_dates=['Date']).drop('Paperback',axis=1)
print(df.head())
df['Time'] = np.arange(len(df.index))

fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=df, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales')
plt.show()
df['Lag_1'] = df['Hardcover'].shift(1)
df = df.reindex(columns=['Hardcover', 'Lag_1'])
print(df.head())
fig, ax = plt.subplots()
ax = sns.regplot(x='Lag_1', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_aspect('equal')
ax.set_title('Lag Plot of Hardcover Sales')
plt.show()
