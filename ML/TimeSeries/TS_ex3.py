from pathlib import Path
from warnings import simplefilter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib_inline.config import InlineBackend

matplotlib.use('tkAgg')


simplefilter("ignore")
sns.set_style("whitegrid")

plt.rc('figure',autolayout=True,figsize=(11,4))
plt.rc('axes', titlesize=14,labelsize='large',titleweight='bold',labelweight='bold',titlepad=10)
plot_params = dict(
    color = '0.75',
    style = '.-',
    markeredgecolor = '0.25',
    markerfacecolor = '0.25',
    legend = False
)

data_dir = Path('../input/ts-course-data')
tunnel = pd.read_csv(data_dir / "tunnel.csv",parse_dates=["Day"])
tunnel = tunnel.set_index("Day")
tunnel = tunnel.to_period()
print(tunnel.head())
print(tunnel.index.dtype)
df = tunnel.copy()
df['Time'] = np.arange(len(tunnel.index))
print(df.head())
df['Lag_1'] = df['NumVehicles'].shift(1)
print(df.head())
from sklearn.linear_model import LinearRegression
X = df.loc[:, ['Lag_1']]
print(X)
X.dropna(inplace=True) # drop missing values in the feature set
print(X.head())
y = df.loc[:, 'NumVehicles']
y, X = y.align(X, join='inner') # drop corresponding values in target
print(y.head())
model = LinearRegression()
model.fit(X, y)
y_pred = pd.Series(model.predict(X),index=X.index)
print(y_pred)
fig, ax = plt.subplots()
ax.plot(X['Lag_1'], y, '.', color='0.25')
ax.plot(X['Lag_1'], y_pred)
ax.set_aspect('equal')
ax.set_ylabel('NumVehicles')
ax.set_xlabel('Lag_1')
ax.set_title('Lag Plot of Tunnel Traffic');
plt.show()
ax = y.plot(**plot_params)
ax = y_pred.plot()
plt.show()