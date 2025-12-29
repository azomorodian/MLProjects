from pathlib import Path
from warnings import simplefilter

import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

simplefilter("ignore")
sns.set_style("whitegrid")
plt.rc('figure',autolayout=True,figsize=(11,5))
plt.rc('axes', titlesize=14, labelsize='large', titleweight='bold', labelweight='bold', titlepad= 10)
plot_param = dict(
    color = '0.75',
    style = '.-',
    markeredgecolor = '0.25',
    markerfacecolor = '0.25',
    legend = False
)

data_dir = Path('../input/ts-course-data')
tunnel = pd.read_csv(data_dir / 'tunnel.csv',parse_dates=['Day'])
tunnel = tunnel.set_index('Day').to_period()

moving_avg = tunnel.rolling(
    window=365,
    center=True,
    min_periods = 183
).mean()
ax = tunnel.plot(style=".",color = "0.8")
moving_avg.plot(ax=ax,linewidth = 3,title = "Tunnel Traffic - 365-Day Moving Average",legend = False)
plt.show()

from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
    index=tunnel.index,
    constant=True,
    order=1,
    drop= True
)

X = dp.in_sample()
print(X.head())

from sklearn.linear_model import LinearRegression

y =  tunnel['NumVehicles']

model = LinearRegression(fit_intercept=False)
model.fit(X,y)

y_pred = pd.Series(model.predict(X),index= X.index)

ax = tunnel.plot(style=".",color = "0.5",title = "Tunnel Traffic - Linear Trend")
_ = y_pred.plot(ax=ax,linewidth = 3 , label = "Trend")
plt.show()

X = dp.out_of_sample(steps=30)

y_fore = pd.Series(model.predict(X),index= X.index)

print(y_fore.head())

ax = tunnel["2005-5":].plot(title= "Tunnel Traffic - Linear Trend Forcast", **plot_param)
ax = y_pred["2005-5":].plot(ax = ax , linewidth = 3 , label = "Trend")
ax = y_fore.plot(ax = ax , linewidth = 3 , label = "Trend Forecase", color = 'C3' )
_ = ax.legend()
plt.show()


