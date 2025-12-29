import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

from pathlib import Path
import matplotlib.pyplot as plt

from ML.Kaggle.learntools.time_series.utils import seasonal_plot
from scipy.signal import periodogram
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression



def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    ts = ts.dropna()
    fs = 365.25
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )

    if ax is None:
        _, ax = plt.subplots()

    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax



plt.rc('figure',autolayout=True)
plt.rc('axes', titlesize=14,labelsize='large',titleweight='bold',labelweight='bold',titlepad=10)



data_dir = Path('../input/ts-course-data')
tunnel = pd.read_csv(data_dir / "tunnel.csv",parse_dates=["Day"])
tunnel = tunnel.set_index("Day")

X = tunnel.copy()
print(X.head())
print(X.index)
print()
X["day"] = X.index.dayofweek
X["week"] = X.index.isocalendar().week

X['dayofyear'] = X.index.dayofyear
X['year'] = X.index.isocalendar().year

print(X.head())

fig , (ax0,ax1) = plt.subplots(2,1,figsize=(11,6))
seasonal_plot(X,y="NumVehicles",period="week",freq="day",ax=ax0)
seasonal_plot(X,y="NumVehicles",period="year",freq="dayofyear",ax=ax1)

#plt.show()
#plot_periodogram(tunnel.NumVehicles)
#plt.show()
tunnel.index = tunnel.index.to_period('D')
fourier = CalendarFourier(freq="YE",order = 10)

dp = DeterministicProcess(
     index= tunnel.index,
     constant=True,
     order= 1,
     seasonal=True,
     additional_terms= [fourier],
     drop= True,
 )
X = dp.in_sample()
y = tunnel["NumVehicles"]
model = LinearRegression(fit_intercept=False)
_ = model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=y.index)
X_fore = dp.out_of_sample(steps=90)
y_fore = pd.Series(model.predict(X_fore),index=X_fore.index)

ax = y.plot(color = '0.25', style= '.' ,title = "Tunnel Traffic _ Seasonal Forcast")
ax = y_pred.plot(ax = ax, label = "Seasonal")
ax = y_fore.plot(ax = ax, label = "Seasonal Forecast",color = 'C3')
_ = ax.legend()
plt.show()
