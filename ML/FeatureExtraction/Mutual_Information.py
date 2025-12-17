import matplotlib
# تغییر بک‌اند به یک بک‌اند غیرتعاملی یا تعاملی استاندارد
# 'Agg' غیرتعاملی است (فقط برای ذخیره نمودار)
# 'TkAgg' یا 'Qt5Agg' تعاملی هستند
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

sns.set_style("whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
     "axes",
     labelweight="bold",
     labelsize="large",
     titleweight="bold",
     titlesize=14,
     titlepad=10,
 )
df = pd.read_csv('../input/fe-course-data/ames.csv')


def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
def plot_mi_scores(scores):
     scores = scores.sort_values(ascending=True)
     width = np.arange(len(scores))
     ticks = list(scores.index)
     plt.barh(width,scores)
     plt.yticks(width, ticks)
     plt.title("Mutual Informastion Scores")



features = ["YearBuilt", "MoSold", "ScreenPorch","OverallQual"]
print(df[features].head())
print(df.columns)
sns.relplot(
    x="value", y="SalePrice", col="variable", data=df.melt(id_vars="SalePrice", value_vars=features), facet_kws=dict(sharex=False),
);
X = df.copy()
y = X.pop("SalePrice")
mi_scores = make_mi_scores(X,y)
print(mi_scores)
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores.head(20))

sns.catplot(x="BldgType", y="SalePrice", data=df, kind="boxen")

plt.show()

