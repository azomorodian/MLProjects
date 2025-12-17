import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
sns.set(style="whitegrid")
plt.rc("figure",autolayout=True)
plt.rc("axes",
       labelweight="bold",
       titleweight="bold",
       labelsize="large",
       titlesize=14,
       labelpad=10)
def score_dataset(X,y,model = XGBClassifier()):
    for colname in X.select_dtypes(['category','object']):
        X[colname], _ = X[colname].factorize()

    score = cross_val_score(model, X, y,cv=5 ,scoring='neg_mean_squared_log_error')
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score
df = pd.read_csv("../input/fe-course-data/ames.csv")
#print(score_dataset(X,y))
X = df.copy()
y = X.pop('SalePrice')
features = ['LotArea', 'TotalBsmtSF', 'FirstFlrSF', 'SecondFlrSF','GrLivArea']

X_scaled = X.loc[:, features]
X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
kmeans = KMeans(n_clusters=10,n_init= 10 , random_state=0)
X["Cluster"] = kmeans.fit_predict(X_scaled)


Xy = X.copy()
Xy["Cluster"] = Xy.Cluster.astype("category")
Xy["SalePrice"] = y

sns.relplot(
    x="value", y="SalePrice", hue="Cluster", col="variable",
    height=4, aspect=1, facet_kws={'sharex': False}, col_wrap=3,
    data=Xy.melt(
        value_vars=features, id_vars=["SalePrice", "Cluster"],
    ),
)
plt.show()
score_dataset(X, y)
