import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

sns.set_style("whitegrid")
plt.rc("figure",autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad = 10
)

def plot_variance(pca,width=8,dpi=100):
    fig, axs = plt.subplots(1,2)
    n = pca.n_components_
    grid = np.arange(1,n+1)
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid,evr)
    axs[0].set(xlabel='Component',title = "% Explained Variance",ylim = (0.0,1.0))

    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0,grid],np.r_[0,cv],'o-')
    axs[1].set(xlabel='Component',title = "% Cumulative Variance",ylim = (0.0,1.0))
    fig.set(figwidth=8, dpi=100)
    return axs

def make_mi_scores(X,y,discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
df = pd.read_csv('../input/fe-course-data/autos.csv')
features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]
X = df.copy()
y = X.pop('price')
X = X.loc[:, features]
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)
print(X_pca)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=component_names,
    index = X.columns
)
print(loadings)
print(pca)

mi_scores = make_mi_scores(X_pca, y, discrete_features=False)
print(mi_scores)


idx = X_pca["PC3"].sort_values(ascending=False).index
cols = ["make", "body_style", "horsepower", "curb_weight"]
print(df.loc[idx, cols])

plot_variance(pca)
plt.show()

df["sports_or_wagon"] = X.curb_weight / X.horsepower
sns.regplot(x="sports_or_wagon", y='price', data=df, order=2);

plt.show()



