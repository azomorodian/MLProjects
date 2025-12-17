import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from category_encoders import MEstimateEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

sns.set_style('whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', titlesize=14, labelsize='large', titleweight='bold', labelweight='bold',titlepad=10)
warnings.filterwarnings('ignore')
def score_dataset(X, y , model = XGBRegressor()):
    for colname in X.select_dtypes(include=['object','category']):
        X[colname], _ = X[colname].factorize()
    score = cross_val_score(model, X, y, scoring='neg_mean_squared_log_error')
    score = -1*score.mean()
    score = np.sqrt(score)
    return score

df = pd.read_csv('../input/fe-course-data/ames.csv')
print(df.select_dtypes(["object"]).nunique())
print(df["SaleType"].value_counts())
print(df["SaleType"].value_counts().sum())
print(df.shape)
print(df.Neighborhood.value_counts())
X_encoder = df.sample(frac=0.20,random_state=0)
y_encoder = X_encoder.pop("SalePrice")
X_pretrain = df.drop(X_encoder.index)
y_train = X_pretrain.pop("SalePrice")
encoder = MEstimateEncoder(m=5.0,cols=['Neighborhood'])
encoder.fit(X_encoder,y_encoder)
X_train = encoder.transform(X_pretrain,y_train)
feature = encoder.cols

plt.figure(dpi=90)
ax = sns.distplot(y_train, kde=True, hist=False)
ax = sns.distplot(X_train[feature], color='r', ax=ax, hist=True, kde=False, norm_hist=True)
ax.set_xlabel("SalePrice")
plt.show()

X = df.copy()
y = X.pop("SalePrice")
score_base = score_dataset(X, y)
score_new = score_dataset(X_train, y_train)

print(f"Baseline Score: {score_base:.4f} RMSLE")
print(f"Score with Encoding: {score_new:.4f} RMSLE")

m = 0
X = df.copy()
y = X.pop('SalePrice')
X["Count"] = range(len(X))
X["Count"][1] = 0
encoder = MEstimateEncoder(cols="Count", m=m)
X = encoder.fit_transform(X, y)
score =  score_dataset(X, y)
print(f"Score: {score:.4f} RMSLE")



