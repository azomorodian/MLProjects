import matplotlib
matplotlib.use('TkAgg')


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

sns.set(style="whitegrid")
plt.rc("figure",autolayout=True)
plt.rc("axes",labelweight="bold",labelsize="large",titleweight="bold",titlesize=14,titlepad = 10)
df =  pd.read_csv("../input/fe-course-data/housing.csv")
print(df.columns)
X = df.loc[:,["MedInc","Latitude","Longitude"]]
print(X.head())
print(len(X))
print(X.columns)
print(type(X))
kmeans = KMeans(n_clusters=6,n_init=20)
X["Cluster"] = kmeans.fit_predict(X)
print(X.head())
X["Cluster"] = X["Cluster"].astype("category")
print(X.head())
print(type(X.Cluster))
sns.relplot(x="Longitude",y="Latitude",hue="Cluster",data=X,height=6)


X["MedHouseVal"] = df["MedHouseVal"]
sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6)
plt.show()