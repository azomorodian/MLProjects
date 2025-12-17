import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from category_encoders import MEstimateEncoder


sns.set_style('whitegrid')
plt.rc('figure',autolayout=True)
plt.rc('axes', titlesize=14, labelsize="large", titleweight="bold",labelweight="bold",titlepad = 10)
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/fe-course-data/movielens1m.csv')
#print(df.head())
df = df.astype(np.uint8,errors='ignore')
print(df.shape)
print("Number of Unique Zipcodes:{}".format(df['Zipcode'].nunique()))
X = df.copy()
y = X.pop('Rating')
X_encode = X.sample(frac=0.25)
y_encode = y[X_encode.index]
X_pretrain = X.drop(X_encode.index)
y_train = y[X_pretrain.index]
print(X_encode.head(),'\n',X.head())
encoder = MEstimateEncoder(cols=['Zipcode'],m=5.0)
encoder.fit(X_encode,y_encode)
X_train = encoder.transform(X_pretrain)
print(X_train.shape)
print(X_train['Zipcode'].head())
print(X_pretrain['Zipcode'].head())
plt.figure(dpi=90)
ax = sns.distplot(y, kde=False, norm_hist=True)
ax = sns.kdeplot(X_train.Zipcode, color='r', ax=ax)
ax.set_xlabel("Rating")
ax.legend(labels=['Zipcode', 'Rating'])
plt.show()


