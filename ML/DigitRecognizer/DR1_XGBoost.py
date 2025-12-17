from cmath import sqrt
from sklearn.model_selection import train_test_split

import pandas as pd
from matplotlib import pyplot as plt
from math import sqrt,floor

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

df_train = pd.read_csv("../input/XPC-Team-Digit-Recognizer/train.csv")
df_test = pd.read_csv("../input/XPC-Team-Digit-Recognizer/test.csv")
print(df_train.head())
print(df_test.head())


X =df_train.copy()
y = X.pop("label")
X_Test = df_test.copy()
print(y.head())
print(X.head())
Data = X.to_numpy()
print(Data.shape)
Data = Data.reshape(-1,28,28)
print(Data.shape)
def imgShow(imdData):
    plt.figure(figsize=(20,20))
    if imdData.ndim==2:
        plt.imshow(imdData, cmap='gray',vmin=0,vmax=1)
        plt.axis('off')
        plt.show()
    else:
        num,dx,dy = imdData.shape
        nx = int(min(floor(sqrt(num)),10))
        num1 = int(min(nx*nx,num))
        for i in range(num1):
            plt.subplot(nx,nx,i+1)
            plt.imshow(imdData[i], cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
        plt.show()
#imgShow(Data)
model = XGBClassifier(booster='gbtree',device = 'cpu',verbosity=2,eta=0.3,gamma=0,max_depth = 6)
df_X = df_train.copy()
df_y = df_X.pop("label")
score = cross_val_score(model,df_X,df_y,cv=5,scoring='accuracy')
print(score)
model.fit(X,y)
y_pred = model.predict(X_Test)
print(pd.DataFrame({'ImageId':range(1,28001),'Label':y_pred}))
submission = pd.DataFrame({'ImageId':range(1,28001),'Label':y_pred})
submission.to_csv('submission.csv',index=False)