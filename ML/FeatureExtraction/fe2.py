import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
def score_dataset(X,y,model = XGBRegressor()):
    for colname in X.select_dtypes(["category","object"]):
        X[colname], _ = X[colname].factorize()
    score = cross_val_score(model,X,y,cv=5,scoring='neg_mean_squared_log_error')
    print(score)
    score = -1*score.mean()
    score = np.sqrt(score)
    return score
df = pd.read_csv("../input/fe-course-data/ames.csv")
X = df.copy()
y = X.pop("SalePrice")
X_1 = pd.DataFrame()
X_1["LivLotRatio"] = (df.GrLivArea / df.LotArea)
X_1["Spaciousness"] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
X_1["TotalOutsideSF"] = df.WoodDeckSF + df.OpenPorchSF+df.EnclosedPorch+df.Threeseasonporch+df.ScreenPorch
#print(X_1.head(10))
# One-hot encode Categorical feature, adding a column prefix "Cat"
X_new = pd.get_dummies(X['HeatingQC'], prefix="Cat")
print(X_new.head(10))
pd.set_option('display.max_columns', None)
print(X.head(10))


# Multiply row-by-row
#X_new = X_new.mul(df.Continuous, axis=0)
# Join the new features to the feature set
#X = X.join(X_new)
X_3 = pd.DataFrame()

# YOUR CODE HERE
X_3["PorchTypes"] = df[[
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "Threeseasonporch",
    "ScreenPorch",
]].gt(0.0).sum(axis=1)

print(df.MSSubClass.unique())
X_4 = pd.DataFrame()
X_4["MSClass"] = df.MSSubClass.str.split('_',n=1,expand = True)[0]
print(X_4)

X_5 = pd.DataFrame()
X_5["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")
print(X_5)
X_2 = pd.get_dummies(X['BldgType'],prefix="Bldg")
# Multiply
X_2 = X_2.mul(X.GrLivArea,axis=0)

X_new = X.join([X_1, X_2, X_3, X_4, X_5])
print(score_dataset(X_new, y))
print(score_dataset(X, y))