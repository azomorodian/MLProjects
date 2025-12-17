import matplotlib
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler

matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.rc("figure", autolayout=True)
plt.rc("axes", titlesize=18, labelsize='large', titleweight='bold',labelweight='bold',titlepad = 10)
plt.rc("animation",html="html5")
from IPython.display import display
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer,make_column_selector
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

fuel = pd.read_csv('../input/dl-course-data/fuel.csv')
X = fuel.copy()
print(X.head())
print(X.shape)
print(X.columns)
print(X.describe())
y = X.pop('FE')
print(y)
preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse_output=False),
     make_column_selector(dtype_include=object)),
)
pd.set_option('display.max_columns', None)
print(X)
X = preprocessor.fit_transform(X)
print(X.shape)
y = np.log(y)
input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))

# Uncomment to see original data
print(fuel.head())
# Uncomment to see processed features
print(pd.DataFrame(X[:10,:]).head())
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(128,activation='relu',input_shape = input_shape ),
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam',loss='mae')
history = model.fit(X,y,batch_size=128,epochs=200)
history_df = pd.DataFrame(history.history)
print(history_df)
plt.plot(history.history['loss'])
plt.show()
learning_rate = 0.05
batch_size = 32
num_examples = 256

