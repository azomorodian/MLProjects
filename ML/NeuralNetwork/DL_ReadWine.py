import matplotlib
matplotlib.use('tkAgg')
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
red_wine = pd.read_csv('../input/dl-course-data/red-wine.csv')
print(red_wine.head())
print(red_wine.shape)
df_train = red_wine.sample(frac= 0.7,random_state= 0)
print(df_train.head())
print(df_train.shape)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(10))

max_ = df_train.max(axis = 0)
min_ = df_train.min(axis = 0)

df_train = (df_train - min_)/(max_ - min_)
df_valid = (df_valid - min_)/(max_-min_)
X_train = df_train.drop('quality',axis = 1)
X_valid = df_valid.drop('quality',axis = 1)
y_train = df_train['quality']
y_valid = df_valid['quality']
print(X_train.shape)
print([X_train.shape[1]])
model = keras.Sequential([
    layers.Dense(units =512,activation='relu',input_shape=[X_train.shape[1]]),
    layers.Dense(units=512,activation='relu'),
    layers.Dense(units=512,activation='relu'),
    layers.Dense(units=1)
])
model.compile(optimizer='adam',loss='mae')
history = model.fit(X_train,
                    y_train,
                    validation_data=(X_valid,y_valid),
                    batch_size=256,
                    epochs=100)

history_df = pd.DataFrame(history.history)
print(history_df)
history_df['loss'].plot();
plt.show()