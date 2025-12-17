import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers , callbacks
from IPython.display import display

red_wine = pd.read_csv('../input/dl-course-data/red-wine.csv')
df_train = red_wine.sample(frac=0.7,random_state=0)
df_valid = red_wine.drop(df_train.index)
pd.set_option('display.max_columns', None)
display(df_train.head())

#SCALE [0,1]
max_ = df_train.max(axis = 0)
min_ = df_train.min(axis = 0)
df_train = (df_train - min_)/(max_-min_)
df_valid = (df_valid - min_)/(max_-min_)
X_train = df_train.drop('quality',axis = 1)
X_valid = df_valid.drop('quality',axis = 1)
y_train = df_train['quality']
y_valid = df_valid['quality']
input_shape = [X_train.shape[1]]
print(input_shape)
early_stopping = callbacks.EarlyStopping(
    min_delta = 0.001,
    patience = 20,
    verbose = 0,
    restore_best_weights = True
)
model = keras.Sequential([
    layers.Dense(units=512,activation='relu',input_shape = input_shape),
    layers.Dense(units=512,activation='relu'),
    layers.Dense(units=512,activation='relu'),
    layers.Dense(units=1)
])
model.compile(
    optimizer='adam',
    loss='mae'
)
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid,y_valid),
    epochs=500,
    batch_size=256,
    callbacks=[early_stopping],
    verbose = 0
)
history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss', 'val_loss']].plot()
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
plt.show()
