import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
plt.rc('figure',autolayout=True)
plt.rc('axes', titlesize=18, labelsize='large', titleweight='bold', labelweight='bold', titlepad= 10)
from tensorflow.keras import layers
from tensorflow import keras

concrete = pd.read_csv('../input/dl-course-data/concrete.csv')
df = concrete.copy()
df_train = df.sample(frac=0.7,random_state= 0 )
df_valid = df.drop(df_train.index)

X_train = df_train.drop('CompressiveStrength',axis=1)
y_train = df_train['CompressiveStrength']
X_valid = df_valid.drop('CompressiveStrength',axis=1)
y_valid = df_valid['CompressiveStrength']

input_shape = [X_train.shape[1]]
model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(units=512,activation='relu',input_shape = input_shape),
    layers.BatchNormalization(),
    layers.Dense(units = 512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(units=512,activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1)
])
model.compile(optimizer='sgd',loss='mae',metrics=['mae'])
history = model.fit(
    X_train,y_train,
    batch_size=64,
    epochs=100,
    validation_data=(X_valid,y_valid),
    verbose = 0
)
history_df = pd.DataFrame(history.history)
history_df.loc[0:,['loss', 'val_loss']].plot()
print("Minimum Validation Loss:{:0.4f}".format(history_df['loss'].min()))
plt.show()

