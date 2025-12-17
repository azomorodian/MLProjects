import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
sns.set_style("whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", titlesize=18, labelsize='large',labelweight='bold', titleweight='bold' ,titlepad = 10)
import pandas as pd
red_wine = pd.read_csv('../input/dl-course-data/red-wine.csv')
df_train = red_wine.sample(frac=0.7,random_state=0)
df_valid = red_wine.drop(df_train.index)

X_train = df_train.drop('quality',axis=1)
X_valid = df_valid.drop('quality',axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(1024, activation='relu',input_shape = [11]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1)
])
model.compile(optimizer='adam',loss='mae')
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid,y_valid),
    batch_size= 256,
    epochs=100,
    verbose=0
)
history_df = pd.DataFrame(history.history)
print(history_df.head())
history_df.loc[:,['loss','val_loss']].plot()
plt.show()


