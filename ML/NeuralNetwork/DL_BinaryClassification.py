import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
ion = pd.read_csv('../input/dl-course-data/ion.csv',index_col= 0)
display(ion)
display(ion.shape)
print(list(ion.columns))
print(ion['Class'])
df = ion.copy()

df['Class'] = df['Class'].map({'good':0,'bad': 1})
df_train = df.sample(frac=0.7,random_state=0)
df_valid = df.drop(df_train.index)
pd.set_option('display.max_columns', None)

max_ = df_train.max(axis = 0)
print(max_)
min_ = df_train.min(axis = 0)
print(min_)
df_train = (df_train - min_)/(max_-min_)
print(df_train)
df_valid = (df_valid - min_)/(max_-min_)
df_train.dropna(axis = 1,inplace = True)
df_valid.dropna(axis = 1,inplace = True)
print(df_train)
print(df_valid)
X_train = df_train.drop('Class', axis = 1)
y_train = df_train['Class']
X_valid = df_valid.drop('Class', axis = 1)
y_valid = df_valid['Class']
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(units=4,activation='relu',input_shape = [33]),
    layers.Dense(units=4,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta = 0.001,
    restore_best_weights=True
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=1000,
    batch_size=512,
    callbacks=[early_stopping],
    verbose = 0
)
history_df = pd.DataFrame(history.history)
history_df.loc[5:,['loss','val_loss']].plot()
history_df.loc[5:,['binary_accuracy','val_binary_accuracy']].plot()
print(("Best Validation Loss : {0:.4f}"+"\nBest Validation Accuracy : {0:0.4f}")
      .format(history_df['val_loss'].min(),history_df['val_binary_accuracy'].max()))
plt.show()
