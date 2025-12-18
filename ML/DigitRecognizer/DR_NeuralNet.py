import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

df = pd.read_csv('../input/XPC-Team-Digit-Recognizer/train.csv')
df_X = df.copy()
df_y = df_X.pop("label")
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size = 0.2, random_state = 41)
#print(("train X {}").format(X_train.head()))
#print(("train y {}").format(y_train.head()))
#print(("test  x {}").format(X_test.head()))
#print(("test  y {}").format(y_test.head()))
X_train = X_train.values  #X_train.to_numpy()
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

input_shape = [X_train.shape[1]]
print(input_shape)
early_stopping = callbacks.EarlyStopping(
    min_delta = 0.0001,
    monitor = 'val_loss',
    patience = 50,
    verbose = 0,
    restore_best_weights = True
)
model = keras.Sequential([
    layers.BatchNormalization(input_shape = input_shape),
    layers.Dense(1024, activation='sigmoid',),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(512, activation='sigmoid'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(256, activation='sigmoid'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=256,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping]
                    )
val_loss, val_accuracy = model.evaluate(X_test, y_test)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
df_test = pd.read_csv('../input/XPC-Team-Digit-Recognizer/test.csv');
df_test = df_test.values
predictions_prob = model.predict(df_test)
predictions = np.argmax(predictions_prob, axis=1)
print(predictions)
