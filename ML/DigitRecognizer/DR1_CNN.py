import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
df_train = pd.read_csv('../input/XPC-Team-Digit-Recognizer/train.csv')
X = df_train.copy()
y = X.pop('label')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train = X_train.values/255.0
y_train = y_train.values
X_test = X_test.values/255.0
y_test = y_test.values
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
print(X_train)
print(X_train.shape)
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), padding='Same', activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, kernel_size=(3, 3), padding='Same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # بلوک دوم کانولوشن (تعداد فیلترها بیشتر می‌شود)
    layers.Conv2D(64, kernel_size=(3, 3), padding='Same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, kernel_size=(3, 3), padding='Same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # لایه نهایی (تصمیم‌گیری)
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    epochs=30,  # تعداد دورها را بیشتر می‌گذاریم، اگر لازم نباشد ارلی استاپ قطع می‌کند
                    batch_size=64,
                    validation_data=(X_test, y_test),
                    callbacks=[learning_rate_reduction, early_stopping])
val_loss, val_accuracy = model.evaluate(X_test, y_test)
print(f"Final Validation Accuracy: {val_accuracy * 100:.2f}%")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
df_test = pd.read_csv('../input/XPC-Team-Digit-Recognizer/test.csv')
rX_Test = df_test.copy()
rX_Test = rX_Test.values / 255.0
rX_Test = rX_Test.reshape(-1, 28, 28, 1)
predictions_prob = model.predict(rX_Test)
predictions = np.argmax(predictions_prob, axis=1)
print(predictions)
