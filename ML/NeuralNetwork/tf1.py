import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.rc("figure",autolayout=True)
plt.rc("axes",titlesize=18,labelsize='large',titleweight='bold',labelweight='bold',titlepad=10)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
concrete = pd.read_csv('../input/dl-course-data/concrete.csv')
print(concrete.head())
model = keras.Sequential([
    layers.Dense(units=512,activation='relu',input_shape = [8]),
    layers.Dense(units=512,activation='relu'),
    layers.Dense(units=512,activation='relu'),
    layers.Dense(units=1)
])

activation_layer = layers.Activation('swish')

x = tf.linspace(-3.0, 3.0, 100)
y = activation_layer(x) # once created, a layer is callable just like a function

plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()