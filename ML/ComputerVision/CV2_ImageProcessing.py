import sys
import os
kaggle_learn_tool_path = r'../Kaggle/'
sys.path.append(kaggle_learn_tool_path)
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
plt.rc('figure',autolayout=True)
plt.rc('axes', titlesize=12, labelsize='small', titleweight='bold', labelweight='bold',titlepad=10)
plt.rc('image',cmap = 'magma')
tf.config.run_functions_eagerly(True)
image_path = '../input/computer-vision-resources/car_illus.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[400, 400])


img = tf.squeeze(image).numpy()
plt.figure(figsize=(12,12))
plt.subplot(2,4,5)
plt.imshow(img,cmap='gray')
plt.axis('off')
import learntools.computer_vision.visiontools as visiontools
from learntools.computer_vision.visiontools import edge, bottom_sobel, emboss, sharpen
kernels = [edge, bottom_sobel, emboss, sharpen]
names = ["Edge Detect", "Bottom Sobel", "Emboss", "Sharpen"]


for i, (kernel, name) in enumerate(zip(kernels, names)):
    plt.subplot(2, 4, i+1)
    visiontools.show_kernel(kernel)
    plt.title(name)


kernel = tf.constant([
  [-1,-1,-1],
  [-1, 8,-1],
  [-1,-1,-1]
])
plt.subplot(2,4,6)
visiontools.show_kernel(kernel)
plt.title('my define')

plt.tight_layout()
plt.show()