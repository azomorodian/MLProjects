import matplotlib
matplotlib.use('tkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
plt.rc('figure',autolayout=True)
plt.rc('axes',titlesize=18,labelsize='large',titleweight='bold',labelweight='bold',titlepad = 10)
plt.rc('image',cmap='magma')
warnings.filterwarnings('ignore')
image_path = '../input/computer-vision-resources/car_feature.jpg'
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image)

#Define kernel
kernel = tf.constant([
    [-1,-1,-1],
    [-1, 8,-1],
    [-1,-1,-1]
], dtype=tf.float32)

# Reformat for batch compatibility.
image = tf.image.convert_image_dtype(image,dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel,[*kernel.shape,1,1])
print(kernel.shape)
print(*kernel.shape)
print([kernel.shape])
print([*kernel.shape])
print([*kernel.shape,1,1])
image_filter = tf.nn.conv2d(input = image, filters = kernel, strides=1 , padding='SAME')
image_detect = tf.nn.relu(image_filter)
plt.figure(figsize=(12,6))
plt.subplot(231)
plt.imshow(tf.squeeze(image),cmap = 'gray')
plt.axis('off')
plt.title('Input')
plt.subplot(232)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title('Filter')
plt.subplot(233)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title('Detect')


image_condense = tf.nn.pool(input = image_detect, window_shape=(2,2),pooling_type='MAX', strides=(2,2), padding='SAME')
plt.subplot(234)
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.title('Condense')
print(image_condense.shape)
print(image.shape)
print(image_detect.shape)
print(image_filter.shape)
plt.show()