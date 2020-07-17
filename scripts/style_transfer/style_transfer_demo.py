import tensorflow as tf

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

# https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

#Define a function to load an image and limit its maximum dimension to 512 pixels.
@tf.function
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3,expand_animations=False)
  #img = tf.image.decode_jpeg(img, channels=3,expand_animations=False)
  img = tf.image.convert_image_dtype(img, tf.float32)
 
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  print(shape)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

#Save image
def imsave(image, title):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  plt.savefig(title)
  #if title:
  #  plt.title(title)


content_image = load_img(content_path)
style_image = load_img(style_path)

#plt.subplot(1, 2, 1)
#imshow(content_image, 'Content Image')

#plt.subplot(1, 2, 2)
#imshow(style_image, 'Style Image')

#Apply style transfer
import tensorflow_hub as hub
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
image = tensor_to_image(stylized_image)

imsave(image, os.path.basename(content_path) + '_' + os.path.basename(style_path))
