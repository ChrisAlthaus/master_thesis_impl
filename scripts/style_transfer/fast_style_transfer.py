from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
#Enable for tensorflow<2.0
tf.enable_eager_execution() 

#import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
from PIL import Image
import time
import functools

import os
import argparse
import shutil
import logging

import tensorflow_hub as hub
import datetime


parser = argparse.ArgumentParser(description='')

parser.add_argument('-styles', '--pathStyles', required=True, help='path to dir of image styles')
parser.add_argument('-images', '--pathImages', required=True, help='path to dir of images')
parser.add_argument('-dFile', '--dFile', default=None, help='run style transfer only on list of input directories')
parser.add_argument('-o', '--output', required=True, help='output path for stylished images')
parser.add_argument('-nC', '--numContents', required=True, help='number of content/base images. Disable with -1 = all images.')
parser.add_argument('-split', default=1.0, type=float, help='used for split/subset of content images (Percentage value).')
parser.add_argument('-splitdirection', default='begin', help='Either reduce to first or last split-percentage of content images.')
parser.add_argument('-nS', '--numStyles', required=True, help='number of styles to apply to each image')
parser.add_argument('-add','--add', default=False, action='store_true')
parser.add_argument('-dimS','--dimStyle', default=256, type=int, help='dimension used for scaling style image, NOT USED')
parser.add_argument('-dimC','--dimContent', default=1024, type=int, help='dimension used for scaling style image, NOT USED')
parser.add_argument('-labelmode', required=True, default='aggregate', help='Used for naming of output file.')
parser.add_argument('-v','--verbose', default=False, action='store_true')

args = parser.parse_args()

if args.verbose:
	logging.basicConfig(level=logging.DEBUG)
else:
	logging.basicConfig(level=logging.INFO)

#tf.debugging.set_log_device_placement(True)

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')  

content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 
		'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

# https://commons.wikimedia.org/wiki/File:Vassily_Kandinsky,_1913_-_Composition_7.jpg
style_path = tf.keras.utils.get_file('kandinsky5.jpg',
		'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


def tensor_to_image(tensor):
	tensor = tensor*255
	tensor = np.array(tensor, dtype=np.uint8)
	if np.ndim(tensor)>3:
		assert tensor.shape[0] == 1
		tensor = tensor[0]
	return PIL.Image.fromarray(tensor)
		
def load_img(path_to_img,scale=False,resize_shape=None):
	img = tf.io.read_file(path_to_img)
	
	try:
		img = tf.image.decode_image(img, channels=3,expand_animations=False)
	except:
		print("Error occured in decode_image function.")
		print("Image path: ", path_to_img)
		return None

	img = tf.image.convert_image_dtype(img, tf.float32)

	if scale:	
		#shape = tf.cast(tf.shape(img)[:-1], tf.float32)
		#long_dim = max(shape)
		#Output image should be at the longest side = max_dim
		#scale = max_dim/long_dim
		#scale = min(1,max_dim / long_dim)

		#new_shape = tf.cast(shape * scale, tf.int32) #TODO: maybe crop better than resize

		new_shape = tf.cast(resize_shape[:-1], tf.int32)[1:]
		#print(new_shape)
		img = tf.image.resize(img, new_shape)
		
	img = img[tf.newaxis, :]
	
	return img 
  
def imshow(image, title=None):
	if len(image.shape) > 3:
		image = tf.squeeze(image, axis=0)
	
	plt.imshow(image)
	if title:
		plt.title(title)


assert os.path.isdir(args.pathStyles), "Style folder not exists."
assert os.path.isdir(args.pathImages), "Image folder not exists."
assert os.path.isdir(args.output), "Output folder not exists."

"""output_dir = os.path.join(args.output, datetime.datetime.now().strftime('%m/%d_%H-%M-%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    raise ValueError("Output directory %s already exists."%output_dir)"""
output_dir = args.output

style_paths = [os.path.join(args.pathStyles, x) for x in os.listdir(args.pathStyles)]
style_paths = np.array(style_paths)

image_paths = [os.path.join(args.pathImages, x) for x in os.listdir(args.pathImages)]
image_paths = np.array(image_paths)

print("Number of content images: ",len(image_paths))
print("Number of style images: ",len(style_paths))

# Remove style & content images from path lists which are too large
MAX_SIZE = 20971520 # 20 MBytes
for file_path in style_paths:
	if os.path.getsize(file_path) > MAX_SIZE:
		style_paths = np.delete(style_paths, np.where(style_paths == file_path))		
		logging.debug("Removed style image %s with size %d ."%(file_path,os.path.getsize(file_path)))
		
for file_path in image_paths:
	if os.path.getsize(file_path) > MAX_SIZE:
		image_paths = np.delete(image_paths, np.where(image_paths == file_path))		
		logging.debug("Removed content image %s with size %d ."%(file_path,os.path.getsize(file_path)))

print("Number of content images without too large images: ",len(image_paths))
print("Number of style images without too large images: ",len(style_paths))

		
# Slicing can be used to divide dataset into train/val subset
assert args.split >= 0 and args.split <= 1
subsetlength = int(args.split * len(image_paths))
if args.splitdirection == 'begin':
	image_paths = image_paths[:subsetlength]
elif args.splitdirection == 'end':
	image_paths = image_paths[-subsetlength:]
else:
	raise ValueError()
print("Number of subset of content images: ",len(image_paths))


print("STYLIZE IMAGES ... ")
print("Number of styles = ",int(args.numStyles))

# copy base image to output folder
if args.add:
	for img_path in image_paths:
		out_img_path = img_path.replace(args.pathImages,args.output)
		shutil.copy2(img_path, out_img_path)


import time
start_time = time.time()

num_counter = 0
# create stylished images for each base image
for img_path in image_paths:
	#print("Stylize image ", os.path.basename(img_path))
	#print("\nContent image ", img_path)

	style_select = style_paths[np.random.choice(len(style_paths), size=int(args.numStyles), replace=False)]
	#print(style_path, style_select)
	out_img_base = os.path.join(args.output, os.path.basename(img_path))

	for style_path in style_select:
		logging.debug("Style image path: %s"%style_path)
		
		#Filename: contentImNumber (coco=12digits) _ styleImNumber(painterByNumbers=6digits)
		
		if args.labelmode == 'aggregate':
			style_filename = os.path.basename(style_path)
			style_im_number = "%06d"%int(os.path.splitext(style_filename)[0])
			file_name = os.path.basename(img_path).replace('.jpg','') + '_' + str(style_im_number) + os.path.splitext(style_filename)[1] 
		elif args.labelmode == 'keepcontent':
			file_name = os.path.basename(img_path)
		else:
			raise ValueError('No valid output file labeling mode.')

		out_img_path = os.path.join(output_dir, file_name)
		logging.debug("Output file path: %s"%out_img_path)

		#Stylize image 
		content_image = load_img(img_path,scale=False)
		
		if(content_image is None):
			continue
		#Output size should be shape of content image
		#content_max_dim = max(tf.cast(tf.shape(content_image)[:-1], tf.float32))
		content_shape = tf.shape(content_image)
		#target_np = content_shape.numpy()
		#target_np[1] = target_np[1] * 2
		#target_np[2] = target_np[2] * 2
		#content_shape = tf.convert_to_tensor(target_np, np.float32)

		#style_image = load_img(style_path,scale=True,resize_shape=tf.shape(content_image))
		style_image = load_img(style_path,scale=True,resize_shape=content_shape)
		
		if(style_image is None):
			continue

		logging.debug("Shape Content Image: %s"%content_image.shape)
		logging.debug("Shape Style Image: %s"%style_image.shape)


		stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]	
		output_img = tensor_to_image(stylized_image)
		#print(type(output_img.size))
		logging.debug("Shape Output Image: %s"%str(output_img.size))
		output_img.save(out_img_path)

		
	
	num_counter = num_counter + 1
	if num_counter % 10 == 0:
		print("Number images transformed: %d"%(num_counter))
	if num_counter == 100:
		print("Time for 100 images = %s seconds" % (time.time() - start_time))

	if int(args.numContents) != -1 and num_counter >= int(args.numContents):
		break
	#print("Stylize image %s done."%(os.path.basename(img_path)))
print("Time: %s seconds" % (time.time() - start_time))
print("STYLIZE IMAGES DONE.")
		
		
