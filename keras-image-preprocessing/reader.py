# this will 'mess up' an image randomly (using params in ImageDataGenerator) and save 
# however many generations you want to the gvalidate and gtrain directories.

# Usage: $ python reader.py

from __future__ import absolute_import
from __future__ import print_function

import os

from keras.preprocessing import image as im

train_copies = 4000
test_copies = 1500

datagen = im.ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest')

for f in os.listdir("train"):
	if not os.path.exists("gtrain/"+f):
		os.makedirs("gtrain/"+f)

	for g in os.listdir("train/"+f):
		img = im.load_img("train/"+f+"/"+g)  # this is a PIL image
		x = im.img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
		x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

		# the .flow() command below generates batches of randomly transformed images
		# and saves the results to the `preview/` directory
		i = 0
		for batch in datagen.flow(x, batch_size=1,
		                          save_to_dir="gtrain/"+f, save_prefix=g, save_format='png'):
		    i += 1
		    if i > train_copies:
		        break

for f in os.listdir("validate"):
	if not os.path.exists("gvalidate/"+f):
		os.makedirs("gvalidate/"+f)

	for g in os.listdir("validate/"+f):
		img = im.load_img("validate/"+f+"/"+g)  # this is a PIL image
		x = im.img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
		x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

		# the .flow() command below generates batches of randomly transformed images
		# and saves the results to the `preview/` directory
		i = 0

		for batch in datagen.flow(x, batch_size=1,
		                          save_to_dir="gvalidate/"+f, save_prefix=g, save_format='png'):
		    i += 1
		    if i > test_copies:
		        break

