# this converts all images in given directories to grayscale. 
# Must already have images in training-images directories generated from reader.py in keras-image-preprocessing.

# usage: $ python convert-grey.py

import cv2
import sys
import os
from os.path import isfile

from keras.preprocessing import image as im

i = 0

if not os.path.exists("conv-test"):
	os.makedirs("conv-test")

if not os.path.exists("conv-train"):
	os.makedirs("conv-train")

for f in os.listdir("training-images"):
	if not isfile("training-images/"+f):
		for g in os.listdir("training-images/"+f):
			if not os.path.exists("conv-train/"+f):
				os.makedirs("conv-train/"+f)
			img = cv2.imread("training-images/"+str(f)+"/"+str(g), 0)
			cv2.imwrite("conv-train/"+str(f)+"/im"+str(i)+".png", img)
			i = i+1

for f in os.listdir("test-images"):
	if not isfile("test-images/"+f):
		for g in os.listdir("test-images/"+f):
			if not os.path.exists("conv-test/"+f):
				os.makedirs("conv-test/"+f)
			img = cv2.imread("test-images/"+str(f)+"/"+str(g),0)
			cv2.imwrite("conv-test/"+str(f)+"/im"+str(i)+".png", img)
			i = i+1