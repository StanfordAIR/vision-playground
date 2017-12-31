'''
USAGE

to use ocr.py at the command line, use code above line and use:
python ocr.py --image images/example_01.png 
python ocr.py --image images/example_02.png  --preprocess blur

to use ocr.py as a class with static method readImg, use code below line and use:
import ocr

preprocessing = "blur or thresh or whatever"
ocr.readImg(img, preprocessing)

'''

from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
import time


# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image to be OCR'd")
# ap.add_argument("-p", "--preprocess", type=str, default="thresh",
# 	help="type of preprocessing to be done")
# args = vars(ap.parse_args())

# # load the example image and convert it to grayscale
# image = cv2.imread(args["image"])
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow("Image", gray)

# # check to see if we should apply thresholding to preprocess the
# # image
# if args["preprocess"] == "thresh":
# 	gray = cv2.threshold(gray, 0, 255,
# 		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# # make a check to see if median blurring should be done to remove
# # noise
# elif args["preprocess"] == "blur":
# 	gray = cv2.medianBlur(gray, 3)

# # write the grayscale image to disk as a temporary file so we can
# # apply OCR to it
# filename = "{}.png".format(os.getpid())
# cv2.imwrite(filename, gray)

# # load the image as a PIL/Pillow image, apply OCR, and then delete
# # the temporary file
# text = pytesseract.image_to_string(Image.open(filename))
# os.remove(filename)
# print(text)

# # show the output images
# # cv2.imshow("Image", image)
# cv2.imshow("Output", gray)
# cv2.waitKey(0)

#<------------------------------------------------------------------->

def readImg(img, preprocess):
	# load the example image and convert it to grayscale
	
	#image = cv2.imread(img)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.flip(gray, 1)

	#invert if needed
	#gray = cv2.bitwise_not(gray)

	cv2.imshow("Image", gray)

	# check to see if we should apply thresholding to preprocess the
	# image
	if preprocess == "thresh":
		gray = cv2.threshold(gray, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# make a check to see if median blurring should be done to remove
	# noise. I suspect the bilateral filter will be the best for us, but
	# this remains to be tested.
	elif preprocess == "blur":
		#gray = cv2.GaussianBlur(gray, (5,5), 3)
		#gray = cv2.medianBlur(gray, 3)
		gray = cv2.bilateralFilter(gray,9,75,75)

	# write the grayscale image to disk as a temporary file so we can
	# apply OCR to it
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, gray)

	# load the image as a PIL/Pillow image, apply OCR, and then delete
	# the temporary file
	# pytesseract.tesseract_cmd.api.setPageSegMode(TessBaseAPI.pageSegMode.PSM_SINGLE_CHAR);
	text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	print(text)

	# show the output images
	# cv2.imshow("Image", image)
	cv2.imshow("Output", gray)
	time.sleep(1) #for webcam.py
	#cv2.waitKey(0)



	

