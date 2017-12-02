'''
Just a simple webcam implementation for ocr.py

'''
import cv2
import ocr
from argparse import Namespace
import time


def show_webcam(mirror=True):
	cam = cv2.VideoCapture(0)
	while True:
		ret_val, img = cam.read()
		if mirror: 
			img = cv2.flip(img, 1)
		cv2.imshow('my webcam', img)
		
		# args.add_argument("-i", "--image", required=True,
	 # 		help="path to input image to be OCR'd")
	 # 	args.add_argumentadd_argument("-p", "--preprocess", type=str, default="thresh",
	 # 		help="type of preprocessing to be done")
		ocr.readImg(img, "")
		if cv2.waitKey(1) == 27: 
			break  # esc to quit
	cv2.destroyAllWindows()

def main():
	# construct the argument parse and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--image", required=True,
	# 	help="path to input image to be OCR'd")
	# ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	# 	help="type of preprocessing to be done")
	# args = vars(ap.parse_args())
	

	show_webcam(mirror=True)

if __name__ == '__main__':
	main()