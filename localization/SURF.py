import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

img = cv2.imread('SURF.png',0)

# Create SURF object. You can specify params here or later. #
surf = cv2.xfeatures2d.SURF_create(2058.75) # set Hessian Threshold to 400
kp, des = surf.detectAndCompute(img,None) # find keypoints and descriptors directly
surf.setUpright(True)
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

plt.imshow(img2),plt.show()