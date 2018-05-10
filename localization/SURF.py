import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

#img = cv2.imread('robot.jpg',0)
#img = cv2.imread('butterfly.png', 0)
img = cv2.imread('SURF.png',0)

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv2.xfeatures2d.SURF_create(2058.75)
#surf = cv2.SURF(400)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)

surf.setUpright(True)
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

plt.imshow(img2),plt.show()

#Next Steps:
#Replace SURF.png with video stream
#Retrun an image for each region of interest
#Some kind of if statemtnt to eliminate duplicate shapes