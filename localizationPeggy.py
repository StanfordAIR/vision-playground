import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

#img = cv2.imread('robot.jpg',0)
#img = cv2.imread('butterfly.png', 0)
img = cv2.imread('temp.png',0)

cv2.imshow('whatever', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv2.xfeatures2d.SURF_create(1609)
#surf = cv2.SURF(400)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)

#surf.setUpright(True)

img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

#cv2.imshow('something', img2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
plt.imshow(img2),plt.show()