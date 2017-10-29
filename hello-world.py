import cv2
import numpy as np

if ("3.3" not in cv2.__version__):
    print "ERROR - make sure you have the correct OpenCV version"
    exit(0)

blank_image = np.zeros((200, 200, 3), np.uint8)

cv2.imshow("Test - it worked!", blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
