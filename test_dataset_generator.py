import numpy as np
import cv2
import random

def drawRectangle(img):
    colors = [(233, 233, 233), (10, 10, 10), (133, 133, 133), (35, 35, 200)]
    point1 = (int(10*random.random()), int(10*random.random()))
    point2 = (int(90 + 10*random.random()), int(90 + 10*random.random()))
    print colors[int(random.random()*4)]
    cv2.rectangle(img, point1, point2, colors[int(4*random.random())], -1);

def overlayChar(data, img):
    num = int(data.shape[0]*random.random())
    M = cv2.getRotationMatrix2D((32/2,32/2),20*(random.random()-.5),1)
    dst = cv2.warpAffine(data[num], M, (32,32))
    dst = 255-dst;
    dst = cv2.resize(dst, (100, 100), interpolation=cv2.INTER_CUBIC)
    new_dst = (dst, dst, dst)
    new_dst = np.stack(new_dst, axis=2);
    img += new_dst
    #img = cv2.addWeighted(img, 1, dst, 1, 0);
    return img;

random.seed();
data = np.load("alphanum-hasy-data-X.npy")
img = np.zeros([100, 100, 3], np.uint8);
drawRectangle(img);
dst = overlayChar(data, img);
cv2.imshow("Test", dst);
cv2.waitKey(0);
