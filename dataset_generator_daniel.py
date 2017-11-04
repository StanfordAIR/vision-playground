import numpy as np
import cv2
import random

def drawRectangle(img):
    # Off-White, Black, Grey, Red
    colors = [(233, 233, 233), (133, 133, 133), (35, 35, 200),(255,255,255)]
    # point1 = (int(10*random.random()), int(10*random.random()))
    # point2 = (int(90 + 10*random.random()), int(90 + 10*random.random()))
    # NOTE: POINT: (x,y)
    point1 = (20,20)
    point2 = (280,280)
    # random_color = colors[int(random.random()*3)]
    random_color =  colors[3]
    # NOTE: Draws a rectangle
    cv2.rectangle(img, point1, point2, random_color, -1);

def drawCircle(img):
    # Off-White, Black, Grey, Red
    colors = [(233, 233, 233), (133, 133, 133), (35, 35, 200),(255,255,255)]
    random_color = colors[3]
    cv2.circle(img,(150,150),100,random_color,-1)

def padCharImg(data):
    padded_data = np.zeros((300,300,3),np.uint8)
    padded_data[100:200,100:200, 0:3] = data
    return padded_data

def overlayChar(data, img):
    # Inverts the color
    data = 255-data;
    num = int(data.shape[0]*random.random())
    # Rotates the image between -10 to 10 degrees
    M = cv2.getRotationMatrix2D((32/2,32/2),180*(random.random()-.5),1)
    # NOTE: OpenCV transformation (warpAffine) Applies the transformation
    dst = cv2.warpAffine(data[num], M, (32,32))

    # NOTE: OpenCV transformation (RESIZE the handwriting image (alphanum..) to match the black image)
    dst = cv2.resize(dst,(100,100),interpolation=cv2.INTER_NEAREST)

    new_dst = (dst, dst, dst)
    new_dst = np.stack(new_dst, axis=2);
    # places the smaller dst character image in a larger dst that matches the background img size
    new_dst = padCharImg(new_dst)

    #img = cv2.addWeighted(img, 1, dst, 1, 0);
    # This is subtracted in order to convert the new_dst white (255,255,255) rgb values to black values (0,0,0)
    return img-new_dst;

# Creates the alphanumeric "text" image array
data = np.load("alphanum-hasy-data-X.npy")

# Creates a black image array (height,width,rgb)
img = np.zeros((300, 300, 3), np.uint8);
drawCircle(img);

# Overlays the "text" image over the "visual" image + modifies the image in orientation etc.
dst = overlayChar(data, img);
cv2.namedWindow("Test",cv2.WINDOW_NORMAL)
cv2.imshow("Test", dst);
cv2.waitKey(0);
