import cv2
from scipy import ndimage

img = cv2.imread("./test.png")
for r in range(4):
    rotated = ndimage.rotate(img, r*90)
    height, width, channels = img.shape 
    xScale = int((width)/10) # don't recalculate in for loop in production
    yScale = int((height)/10)
    for x in range(xScale):
        for y in range(yScale):
            crop_img = img[10*y:10*y+50, 10*x:10*x+50]
            h,w,channels=crop_img.shape
            if h==50 and w==50:
                cv2.imwrite("img1/background" + "r" + str(r)+"x"+str(x)+"y"+str(y) + ".png", crop_img)
