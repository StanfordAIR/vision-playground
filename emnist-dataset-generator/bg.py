import cv2
from scipy import ndimage

img = cv2.imread("field-images/field.jpg")
i = 0
for r in range(4):
    rotated = ndimage.rotate(img, r*90)
    height, width, channels = img.shape 
    xScale = int((width)/10) # don't recalculate in for loop in production
    yScale = int((height)/10)
    for x in range(xScale):
        for y in range(yScale):
            crop_img = img[10*y:10*y+100, 10*x:10*x+100]
            h,w,channels=crop_img.shape
            if h==100 and w==100:
                cv2.imwrite("backgrounds/background"+str(i)+".png", crop_img)
                i += 1
