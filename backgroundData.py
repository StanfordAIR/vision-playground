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

# Next step: 
#   for image in ./img1:
#       s = np.random.randn(10)
#       shape = cv.imread ("shape"+str(s))
#       newShape = keras.rotate(shape, random degrees)
#       r = np.random.randn(500000) # about 700000 images in the EMNIST, less if we only count capital letters and numbers
#       name = EMNIST[r].name
#       letter = EMNIST[r].image
#       shape = cv2.copyTo(keras.rotate(letter, random degrees), shape, NO_TRANSLATION,NO_ROTATION)
#       newImage = cv2.copyTo(shape,image,SOME_TRANSLATION,SOME_ROTATION)
#       cv2.imwrite("../newimgs/"+"letter"+letter+"shape"+str(s)+"index"+str(index)+".png",newImage)