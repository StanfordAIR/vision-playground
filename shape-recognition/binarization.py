import numpy as np
import cv2



def main():

    showImgs = False

    for i in range(0, 100):
        name = str(i)
        img = cv2.imread('test/' + name + '.png')

        img = img[0:48, 0:48]

        displayImg(img, "Original Image", showImgs)

        #grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img2 = cv2.medianBlur(img, 3)


        kmeansImg, centers, labels = kmeans(img2, 3)

        displayImg(kmeansImg, 'After Kmeans', showImgs)

        grayImage = cv2.cvtColor(kmeansImg, cv2.COLOR_BGR2GRAY)

        displayImg(grayImage, "Grayscaled Image", showImgs)


        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        grayImage = clahe.apply(grayImage)

        displayImg(grayImage, "Grayscaled Image after CLAHE", showImgs)


        kmeansGreyImg, centersGrey, labelsGrey = kmeans(grayImage, 2)

        #print centersGrey


        ret, thresh = cv2.threshold(grayImage, calculateThreshold(centersGrey), 255, 0)

        displayImg(thresh, "Thresholded Image", showImgs)


        invertedImg = blackBackground(thresh)

        displayImg(invertedImg, "Inverted Image", showImgs)

        bordersize=2
        invertedImg=cv2.copyMakeBorder(invertedImg, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )

        displayImg(invertedImg, "Bordered img", showImgs)



        im2, contours, hierarchy = cv2.findContours(invertedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largestContour = findLargestContour(contours)
        #cv2.drawContours(invertedImg, contours, -1, (0,255,0), 3)

        if cv2.pointPolygonTest(largestContour, (1, 1), measureDist=False) >= 0:
            contours.remove(largestContour)
            largestContour = findLargestContour(contours)


        mask = np.zeros((invertedImg.shape[0], invertedImg.shape[1], 1), dtype = "uint8")

        cv2.drawContours(mask, [largestContour], 0, (255,255,255), -1)

        #print invertedImg.shape

        mask = mask[2:50, 2:50]

        displayImg(mask, "Mask", showImgs)

        cv2.imwrite('test-ouput/' + name + '.png', mask)

        ###Stuff for IDing color - not needed for shape identification

        '''maskedImg = cv2.bitwise_and(img, img, mask=mask)

        cv2.imwrite('Output/' + name + '.png', maskedImg)

        displayImg(maskedImg, "Masked Image")

        shapeCutOut = []
        
        for i in range(0, maskedImg.shape[0]):
            for j in range(0, maskedImg.shape[1]):
                if mask[i][j]:
                    shapeCutOut = np.append(shapeCutOut, maskedImg[i][j])
        

        colorIDkmeansImg, colorIDcenters, colorIDLabels = kmeans(shapeCutOut, 2)
        unique, counts = np.unique(colorIDLabels, return_counts=True)
        countsDict = dict(zip(unique, counts))

        if countsDict[0] > countsDict[1]:
            shapeColor = colorIDcenters[0]
            alphanumericColor = colorIDcenters[1]
        else:
            shapeColor = colorIDcenters[1]
            alphanumericColor = colorIDcenters[0]

        #print tuple((int(shapeColor[0]), int(shapeColor[1]), int(shapeColor[2]) ))
        cv2.circle(maskedImg, (0, 0), 10, tuple((int(shapeColor[0]), int(shapeColor[1]), int(shapeColor[2]) )), -1)
        cv2.circle(maskedImg, (50, 50), 10, tuple((int(alphanumericColor[0]), int(alphanumericColor[1]), int(alphanumericColor[2]) )), -1)
        displayImg(maskedImg, "ID'd color")'''

    


def findLargestContour(contours):
    contourAreas = []
    for contour in contours:
        contourAreas.append(cv2.contourArea(contour))

    largestContour = contours[contourAreas.index(max(contourAreas))]
    return largestContour 


def calculateThreshold(centers):
    center1 = np.mean(centers[0])
    center2 = np.mean(centers[1])
    avg = np.mean([center1, center2])
    return int(avg)



def blackBackground(img):
    if img[1,1] != 0:
        return cv2.bitwise_not(img)
    else:
        return img

def displayImg(img, frameName, displayImg):
    if displayImg:
        cv2.imshow(frameName, img)
        cv2.waitKey()
        cv2.destroyAllWindows()


def kmeans(img, numClusters):
    #print img.shape
    Z = img.reshape((-1,3))

    #convert to np.float32
    Z = np.float32(Z)

    #define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = numClusters
    ret, labels, centers=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    #Now convert back into uint8, and make original image 
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    retImg = res.reshape((img.shape))
    return retImg, centers, labels


def get_neighborhood_variance(neighboorhood, mean):
    error = 0
    for i in range(neighboorhood.shape[0]):
        for j in range(neighboorhood.shape[1]):
            pixel = neighboorhood[i,j]
            error += np.linalg.norm(pixel - mean)
    error /= neighboorhood.size
    return error


def generate_rough_mask(image, threshold, radius):
    points = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            neighboorhood = image[x-radius:x+radius,y-radius:y+radius]
            if get_neighborhood_variance(neighboorhood, image[x,y]) < threshold:
                points.append((x,y))
    return points



if __name__ == "__main__":
    main()
