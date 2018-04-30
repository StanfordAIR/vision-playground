import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def main():
    showImgs = False
    for i in range(0, 10000):
        name = str(i)
        img = cv2.imread('../dataset/'+name+'.png')

        img2 = cv2.blur(img, (7, 7))
        img2 = (img2 * .8).astype(np.uint8)

        img3 = cv2.blur(img, (11, 11))
        img3 = (img3 * .5).astype(np.uint8)

        new_img = np.concatenate((img, img2, img3), axis=2)

        #displayImg(img, "Original Image", showImgs)
        labels = kmeans(new_img, 3)
        img1, img2, img3 = generateThreeImages(labels)
        plt.imsave('img1/'+name+'.png', img1)
        plt.imsave('img2/'+name+'.png', img2)
        plt.imsave('img3/'+name+'.png', img3)

def generateThreeImages(labels):
	zeros = np.zeros((50, 50), np.uint8)
	ones = np.zeros((50, 50), np.uint8)
	ones.fill(255)

	img1 = np.zeros((50, 50), np.uint8)
	img2 = np.zeros((50, 50), np.uint8)
	img3 = np.zeros((50, 50), np.uint8)

	labels = np.reshape(labels, (50, 50))
	img1 = np.where(labels == 0, zeros, ones)
	img2 = np.where(labels == 1, zeros, ones)
	img3 = np.where(labels == 2, zeros, ones)

	return img1, img2, img3
def cvtBinaryImage(img):
	mat1 = np.zeros((50, 50), np.uint8)
	mat2 = np.zeros((50, 50), np.uint8)
	mat2.fill(255)
	final = np.where(img, mat1, mat2);
	return final

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
    plt.imshow(img)
    plt.show()

def kmeans(img, numClusters):
    #print img.shape
    Z = img.reshape((-1,3))

    #convert to np.float32
    Z = np.float32(Z)
    kmeans = KMeans(n_clusters=3, init="k-means++", random_state=0).fit(Z.reshape(2500, 9))
    return kmeans.labels_

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
