import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random
import operator

class KMeansData():
    def __init__(self, pixel, x, y):
        self.pixel = pixel
        self.x = x
        self.y = y


def hashlist(pixel):
	pixel_str = "" + str(pixel[0]) + "-" + str(pixel[1]) + "-" + str(pixel[2])
	return pixel_str

def get_bounding_rectangle(img):
	min_x = 0
	min_y = 0
	for i in range(0, 50):
		flag = False;
		for j in range(0, 50):
			if (img[j][i] == True):
				flag = True;
		if (flag==True):
			min_y = i;
			break;
	for i in range(0, 50):
		flag = False;
		for j in range(0, 50):
			if (img[i][j] == True):
				flag = True;
		if (flag==True):
			min_x = i;
			break;
	max_y = 0
	max_x = 0
	for i in range(min_y+1, 50):
		flag = True;
		for j in range(0, 50):
			if (img[j][i] == True):
				flag = False;
		if (flag==True):
			max_y = i;
			break;
	for i in range(min_x+1, 50):
		flag = True;
		for j in range(0, 50):
			if (img[i][j] == True):
				flag = False;
		if (flag==True):
			max_x = i;
			break;
	return (min_x, min_y, max_x, max_y)

def random_centroids(img, k, size):
	centroids = []
	for i in xrange(k):

		if i != 0:
			num = int(random.random() * (size-1))
			while (np.array_equal(img[num].pixel, centroids[i-1])):
				num = int(random.random() * (size-1))
			centroids.append(img[num].pixel);
		else:
			centroids.append(img[int(random.random() * (size-1))].pixel);
	print centroids
	return centroids

def find_closest(pixel, centroids, k):
	min_distance = 10000000000
	distance = 0
	min_id = 100
	for i in xrange(k):
		distance = np.linalg.norm(pixel - centroids[i])**2

		if(distance < min_distance):
			min_distance = distance
			min_id = i

	return min_id

def assignment(img, groups, centroids, k, size):
	for i in xrange(size):
		groups[find_closest(img[i].pixel, centroids, k)].append(img[i]);
	return groups

def update_centroids(groups, centroids, k):
	avgs = np.zeros([3, 3], dtype=float)
	nums = np.ones(2, dtype=int)

	for i in xrange(k):
		elements_in_group = groups[i]

		for element in elements_in_group:

			avgs[i] += element.pixel
			nums[i] += 1;

	for i in xrange(k):
		avgs[i][0] /= nums[i]
		avgs[i][1] /= nums[i]
		avgs[i][2] /= nums[i]

		centroids[i] = np.copy(avgs[i]);
	return centroids

def k_means_algorithm(img, k, size=0):
	groups = dict()


	centroids = random_centroids(img, k, size)
	print centroids
	for i in xrange(10):
		for j in xrange(k):
			groups[j] = []

		groups = assignment(img, groups, centroids, k, size)

		centroids = update_centroids(groups, centroids, k)
	return centroids, groups
	
def figure_out_shape_color(img, mask):
	i = 0
	color_list = []
	for x in xrange(50):
		for y in xrange(50):
			if(mask[x][y]):
				color_list.append(KMeansData(img[x][y], x, y))
				i += 1
	centroids, groups = k_means_algorithm(color_list, 2, i)
	if(len(groups[0]) > len(groups[1])):
		return centroids[0], centroids[1], groups
	else:
		return centroids[1], centroids[0], groups


def get_char(img, mask, groups, color):
	new_img = np.zeros([50, 50, 3], dtype=np.uint8)
	if (len(groups[0]) > len(groups[1])):
		for element in groups[1]:
			new_img[element.x][element.y] = color
	else:
		for element in groups[0]:
			new_img[element.x][element.y] = color	
	return new_img

for i in range(19, 10000):
	img = cv.imread('../dataset/'+str(i)+'.png')
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

	img_1 = img;
	mask = np.zeros(img.shape[:2],np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	rect = (0, 0, 49, 49)
	cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	img_copy = img*mask2[:,:,np.newaxis]
	img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
	mask3 = img_copy > 10;
	new_rect = get_bounding_rectangle(mask3)
	color0, color1, groups = figure_out_shape_color(img_1, mask3)
	print color0, color1
	mat1 = np.zeros((50, 50), np.uint8)
	mat2 = np.zeros((50, 50), np.uint8)
	mat2.fill(255)

	final = np.where(mask3, mat1, mat2);
	
	cv.imwrite('shape_mask/'+str(i)+'.png', final)

	char_img = get_char(img, mask3, groups, np.array([255, 255, 255], dtype=np.uint8))
	cv.imwrite('letter_mask/'+str(i)+'.png', char_img)

	plt.show()
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
# cv.grabCut(img,mask,new_rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask2[:,:,np.newaxis]
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# mask4 = img>10;
# y, x = np.nonzero(mask4)
# x = x - np.mean(x)
# y = y - np.mean(y)
# coords = np.vstack([x, y])
# cov = np.cov(coords)
# evals, evecs = np.linalg.eig(cov)
# sort_indices = np.argsort(evals)[::-1]
# evec1, evec2 = evecs[:, sort_indices]
# x_v1, y_v1 = evec1  # Eigenvector with largest eigenvalue
# x_v2, y_v2 = evec2
# new_rect = get_bounding_rectangle(mask4)
# scale = 20
# plt.imshow(mask3)
# plt.show()
# plt.plot([x_v1*-scale*2, x_v1*scale*2],
#          [y_v1*-scale*2, y_v1*scale*2], color='red')
# plt.plot([x_v2*-scale, x_v2*scale],
#          [y_v2*-scale, y_v2*scale], color='blue')
# plt.plot(x, y, 'k.')
# plt.axis('equal')
# plt.gca().invert_yaxis()  # Match the image system with origin at top left
# plt.show()
