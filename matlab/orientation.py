import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def get_bounding_rectangle(img):
	min_x = 0
	min_y = 0
	print img
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
	return (min_x+3, min_y+3, max_x-3, max_y-3)

img = cv.imread('../dataset/1075.png')
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

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
cv.grabCut(img,mask,new_rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
mask4 = img>10;
y, x = np.nonzero(mask4)
x = x - np.mean(x)
y = y - np.mean(y)
coords = np.vstack([x, y])
cov = np.cov(coords)
evals, evecs = np.linalg.eig(cov)
sort_indices = np.argsort(evals)[::-1]
evec1, evec2 = evecs[:, sort_indices]
x_v1, y_v1 = evec1  # Eigenvector with largest eigenvalue
x_v2, y_v2 = evec2
new_rect = get_bounding_rectangle(mask4)
scale = 20
plt.plot([x_v1*-scale*2, x_v1*scale*2],
         [y_v1*-scale*2, y_v1*scale*2], color='red')
plt.plot([x_v2*-scale, x_v2*scale],
         [y_v2*-scale, y_v2*scale], color='blue')
plt.plot(x, y, 'k.')
plt.axis('equal')
plt.gca().invert_yaxis()  # Match the image system with origin at top left
plt.show()
