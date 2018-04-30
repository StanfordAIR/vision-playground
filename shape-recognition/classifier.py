import os
import keras.models as mdl
import json
import cv2
import numpy as np
from keras.layers import Dense
from keras.layers import Conv2D, Flatten
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import np_utils


def init_architecture(num_layers, num_neurons):
	model = mdl.Sequential()
	model.add(Dense(units=num_neurons[1], activation='relu', input_shape=(1,48,48)))
	for i in range(num_layers - 1):
		model.add(Dense(units=num_neurons[i], activation='relu'))
	return model


#This method should return a set of training data, the images and the labels.
def get_data():

	data = json.load(open('labels.json'))

	s = (5000,48,48)
	count = 0
	trainImages = np.zeros(s)
	trainLabels = []
	testImages = np.zeros(s)
	testLabels = []


	for filename in os.listdir('train'):
		if "png" in filename:
			trainImages[count] = np.asarray(cv2.imread('train/'+filename, cv2.IMREAD_GRAYSCALE))
			count+=1
			trainLabels.append(shapeToInt(data[filename]['Shape']))

	count = 0
	for filename in os.listdir('test'):
		if "png" in filename:
			testImages[count] = np.asarray(cv2.imread('test/'+filename, cv2.IMREAD_GRAYSCALE))
			count+=1
			testLabels.append(shapeToInt(data[filename]['Shape']))


	trainImages = trainImages.reshape(trainImages.shape[0], 1, 48, 48)
	testImages = testImages.reshape(testImages.shape[0], 1, 48, 48)
	trainLabels = np.array(trainLabels)
	testLabels = np.asarray(testLabels)
	trainLabels = np_utils.to_categorical(trainLabels, 9)
	testLabels = np_utils.to_categorical(testLabels, 9)

	return trainImages, trainLabels, testImages, testLabels


def shapeToInt(shape):
	if shape == "Circle":
		return 0
	elif shape == "SemiCircle":
		return 1
	elif shape == "Rectangle":
		return 2
	elif shape == "Trapezoid":
		return 3
	elif shape == "Pentagon" : 
		return 4
	elif shape == "Hexagon":
		return 5
	elif shape == "Heptagon":
		return 6
	elif shape == "Octagon":
		return 7
	elif shape == "Star":
		return 8


def main():
	#Paremeters; Fisne tune these as needed.
	num_layers = 2
	num_neurons = [64, 32]
	num_classes = 32
	num_epochs = 10
	batch_size = 32

	trainImages, trainLabels, testImages, testLabels = get_data()

	model = init_architecture(num_layers, num_neurons)
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(9, activation='softmax'))

	#Classification layer. Outputs confidences. Model must be
	#trained using one-hot encoded ground truth values.
	#model.add(Dense(units=num_classes, activation='relu'))
	model.compile(loss=categorical_crossentropy,
				  optimizer=Adam(lr=0.001), metrics=['accuracy'])

	#Train.
	model.fit(trainImages, trainLabels, batch_size=batch_size, epochs=num_epochs)

	#Test
	print "############################"
	accuracy = model.evaluate(testImages, testLabels, batch_size=batch_size, verbose=1)
	print(accuracy)



if __name__ == "__main__":
    main()