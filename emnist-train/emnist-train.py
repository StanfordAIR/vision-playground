import numpy as np
import pandas as pd
import random
import csv

import keras as K

train_db = pd.read_csv("data/emnist-balanced-train.csv")
test_db  = pd.read_csv("data/emnist-balanced-test.csv")

charInd = random.randint(0,112000)
emnist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

num_classes = 47
y_train = train_db.iloc[:,0]
y_train = K.utils.np_utils.to_categorical(y_train, num_classes)

x_train = train_db.iloc[:,1:]
x_train = x_train.astype('float32')
x_train /= 255

alphanum = np.where(y_train[charInd]==1.)[0][0]
print(emnist[alphanum])

inp = K.layers.Input(shape=(784,))
hidden_1 = K.layers.Dense(1024, activation='relu')(inp)
dropout_1 = K.layers.Dropout(0.2)(hidden_1)
# hidden_2 = K.layers.Dense(1024, activation='relu')(dropout_1)
# dropout_2 = K.layers.Dropout(0.2)(hidden_2)
out = K.layers.Dense(num_classes, activation='softmax')(hidden_1) # change to hidden_2 with second layer 
model = K.models.Model(outputs=out, inputs=inp)

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model.fit(x_train, y_train, # Train the model using the training set...
          batch_size=512, epochs=1,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation

y_test = test_db.iloc[:,0]
y_test = K.utils.np_utils.to_categorical(y_test, num_classes)
print ("y_test:", y_test.shape)

x_test = test_db.iloc[:,1:]
x_test = x_test.astype('float32')
x_test /= 255
print ("x_test:",x_train.shape)

print(model.evaluate(x_test, y_test, verbose=1)) # Evaluate the trained model on the test set

let = np.array([x_train.iloc[charInd,:]])

prediction = model.predict(let)
print(prediction)