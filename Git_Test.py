
#C:\Users\niels\OneDrive\Documents\GitHub\Test-repo\Git_Test.py
#We are importing the nessary pakages

import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np



#minst is a biult in dataset in tensorflow that contains a 28 by 28 grayscale images of handwriten digits from (0 - 9)
#in .keras refers to a hige level API. it is used to train modles in a easy and intutive way. with tf.keras you not only get to use karas but all of the other methods that would ubrella it in tensor flow
#tf.keras allows you to easily add layers, specify activation functions, optimizers, loss functions with even more advance things like callbacks and check points
#the .datasets refers to a collection of prepackeged datasets that are commenly used for training

mnist = tf.keras.datasets.mnist


#.loaddata() i a method that loads the minist data set with both its  the training and test data as well as the corasponding lables
#they are loaded onto tuples witch are inmutable and cant be changed after cration
#The first tuple: (x_train, y_train) contains:
	#x_train: The training images (as a NumPy array).
	#y_train: The training labels (as a NumPy array).
#The second tuple: (x_test, y_test) contains:
	#x_test: The test images (as a NumPy array).
	#y_test: The test labels (as a NumPy array).

(x_train, y_train), (x_test, y_test) = mnist.load_data()



#ok so what the fuck is normalization
#normalization i  the prosses of of adjusting the scvale of data to a smaller,more manageble range. for contect its commen to normalize the pixle vals to a range between 1 and 0
#this is importent becuase ra  pixle data ranges from 0 to 255 and this can make it compitationaly diffuclt for the nural network to work efficently.
#axis = 1 normilize s the data across each row of pixles of each imege 
#.utils is a submodule in keras that contain usful functions in tasks. like normalize

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)



#Sequential is a modle a stack of layers that are arranged in a liner fashion. this works good for freeforward networks. if it was a more complex modle with multiple inputs you might want to use a more functional input instead
#.models is a submod of keras

model = tf.keras.models.Sequential()


#Flatten is a layer in keras that converts multi-dimentinal input data into a one dementional array

model.add(tf.keras.layers.Flatten())

#these are the layers inside the nural network.
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

#Adme is short for adaptive moment estimation is a common optimization algo
#sparse_categorical_crossentropy is a loss function used for multi-class classification problems where each sample belongs to one of several classes.
#accuracy is the metric to want to track 

model.compile(optimizer= 'adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 3)

val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_acc)
print(val_loss)

# Make predictions on the test set
predictions = model.predict(x_test)

# Print predictions for the first test image
print("Prediction for the first test image:", np.argmax(predictions[10]))

plt.imshow(x_test[10], cmap='binary')
plt.show()