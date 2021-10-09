# Import the MINST dataset

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras_visualizer import visualizer
from sklearn.metrics import confusion_matrix
mnist = tf.keras.datasets.mnist

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#print(training_images.shape)
#print(training_images[0])

#normalising the data to quicken training process

training_images = training_images/255
test_images = test_images/255

# Building the model with 3 layers
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)), #Input Layer
                                    tf.keras.layers.Dense(128, activation= 'relu'), #Hidden Layer
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)]) #Output Layer

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs =5)
prediction=model.predict(test_images)
print(np.argmax(prediction[0]))

# matrix = confusion_matrix(test_labels,prediction)
# print('Confusion matrix : \n',matrix)

_, accuracy = model.evaluate(test_images,test_labels)
print("%0.3f" % accuracy)

# visualizer(model, format='png', view=True)
print(model.summary())

model.save('saved_model')
