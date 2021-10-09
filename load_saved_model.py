

#To load the saved model

import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras_visualizer import visualizer #For visualizing the model
mnist = tf.keras.datasets.mnist
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

#https://towardsdatascience.com/visualizing-keras-models-4d0063c8805e

if __name__ == "__main__":

    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    #Loading the trained model
    new_model = tf.keras.models.load_model('saved_model')

    new_model.fit(training_images, training_labels, epochs =5)
    prediction=model.predict(test_images)
    #print(np.argmax(prediction[0])

    _, accuracy = new_model.evaluate(test_images,test_labels)
    print("%0.3f" % accuracy)


    #Print Confusion Matrix
    #matrix = confusion_matrix(y_test,y_pred)
    #print('Confusion matrix : \n',matrix)

    # Check its architecture
    print("Model Summary : ")
    new_model.summary()

    #To visualize the model
    #visualizer(new_model, format='png', view=True)
)


