from tensorflow.python.keras.models import load_model

# Testing
import numpy as np
from tensorflow.python.keras.preprocessing import image

from tensorflow.python.keras.utils.vis_utils import plot_model

import pickle
import matplotlib.pyplot as plot

# import the trained model
classifier = load_model('model_saved_2019-06-18.h5')

test_imageA = image.load_img('TestData/A/0.jpg', grayscale=True)
test_imageA = image.img_to_array(test_imageA)
test_imageA = np.expand_dims(test_imageA, axis=0)
resultA = classifier.predict(test_imageA)
print("A")
print(resultA)
# train_set.class_indices
# if resultA[0][0] > 0.5:
#     prediction = 'A'
# else:
#     prediction = 'B'
# print(prediction)

test_imageB = image.load_img('TestData/B/16.jpg', grayscale=True)
test_imageB = image.img_to_array(test_imageB)
test_imageB = np.expand_dims(test_imageB, axis=0)
resultB = classifier.predict(test_imageB)
print("B")
print(resultB)

test_imageC = image.load_img('TestData/C/20.jpg', grayscale=True)
test_imageC = image.img_to_array(test_imageC)
test_imageC = np.expand_dims(test_imageC, axis=0)
resultC = classifier.predict(test_imageC)
print("C")
print(resultC)

test_imageC = image.load_img('TestData/R/12.jpg', grayscale=True)
test_imageC = image.img_to_array(test_imageC)
test_imageC = np.expand_dims(test_imageC, axis=0)
resultC = classifier.predict(test_imageC)
print("R")
print(resultC)

test_imageC = image.load_img('TestData/V/2.jpg', grayscale=True)
test_imageC = image.img_to_array(test_imageC)
test_imageC = np.expand_dims(test_imageC, axis=0)
resultC = classifier.predict(test_imageC)
print("V")
print(resultC)

test_imageC = image.load_img('TestData/W/7.jpg', grayscale=True)
test_imageC = image.img_to_array(test_imageC)
test_imageC = np.expand_dims(test_imageC, axis=0)
resultC = classifier.predict(test_imageC)
print("W")
print(resultC)

classifier.summary()
plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# load the history from a file in order to plot the metrics
history = pickle.load(open("train_history_dict_2019-06-17.txt", "rb"))
# list all data in history
print(history.keys())
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# summarize history for accuracy
plot.plot(history['acc'])
plot.plot(history['val_acc'])
plot.title('model accuracy')
plot.ylabel('accuracy')
plot.xlabel('epoch')
plot.legend(['train', 'test'], loc='upper left')
plot.show()
# summarize history for loss
plot.plot(history['loss'])
plot.plot(history['val_loss'])
plot.title('model loss')
plot.ylabel('loss')
plot.xlabel('epoch')
plot.legend(['train', 'test'], loc='upper left')
plot.show()
