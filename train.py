from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense

import csv
import pickle

# number of letters
units = 28

# initialize the CNN
classifier = Sequential();

# Step 1: convolution
classifier.add(Convolution2D(5, 5, input_shape=(50, 50, 1), padding='same', activation='relu'))

# Step 2: pooling
classifier.add(MaxPooling2D(pool_size=(4, 4)))

# Add a convolutional layer
classifier.add(Convolution2D(15, 5, input_shape=(50, 50, 1), padding='same', activation='relu'))

# Add another max pooling layer
classifier.add(MaxPooling2D(pool_size=(4, 4)))

# Step 3: Flattening
classifier.add(Flatten())

# Step 4: Full connection
# classifier.add(Dense(output_dim=128, activation='relu'))
# classifier.add(Dense(output_dim=1, activation='sigmoid'))
classifier.add(Dense(units=units, activation='softmax'))

classifier.summary()

# Compiling the CNN
classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print('Fitting the CNN to the images')
# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

print('Train set loaded')
train_set = train_datagen.flow_from_directory('C:/Users/Deeathex/PycharmProjects/TestSignLang1/TrainData',
                                              target_size=(50, 50),
                                              color_mode='grayscale',
                                              batch_size=32,
                                              class_mode='categorical')

print('Hot-encoding labels')
# Hot-encoding labels
labels = []
for i in range(0, units):
    labels.append(i)

from keras.utils import np_utils

# One-hot encode the training labels
y_train_OH = np_utils.to_categorical(labels)

# One-hot encode the test labels
y_test_OH = np_utils.to_categorical(labels)

# Training the network
history = classifier.fit_generator(
    train_set,
    steps_per_epoch=500,
    epochs=4,
    validation_data=train_set,
    validation_steps=10
)

import datetime

now = datetime.datetime.now()
date_index = now.strftime("%Y-%m-%d")

# write history dictionary in a file
print('Saving the history metrics in a file;')
w = csv.writer(open("history_metrics_" + date_index + ".csv", "w"))
for key, val in history.history.items():
    w.writerow([key, val])

with open('train_history_dict_' + date_index + '.txt', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print('Saving the model')
# Save the model and load it
classifier.save('model_saved_' + date_index + '.h5')
