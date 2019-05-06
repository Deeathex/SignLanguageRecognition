from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initialize the CNN
classifier = Sequential();

# Step 1: convolution
classifier.add(Convolution2D(5, 5, input_shape=(50, 50, 3), padding='same', activation='relu'))

# Step 2: pooling
classifier.add(MaxPooling2D(pool_size=(4, 4)))

# Add a convolutional layer
classifier.add(Convolution2D(15, 5, input_shape=(50, 50, 3), padding='same', activation='relu'))

# Add another max pooling layer
classifier.add(MaxPooling2D(pool_size=(4, 4)))

# Step 3: Flattening
classifier.add(Flatten())

# Step 4: Full connection
# classifier.add(Dense(output_dim=128, activation='relu'))
# classifier.add(Dense(output_dim=1, activation='sigmoid'))
classifier.add(Dense(3, activation='softmax'))

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
                                              batch_size=32,
                                              class_mode='categorical')

print('Hot-encoding labels')
# Hot-encoding labels
labels = [0, 1, 2]

from keras.utils import np_utils

# One-hot encode the training labels
y_train_OH = np_utils.to_categorical(labels)

# One-hot encode the test labels
y_test_OH = np_utils.to_categorical(labels)

# Training the network
classifier.fit_generator(
    train_set,
    steps_per_epoch=100,
    epochs=4,
    validation_data=train_set,
    validation_steps=10
)

print('Fitting the model;')
# hist = classifier.fit(x=train_set, y=y_train_OH, batch_size=32, validation_split=0.2, epochs=4)

print('Saving the model')
# Save the model and load it
classifier.save('modelSavedAB.h5')
