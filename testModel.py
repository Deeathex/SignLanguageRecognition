from keras.models import load_model

# import the trained model
classifier = load_model('modelSavedAB.h5')

# Testing
import numpy as np
from keras.preprocessing import image

test_imageA = image.load_img('TestData/A/1000.jpg')
test_imageA = image.img_to_array(test_imageA)
test_imageA = np.expand_dims(test_imageA, axis=0)
resultA = classifier.predict(test_imageA)
print(resultA)
# train_set.class_indices
# if resultA[0][0] > 0.5:
#     prediction = 'A'
# else:
#     prediction = 'B'
# print(prediction)

test_imageB = image.load_img('TestData/B/1000.jpg')
test_imageB = image.img_to_array(test_imageB)
test_imageB = np.expand_dims(test_imageB, axis=0)
resultB = classifier.predict(test_imageB)
print(resultB)
# train_set.class_indices
# if resultB[0][0] > 0.5:
#     prediction = 'A'
# else:
#     prediction = 'B'
# print(prediction)

# classifier.summary()
