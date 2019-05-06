import numpy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# import the trained model
classifier = load_model('modelSavedAB.h5')

test_datagen = ImageDataGenerator(rescale=1. / 255)

print('Test set loaded')
test_set = test_datagen.flow_from_directory('C:/Users/Deeathex/PycharmProjects/TestSignLang1/TestData',
                                            target_size=(50, 50),
                                            batch_size=32,
                                            class_mode='categorical')

# evaluate_generator to compare predictions vs known outputs

# score = classifier.evaluate_generator(
#     test_set,
#     steps=1200,
#     verbose=1
# )
#
# print(classifier.metrics_names)
# print('Test accuracy:\n', score)

predictions = classifier.predict_generator(
    test_set,
    steps=1200,
    verbose=1
)


