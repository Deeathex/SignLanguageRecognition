import glob
import cv2
import os


# def flip_images():
#     letter = "Z"
#     k = 1261
#     for filename in glob.glob('C:/Users/Deeathex/PycharmProjects/TestSignLang1/TrainData/' + letter + '/*.JPG'):
#         img = cv2.imread(filename)
#         vertical_img = cv2.flip(img, 1)
#         cv2.imwrite('C:/Users/Deeathex/PycharmProjects/TestSignLang1/TrainData/' + letter + '/' + str(k) + '.JPG',
#                     vertical_img)
#         k += 1


def rename_images():
    k = 2038
    for filename in glob.glob('C:/Users/Deeathex/PycharmProjects/TestSignLang1/SavedImages/*.JPG'):
        original_image = cv2.imread(filename)
        cv2.imwrite('C:/Users/Deeathex/PycharmProjects/TestSignLang1/SavedImages/' + str(k) + '.JPG', original_image)
        os.remove(filename)
        k += 1


