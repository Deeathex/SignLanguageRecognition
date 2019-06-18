import glob
import cv2
import os

image_list = []
image_names = []

k = 413
for filename in glob.glob('C:/Users/Deeathex/PycharmProjects/TestSignLang1/SavedImages/*.JPG'):
    original_image = cv2.imread(filename)
    image_list.append(original_image)
    image_name = filename.split("\\")
    new_name = image_name[0]+"/" + str(k) + ".JPG"
    print(new_name)
    k += 1
    image_names.append(new_name)
    # os.remove(filename)

i = 0
for original_image in image_list:
    cv2.imwrite(image_names[i], original_image)
    i = i + 1
