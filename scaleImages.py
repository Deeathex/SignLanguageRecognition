import glob
import cv2

image_list = []
image_names = []

alphabet = ['N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for letter in alphabet:
    for filename in glob.glob('C:/Users/Deeathex/PycharmProjects/TestSignLang1/TrainData/'+letter+'/*.JPG'):
        original_image = cv2.imread(filename)
        image_list.append(original_image)
        # image_name = filename.split("\\")
        # image_names.append(image_name[1])
        image_names.append(filename)

    i = 0
    for original_image in image_list:
        new_image = cv2.resize(original_image, dsize=(50, 50))
        cv2.imwrite(image_names[i], new_image)
        i = i + 1
