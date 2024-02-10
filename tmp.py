import os
import cv2
import numpy as np


# path = './imgs'
# directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
# imgs = {}
# for directory in directories:
#     img_path = os.path.join(path, directory)
#     file_list = []
#     for root, dirs, files in os.walk(img_path):
#         for file in files:
#             img_path = os.path.join(root, file)
#             file_list.append(img_path)
#     imgs[directory] = file_list
# print(imgs)

# img = cv2.imread('./imgs\MJ\MJ0.jpg')
# print(type(img))
# print(type(img) is np.ndarray)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image = cv2.imread('./imgs\DSW\DSW1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
print(faces)
input('----------------')
face_image = image
for (x, y, w, h) in faces:
    face_image = image[y:y + h, x:x + w]
cv2.imshow('Faces detected', face_image)
cv2.waitKey(0)



