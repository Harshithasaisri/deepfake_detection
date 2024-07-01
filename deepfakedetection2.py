# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:07:24 2024

@author: HARSHITHA
"""

import numpy as np
import os
import cv2
import dlib
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

# Define the custom layer
import tensorflow as tf

class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs, **kwargs):
        return inputs * self.scale

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({'scale': self.scale})
        return config

# Image data paths
a = r"C:\Users\HARSHITHA\Downloads\deepfakeds\realds"
b = r"C:\Users\HARSHITHA\Downloads\deepfakeds\fakeds"

train_frame_folder_a = os.listdir(a)
train_frame_folder_b = os.listdir(b)

list_of_train_data_a = [f for f in train_frame_folder_a if f.endswith('.mp4')]
list_of_train_data_b = [f for f in train_frame_folder_b if f.endswith('.mp4')]
detector = dlib.get_frontal_face_detector()

for vid in list_of_train_data_a:
    count = 0
    cap = cv2.VideoCapture(os.path.join(a, vid))
    print(vid)
    frameRate = cap.get(5)
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % ((int(frameRate)+1)*1) == 0:
            face_rects, scores, idx = detector.run(frame, 0)
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                crop_img = frame[y1:y2, x1:x2]
                cv2.imwrite(r"C:\Users\HARSHITHA\Downloads\deepfakeds\real/" + vid.split('.')[0] + '_' + str(count) + '.png', cv2.resize(crop_img, (128, 128)))
                count += 1
    cap.release()

for vid in list_of_train_data_b:
    count = 0
    cap = cv2.VideoCapture(os.path.join(b, vid))
    print(vid)
    frameRate = cap.get(5)
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % ((int(frameRate)+1)*1) == 0:
            face_rects, scores, idx = detector.run(frame, 0)
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                crop_img = frame[y1:y2, x1:x2]
                cv2.imwrite(r"C:\Users\HARSHITHA\Downloads\deepfakeds\fake/" + vid.split('.')[0] + '_' + str(count) + '.png', cv2.resize(crop_img, (128, 128)))
                count += 1
    cap.release()

input_shape = (128, 128, 3)
data_dir = r"C:\Users\HARSHITHA\Downloads\deepfakeds"
real_data = [f for f in os.listdir(data_dir+'/real') if f.endswith('.png')]
fake_data = [f for f in os.listdir(data_dir+'/fake') if f.endswith('.png')]

X = []
Y = []

for img in real_data:
    X.append(img_to_array(load_img(data_dir+'/real/'+img)).flatten() / 255.0)
    Y.append(1)
for img in fake_data:
    X.append(img_to_array(load_img(data_dir+'/fake/'+img)).flatten() / 255.0)
    Y.append(0)

# Normalization and reshaping
X = np.array(X)
Y = to_categorical(Y, 2)
X = X.reshape(-1, 128, 128, 3)

# Train-Test split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=5)

# Build and compile the model
googleNet_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
googleNet_model.trainable = True
model = Sequential()
model.add(googleNet_model)
model.add(GlobalAveragePooling2D())
model.add(CustomScaleLayer(scale=0.17))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=1e-5), metrics=['accuracy'])

model.summary()

# Train the model
EPOCHS = 3
BATCH_SIZE = 100
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, Y_val), verbose=1)

# Save the model
model.save(r"C:\Users\HARSHITHA\Downloads\deepfakeds\saved_deepfakemodel2.h5")
