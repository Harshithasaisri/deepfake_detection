# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:14:19 2024

@author: HARSHITHA
"""

import tkinter as tk
from tkinter import filedialog, StringVar, Label, Button
import dlib
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf

# Define the custom layer
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

# Register the custom layer
get_custom_objects().update({'CustomScaleLayer': CustomScaleLayer})

class DeepFakeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepFake Detection")
        self.root.geometry("400x200")
        
        self.var = StringVar()
        self.var.set("Upload a video to detect")
        
        self.label = Label(root, textvariable=self.var)
        self.label.pack(pady=20)
        
        self.upload_button = Button(root, text="Upload Video", command=self.upload_file)
        self.upload_button.pack(pady=20)
        
        self.model_path = r"C:\Users\HARSHITHA\Downloads\deepfakeds\deepfake_detectionfrontend\saved_deepfakemodel.h5"
        self.model = self.load_model(self.model_path)
        self.detector = dlib.get_frontal_face_detector()
        
    def load_model(self, model_path):
        try:
            with tf.keras.utils.custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
                model = load_model(model_path)
            return model
        except Exception as e:
            self.var.set(f"Error loading model: {e}")
            raise
        
    def upload_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.var.set("Processing...")
            self.root.update()
            result = self.detect_deepfake(file_path)
            self.var.set(f"Result: {result}")
        
    def detect_deepfake(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frameRate = cap.get(cv2.CAP_PROP_FPS)
        dic = {}
        
        while cap.isOpened():
            frameId = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
            if not ret:
                break
            if frameId % int(frameRate) == 0:
                face_rects, _, _ = self.detector.run(frame, 0)
                for d in face_rects:
                    x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
                    crop_img = frame[y1:y2, x1:x2]
                    data = img_to_array(cv2.resize(crop_img, (128, 128))).flatten() / 255.0
                    data = data.reshape(-1, 128, 128, 3)
                    predict_x = self.model.predict(data)
                    classes_x = np.argmax(predict_x, axis=1)
                    dic[classes_x[0]] = dic.get(classes_x[0], 0
