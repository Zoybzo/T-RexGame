import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import hypot
import pyautogui
import dlib


# DNN Module
model_weights = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
model_arch = "model/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(model_arch, model_weights)


# detect the face with the highest confidence in image.
def face_detector(image, threshold=0.7):
    # Get the height and width of the image
    h, w = image.shape[:2]
    # Create a 4D blob as input from image cv2.dnn.blobFromImage(image, scalefactor(scaling ratio), size(resize image),
    # mean(reduce the impact of light), swapRB, crop, ddepth)
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    # Run forward pass to compute output
    faces = net.forward()
    # Get the confidence of each face
    prediction_scores = faces[:, :, :, 2]
    # Get the face with the highest confidence
    i = np.argmax(prediction_scores)
    face = faces[0, 0, i]
    confidence = face[2]
    if confidence > threshold:
        box = face[3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        annotated_frame = cv2.rectangle(image.copy(), (x1, y1), (x2, y2), (0, 0, 255), 2)
        output = (annotated_frame, (x1, y1, x2, y2), True, confidence)
    else:
        output = (image, (), False, 0)

    return output


def detect_landmarks(box, image):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
