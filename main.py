import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import hypot
import pyautogui
import dlib

MOUTH = 0.5
PROXIMITY = 260

# DNN Module
model_weights = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
model_arch = "model/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(model_arch, model_weights)

predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")


# detect the face with the highest confidence in image.
def face_detector(image, threshold=0.7):
    # Get the height and width of the image
    h, w = image.shape[:2]
    # Create a 4D blob as input from image cv2.dnn.blobFromImage(image, scalefactor(scaling ratio), size(resize image),
    # mean(reduce the impact of light), swapRB, crop, ddepth)
    # because scalefactor is 1.0, so the coordinates of the detection scaled down to 0-1 range
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    # Run forward pass to compute output
    # TODO: What is the output?
    faces = net.forward()
    # Get the confidence of each face
    prediction_scores = faces[:, :, :, 2]
    # Get the face with the highest confidence
    i = np.argmax(prediction_scores)
    face = faces[0, 0, i]
    confidence = face[2]
    # if confidence is greater than the threshold
    if confidence > threshold:
        # Get the top-left point and bottom-right point
        # enlarge the coordinates
        box = face[3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        # Add the box and produce a new frame
        annotated_frame = cv2.rectangle(image.copy(), (x1, y1), (x2, y2), (0, 0, 255), 2)
        # return the new frame or the original frame
        output = (annotated_frame, (x1, y1, x2, y2), True, confidence)
    else:
        output = (image, (), False, 0)

    return output


# Add marks to the face
def detect_landmarks(box, image):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get the coordinates
    (x1, y1, x2, y2) = box
    # Add marks
    shape = predictor(gray_scale, dlib.rectangle(x1, y1, x2, y2))
    landmarks = shape_to_np(shape)
    for (x, y) in landmarks:
        annotated_image = cv2.circle(image, (x, y), 2, (0, 127, 255), -1)

    return annotated_image, landmarks


def shape_to_np(shape):
    landmarks = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        landmarks[i] = (shape.part(i).x, shape.part(i).y)

    return landmarks


def is_mouth_open(landmarks, ar_threshold=MOUTH):
    A = hypot(landmarks[50][0] - landmarks[58][0], landmarks[50][1] - landmarks[58][1])
    B = hypot(landmarks[52][0] - landmarks[56][0], landmarks[52][1] - landmarks[56][1])
    C = hypot(landmarks[48][0] - landmarks[54][0], landmarks[48][1] - landmarks[54][1])

    # the average of distance A, B : distance C
    mouth_aspect_ratio = (A + B) / (2.0 * C)
    if mouth_aspect_ratio > ar_threshold:
        return True, mouth_aspect_ratio
    else:
        return False, mouth_aspect_ratio


def face_proximity(box, image, proximity_threshold=PROXIMITY):
    # Get the width and height of the face
    face_w = box[2] - box[0]
    face_h = box[3] - box[1]
    # Get theta
    theta = np.arctan(face_h / face_w)
    # Get the guide width and height by theta, ensure that the two triangles are similar
    # proximity_threshold is the diagonal distance of the guide
    guide_w = np.cos(theta) * proximity_threshold
    guide_h = np.sin(theta) * proximity_threshold
    # Get the center of the face for determining the position of the guide
    mid_x, mid_y = (box[2] + box[0]) / 2, (box[3] + box[1]) / 2
    guide_topleft = int(mid_x - (guide_w / 2)), int(mid_y - (guide_h / 2))
    guide_bottomeright = int(mid_x + guide_w / 2), int(mid_y + (guide_h / 2))
    cv2.rectangle(image, guide_topleft, guide_bottomeright, (0, 255, 255), 2)
    # Get the diagonal distance of the face
    diagonal = hypot(face_w, face_h)
    if diagonal > proximity_threshold:
        return True, diagonal
    else:
        return False, diagonal


def __init__():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Dino!', cv2.WINDOW_NORMAL)
    pyautogui.PAUSE = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_image, box_coords, status, conf = face_detector(frame)
        if status:
            landmark_image, landmarks = detect_landmarks(box_coords, face_image)
            mouthOpen, _ = is_mouth_open(landmarks)
            faceProximity, _ = face_proximity(box_coords, face_image)

            if mouthOpen:
                pyautogui.keyDown('space')
                mouth_status = 'Open'
            else:
                pyautogui.keyUp('space')
                mouth_status = 'Close'
            # if mouthOpen:
            #     pyautogui.press('space')
            #     mouth_status = 'Open'
            # else:
            #     mouth_status = 'Close'

            if faceProximity:
                pyautogui.keyDown('down')
                face_status = 'close'
            else:
                pyautogui.keyUp('down')
                face_status = 'far'

            cv2.putText(frame, 'Mouth: {}'.format(mouth_status), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 127, 255), 2)
            cv2.putText(frame, 'Face: {}'.format(face_status), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 127, 255),
                        2)
        cv2.imshow('Dino!', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyWindow('Dino!')


__init__()
