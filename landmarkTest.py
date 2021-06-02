import cv2

import main


cap = cv2.VideoCapture(0)
cv2.namedWindow('landmark Detection', cv2.WINDOW_NORMAL)
while True:
    # TODO: What is the return type?
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    face_image, box_coords, status, conf = main.face_detector(frame)
    if status:
        landmark_image, landmarks = main.detect_landmarks(box_coords, face_image)
    cv2.imshow('Landmark Detection', landmark_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('landmark Detection')