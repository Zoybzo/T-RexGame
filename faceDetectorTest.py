import cv2
from main import face_detector

cap = cv2.VideoCapture(0)
cv2.namedWindow('face Detection', cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    annotated_frame, coords, status, conf = face_detector(frame)
    cv2.imshow('face Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow('face Detection')