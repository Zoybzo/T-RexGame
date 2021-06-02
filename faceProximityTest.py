import cv2

import main

cap = cv2.VideoCapture(0)
cv2.namedWindow('face Proximity', cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    face_image, box_coords, status, conf = main.face_detector(frame)
    if status:
        is_face_approach, _ = main.face_proximity(box_coords, face_image, proximity_threshold=main.PROXIMITY)
        cv2.putText(face_image, 'Is Face Approach: {}'.format(is_face_approach), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 127, 255), 2)

    cv2.imshow('face Proximity', face_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('face Proximity')