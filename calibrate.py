import cv2
import main

cap = cv2.VideoCapture(0)
cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    face_image, box_coords, status, conf = main.face_detector(frame)
    if status:
        landmark_image, landmarks = main.detect_landmarks(box_coords, face_image)

        _, mouth_ar = main.is_mouth_open(landmarks)
        _, proximity = main.face_proximity(box_coords, face_image)

        # Get the threshold
        ar_threshold = mouth_ar*1.4
        proximity_threshold = proximity*1.3

        cv2.putText(frame, 'Aspect ratio threshold: {:.2f}'.format(ar_threshold), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 127, 255), 2)
        cv2.putText(frame, 'Proximity threshold: {:.2f}'.format(proximity_threshold), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 127, 255), 2)
    cv2.imshow('Calibration', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('Calibration')