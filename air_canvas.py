import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path = "hand_landmarker.task")

options = vision.HandLandmarkerOptions(base_options = base_options,
                                       running_mode = vision.RunningMode.VIDEO,
                                       num_hands = 1, 
                                       min_hand_detection_confidence = 0.6,
                                       min_hand_presence_confidence = 0.6,
                                       min_tracking_confidence = 0.6)

model = vision.HandLandmarker.create_from_options(options)

stream = cv2.VideoCapture(0)

while True:
    ret, frame = stream.read()

    if not ret:
        print("Error: No valid frame found")
        break
    
    frame = cv2.flip(frame,1)
    cv2.imshow("Air Canvas", frame)

    if cv2.waitKey(1) == 27:
        break

stream.release()
cv2.destroyAllWindows()

