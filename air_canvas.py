import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOpetions(model_asset_path = "hand_landmarker.task")

options = vision.HandLandmarkerOptions(base_options = base_options,
                                       running_mode = vision.RunningMode.VIDEO,
                                       num_hands = 1, 
                                       min_hand_detection_confidence = 0.6,
                                       min_hand_presence_confidence = 0.6,
                                       min_tracking_confidence = 0.6)

model = vision.HandLadmarker.create_from_options(options)