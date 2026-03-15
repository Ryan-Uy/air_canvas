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

HAND_CONNECTIONS = [(0,1), (1,2), (2,3), (3,4), #thumb
                    (5,6), (6,7), (7,8), #index
                    (9,10), (10,11), (11,12), #middle
                    (13,14), (14,15), (15,16), #ring
                    (17,18),(18,19), (19,20), #pinky
                    (0,5), (5,9), (9,13), (13,17), (17,0) #palm
                    ]

def main():
    stream = cv2.VideoCapture(0)
    timestamp = 0

    active_color = (0,0,255)

    while True:
        ret, frame = stream.read()

        if not ret:
            print("Error: No valid frame found")
            break
    
        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb)
        result = model.detect_for_video(mp_image, timestamp)
        timestamp += 1

        points = get_points(result, [], frame)

        draw_skeleton(points, frame, active_color)

        cv2.imshow("Air Canvas", frame)

        if cv2.waitKey(1) == 27:
            break

    stream.release()
    cv2.destroyAllWindows()

def get_points(result, points, frame):
    h,w,_ = frame.shape
    if result.hand_landmarks:
        for landmark in result.hand_landmarks[0]:
            x,y = int(landmark.x*w), int(landmark.y*h)
            points.append((x,y))
    return points

def draw_skeleton(points, frame, active_color):
    if points:
        for start,end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0,255,0), 2)
        for point in points:
            cv2.circle(frame, point, 4, active_color, -1)

if __name__ == "__main__":
    main()
