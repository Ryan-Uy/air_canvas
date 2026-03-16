import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_OPTIONS = python.BaseOptions(model_asset_path = "hand_landmarker.task")

OPTIONS = vision.HandLandmarkerOptions(base_options = BASE_OPTIONS,
                                       running_mode = vision.RunningMode.VIDEO,
                                       num_hands = 1, 
                                       min_hand_detection_confidence = 0.6,
                                       min_hand_presence_confidence = 0.6,
                                       min_tracking_confidence = 0.6)

MODEL = vision.HandLandmarker.create_from_options(OPTIONS)

HAND_CONNECTIONS = [(0,1), (1,2), (2,3), (3,4), #thumb
                    (5,6), (6,7), (7,8), #index
                    (9,10), (10,11), (11,12), #middle
                    (13,14), (14,15), (15,16), #ring
                    (17,18),(18,19), (19,20), #pinky
                    (0,5), (5,9), (9,13), (13,17), (17,0) #palm
                    ]

COLORS = {(0,0,0) : 'Black',
          (255,255,255) : 'White',
          (0,0,255) : 'Red',
          (0,165,255) : 'Orange',
          (0,255,255) : 'Yellow',
          (0,255,0) : 'Green',
          (255,0,0) : 'Blue',
          (128,0,128) : 'Purple'}


def main():
    stream = cv2.VideoCapture(0)
    timestamp = 0

    active_color = (0,0,255)
    mode = "Select"

    last_points = []

    while True:
        ret, frame = stream.read()

        if not ret:
            print("Error: No valid frame found")
            break
    
        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb)
        result = MODEL.detect_for_video(mp_image, timestamp)
        timestamp += 1

        points = get_points(result, [], frame)

        mode = check_mode(points)

        if mode == 'Select':
            active_color = choose_color(points) 
        else:
            draw(points, last_points)

        draw_skeleton(points, frame, active_color)

        last_points = points     

        cv2.putText(frame, f"Mode: {mode}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        cv2.putText(frame, F"Current color: ", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        cv2.putText(frame, COLORS[active_color], (180,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, active_color, 2)


        cv2.imshow("Air Canvas", frame)

        if cv2.waitKey(1) == 27:
            break

    stream.release()
    cv2.destroyAllWindows()

def check_mode(points):
    return 'Select' #temporary

def draw(points, last_points):
    pass

def choose_color(points):
    fingertips = [(8,7), (12,11), (16,15), (20,19)]
    finger_count = 0
    if points:
        #loops for fingers excluding thumb
        for tip, joint in fingertips: 
            if points[tip][1] < points[joint][1]:
                finger_count += 1
        #checks thumb angle between 3,2,1
        p1 = np.array(points[3])
        p2 = np.array(points[2])
        p3 = np.array(points[1])
        v1 = p1 - p2
        v2 = p3 - p2
        angle = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

        if angle > 3*np.pi/4:
            finger_count += 1

    colors = {0:(0,0,255), #red 
              1:(0,165,255), #orange
              2:(0,255,255), #yellow
              3:(0,255,0), #green
              4:(255,0,0), #blue
              5:(128,0,128) #purple
              }
    
    return colors[finger_count]
    


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
