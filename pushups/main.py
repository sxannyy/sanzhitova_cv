from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from pathlib import Path
import numpy as np
import time
import cv2

MODEL_FILENAME = "yolo11n-pose.pt"
OUTPUT_VIDEO = "result.mp4"
ANGLE_THRESHOLD_LOW = 80
ANGLE_THRESHOLD_HIGH = 140
RESET_TIME_SECONDS = 90

def calculate_distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_angle(a, b, c):
    angle1 = np.arctan2(c[1] - b[1], c[0] - b[0])
    angle2 = np.arctan2(a[1] - b[1], a[0] - b[0])
    angle_deg = np.degrees(angle1 - angle2)
    angle_deg = (angle_deg + 360) % 360
    return 360 - angle_deg

def analyze_pose(image, keypoints):
    try:
        left_ear_seen = keypoints[3][0] > 0 and keypoints[3][1] > 0
        right_ear_seen = keypoints[4][0] > 0 and keypoints[4][1] > 0

        left = (keypoints[5], keypoints[7], keypoints[9], keypoints[11])
        right = (keypoints[6], keypoints[8], keypoints[10], keypoints[12])

        if left_ear_seen and not right_ear_seen:
            shoulder, elbow, wrist, hip = left
        else:
            shoulder, elbow, wrist, hip = right

        arm_angle = calculate_angle(shoulder, elbow, wrist)
        body_angle = calculate_angle(shoulder, hip, (shoulder[0], hip[1]))

        x, y = int(elbow[0]) + 10, int(elbow[1]) + 10
        cv2.putText(image, f"angle:{arm_angle:.1f}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 255, 25), 1)

        if not (20 < body_angle < 330):
            arm_angle = 0

        return arm_angle
    except Exception:
        return 0

model_path = Path(__file__).parent / MODEL_FILENAME
model = YOLO(model_path)

cam = cv2.VideoCapture("http://192.168.0.106/video") #подключаюсь через приложение ip-webcam
cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)

writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 400))
count = 0
is_sitting = False
last_time = time.time()

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    results = model(frame)
    if not results:
        continue

    result = results[0]
    keypoints_data = result.keypoints.xy.tolist()
    if not keypoints_data or not keypoints_data[0]:
        continue

    keypoints = keypoints_data[0]
    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated = annotator.result()

    angle_value = analyze_pose(annotated, keypoints)
    current_time = time.time()

    if angle_value > 0:
        if not is_sitting and angle_value <= ANGLE_THRESHOLD_LOW:
            count += 1
            is_sitting = True
            last_time = current_time
        elif is_sitting and angle_value > ANGLE_THRESHOLD_HIGH:
            is_sitting = False
            last_time = current_time

    if (current_time - last_time) > RESET_TIME_SECONDS:
        count = 0

    cv2.putText(annotated, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    resized_frame = cv2.resize(annotated, (640, 400))
    writer.write(resized_frame)

    cv2.imshow("YOLO", resized_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
writer.release()
cv2.destroyAllWindows()
