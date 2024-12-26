import cv2
import numpy as np

video_path = 'output.avi'
image_path = 'monkey.png'

image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

cap = cv2.VideoCapture(video_path)

h, w = image.shape[:2]

frame_count = 0
image_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_frame, image, cv2.TM_CCOEFF_NORMED)
    threshold = 0.2
    loc = np.where(result >= threshold)

    if len(loc[0]) > 800 and len(loc[0]) < 1000:
        image_count += 1

        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Match: {len(loc[0])}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    frame_count += 1

cap.release()

print("Количество кадров с моим изображением: ", frame_count)