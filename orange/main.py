import cv2
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from skimage import draw

path = Path(__file__).parent
model_path = path / "facial_best.pt"

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -5)
cap.set(cv2.CAP_PROP_EXPOSURE, 10)

while cap.isOpened():

    ret, image = cap.read()

    # image = cv2.imread(str(path / "my_photo.png"))
    oranges = cv2.imread(str(path / "oranges.png"))
    hsv_oranges = cv2.cvtColor(oranges, cv2.COLOR_BGR2HSV)
    gray = cv2.GaussianBlur(hsv_oranges, (15, 15), 0)

    lower = np.array((11, 240, 200))
    upper = np.array((15, 255, 255))

    mask_orange = cv2.inRange(hsv_oranges, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=cv2.contourArea)

    m = cv2.moments(sorted_contours[-1])
    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])

    bbox = cv2.boundingRect(sorted_contours[-1])

    model = YOLO(model_path)
    result = model(image)[0]

    masks = result.masks

    annotated = result.plot()

    if not masks:
        continue

    global_mask = masks[0].data.cpu().numpy()[0, :, :]

    for mask in masks:
        global_mask += mask.data.cpu().numpy()[0, :, :]

    rr, cc = draw.disk((2, 2), 2)
    struct = np.zeros((4, 4), np.uint8)
    struct[rr, cc] = 1

    global_mask = cv2.resize(global_mask, (image.shape[1], image.shape[0])).astype("uint8")
    global_mask = cv2.GaussianBlur(global_mask, (5, 5), 0)
    global_mask = cv2.dilate(global_mask, struct)
    global_mask = global_mask.reshape(image.shape[0], image.shape[1], 1)

    parts = (image * global_mask).astype("uint8")

    pos = np.where(global_mask > 0)
    min_y, max_y = int(np.min(pos[0]) * 0.9), int(np.max(pos[0]) * 1.1)
    min_x, max_x = int(np.min(pos[1]) * 0.9), int(np.max(pos[1]) * 1.1)
    global_mask = global_mask[min_y:max_y, min_x:max_x]
    parts = parts[min_y:max_y, min_x:max_x]

    resized_parts = cv2.resize(parts, (bbox[2], bbox[3]))
    resized_mask = cv2.resize(global_mask, (bbox[2], bbox[3])) * 255

    x, y, w, h = bbox
    roi = oranges[y:y+h, x:x+w]
    bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(resized_mask))
    combined = cv2.add(bg, resized_parts)
    oranges[y:y+h, x:x+w] = combined

    cv2.imshow("Image", combined)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()