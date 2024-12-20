import zmq
import cv2
import numpy as np

flimit = 100
slimit = 80

def fupdate(value):
    global flimit
    flimit = value

def supdate(value):
    global slimit
    slimit = value

background = {
    "lower": (0, 0, 0),
    "upper": (200, 90, 200),
}

def on_mouse_callback(event, x, y, *params):
    global position
    if event == cv2.EVENT_LBUTTONDOWN:
        position = [y, x]

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
port = 5555
socket.connect(f"tcp://192.168.0.100:{port}")

window_name = "client"
cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback(window_name, on_mouse_callback)

cv2.createTrackbar("F", window_name, flimit, 255, fupdate)
cv2.createTrackbar("S", window_name, slimit, 255, supdate)

while True:
    msg = socket.recv()
    frame = cv2.imdecode(np.frombuffer(msg, np.uint8), -1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, background["lower"], background["upper"])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # mask = cv2.medianBlur(mask, 5)

    result_masked = cv2.bitwise_and(mask, mask, mask=gray)

    ret, thresh2 = cv2.threshold(result_masked, 200, 255, cv2.THRESH_BINARY_INV)

    edges = cv2.Canny(thresh2, flimit, slimit)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cubes = 0
    sphere = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4250:
            cubes += 1
        elif area > 9400:
            sphere += 1

    cv2.drawContours(frame, contours, -1, (255, 255, 255), 3)
    cv2.putText(frame, f'Objects: {cubes} cubes and {sphere} spheres', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    key = cv2.waitKey(100)
    if key == ord('q'):
        break

cv2.imshow(window_name, edges)
cv2.destroyAllWindows()