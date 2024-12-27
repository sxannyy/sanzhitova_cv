import cv2
import time

cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)

camera = cv2.VideoCapture(0)

# Разные границы из-за не насыщенного красного цвета
lower_red1 = (0, 100, 100)
upper_red1 = (10, 255, 255)
lower_red2 = (170, 100, 100)
upper_red2 = (180, 255, 255)

D = 0.09
prev_time = time.time()
curr_time = time.time()
r = 1

trajectory = []
speed_values = []

# Настройки камеры из-за засветов
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1)
camera.set(cv2.CAP_PROP_EXPOSURE, 80)

vid_write = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('balls.mp4', vid_write, 20.0, (640, 480))

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    curr_time = time.time()
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_count = len(contours)

    if ball_count > 0:
        c = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        trajectory.append((int(x), int(y)))
        if len(trajectory) > 10:
            trajectory.pop(0)
        if r > 10:
            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 255), -1)
            cv2.circle(frame, (int(x), int(y)), int(r), (255, 0, 255), 2)

        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i], trajectory[i-1], (120 * (i / 10), 0, 125), i)

        time_diff = curr_time - prev_time
        if len(trajectory) >= 2:
            p1 = trajectory[-1]
            p2 = trajectory[-2]
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            dist = (dx**2 + dy**2) ** 0.5
            pxl_per_m = D / (2 * r)
            dist *= pxl_per_m
            speed = dist / time_diff
            
            speed_values.append(speed)
            if len(speed_values) > 100:
                speed_values.pop(0)

            average_speed = sum(speed_values) / len(speed_values)
            
            cv2.putText(frame, f"Instant Speed: {speed:.3f} m/s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
            cv2.putText(frame, f"Avg Speed: {average_speed:.3f} m/s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
            cv2.putText(frame, f"Ball Count: {ball_count}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
            
            prev_time = curr_time

    out.write(frame)

    cv2.imshow("Mask", mask)
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

camera.release()
out.release()
cv2.destroyAllWindows()
