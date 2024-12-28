import zmq
import cv2
import numpy as np
import time
import random
import pymunk
cv2.namedWindow("Client recv", cv2.WINDOW_GUI_NORMAL)
capture = cv2.VideoCapture('rtsp://12.71.17.199:8080/h264.sdp')


def drawsegments(s,segments,contours):
    for seg in segments:
        s.remove(seg[0],seg[1])
    segments=[]
    for ct in range(len(contours)):
        epsilon = 0.03 * cv2.arcLength(contours[ct], True)
        approximations = cv2.approxPolyDP(contours[ct], epsilon, True)
        for i in range(len(approximations) - 1):
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Segment(body, (approximations[i][0][0],approximations[i][0][1]), (approximations[i+1][0][0],approximations[i+1][0][1]), 5)  # Ширина сегмента = 2
            segments.append([shape,body])
            shape.elasticity = 1
            s.add(body, shape)
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Segment(body, (approximations[-1][0][0],approximations[-1][0][1]), (approximations[0][0][0],approximations[0][0][1]), 5)  # Ширина сегмента = 2
        segments.append([shape,body])
        shape.elasticity = 1
        s.add(body, shape)
    return segments
ret, frame = capture.read()
global b
s=pymunk.Space((frame.shape[0],frame.shape[1]))
b=pymunk.Body(1,100)
c=pymunk.Circle(b,5)
s.add(b,c)
b.position=(100,100)
c.elqasticity = 0.5
c.friction = 0.5
s.gravity=0,50
segments=[]
lsttm=time.time()-10
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Проверка на левую кнопку мыши
        b.position=(x,y)
        b.velocity = (0, 0)


cv2.setMouseCallback("Client recv", mouse_callback)
#image_path = 'ph1.jpg'
lower_red1 = np.array([0, 0, 0])  
upper_red1 = np.array([255, 40, 90])

#lower_red1 = np.array([0, 0, 0])  
#upper_red1 = np.array([255, 40, 90])

while True:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bb=cv2.inRange(hsv, (60,20,20), (90,255,255))
    #bb=cv2.inRange(hsv, (30,0,0), (80,20,110))
    bb=cv2.dilate(bb,(10,10),iterations=10)
    _, thresh = cv2.threshold(bb, 1, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = thresh
    #cv2.imshow("Client recv", closed)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = frame.copy()
    cropped_area = None
    warped_cropped_area = None
    maxcnt=contours[0]
    for cnt in contours:
        if cv2.contourArea(cnt)>cv2.contourArea(maxcnt):
            maxcnt=cnt
    x,y,w,h = cv2.boundingRect(maxcnt)
    warped_cropped_area=frame[y:y+h,x:x+w,:]
    hsv=cv2.cvtColor(warped_cropped_area, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red1, upper_red1)
    mask=cv2.morphologyEx(mask, cv2.MORPH_ERODE, (15,15))
    mask=cv2.dilate(mask,(10,10),iterations=15)
    contours,_=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #здесь обрабатывается frame и на выходе получается результат findContours в переменной contours как показано выше 
    
    if time.time()-lsttm>3:
        segments=drawsegments(s,segments,contours)
        lsttm=time.time()
    s.step(0.1)
    empta=np.zeros_like(warped_cropped_area)
    cd=int(list(b.position)[0]),int(list(b.position)[1])
    cv2.circle(empta, cd, 2, (255,255,255), 10)
    
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    cv2.imshow("Client recv",empta)
cv2.destroyAllWindows()