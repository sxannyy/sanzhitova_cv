from ultralytics import YOLO
from pathlib import Path
import cv2
import time

path = Path(__file__).parent
model_path = path / "best.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1)
cap.set(cv2.CAP_PROP_EXPOSURE, 100)

state  = "idle"
prev_time = 0
curr_time = 0
player1_hand = ""
player2_hand = ""
game_result = ""
timer = 3

while cap.isOpened():
    ret, frame = cap.read()
    cv2.putText(frame, f"{state} - {3 - timer:.1f} - {game_result}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)
    results = model(frame)
    result = results[0]

    if not results:
        continue

    if len(result.boxes.xyxy) == 2:
        labels = []
        for label, xyxy in zip(result.boxes.cls, result.boxes.xyxy):
            x1, y1, x2, y2 = xyxy.cpu().numpy().astype("int")
            print(result.boxes.cls)
            print(result.names)
            labels.append(result.names[label.item()].lower())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, f"{labels[-1]}", (x1+20, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        player1_hand, player2_hand = labels
        if player1_hand == "rock" and player2_hand == "rock" and state == "idle":
            state = "wait"
            prev_time = time.time()
    if state == "wait":
        timer = round(time.time() - prev_time, 1)
    if timer >= 3:
        timer = 3
        if state == "wait":
            state = "result"
            if player1_hand == player2_hand:
                game_result = "draw"
            elif player1_hand == "scissors":
                if player2_hand =="rock":
                    game_result = "player 2 win"
                else:
                    game_result = "player 1 win"
            elif player1_hand == "rock":
                if player2_hand == "paper":
                    game_result = "player 2 win"
                else:
                    game_result = "player 1 win"
            elif player1_hand == "paper":
                if player2_hand == "scissors":
                    game_result = "player 2 win"
                else:
                    game_result = "player 1 win"

    cv2.imshow("YOLO", frame)
    key = cv2.waitKey(1)

    if key == ord("r"):
        state = "idle"
        prev_time = 0
        player1_hand = ""
        player2_hand = ""
        game_result = ""
        timer = 3

    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
