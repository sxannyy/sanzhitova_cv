import time
import cv2
import numpy as np
import pyautogui
from mss import mss

pyautogui.PAUSE = 0.001

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    return thresh

def detect_obstacle(thresh, danger_zone):
    danger_area = thresh[danger_zone[1]:danger_zone[3], danger_zone[0]:danger_zone[2]]
    contours, _ = cv2.findContours(danger_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def binarize(img):
    return img[:, :, 2] > 83

def get_game_img(sct, game_x, game_y, game_width, game_height):
    screenshot = np.array(sct.grab({"top": game_y, "left": game_x, "width": game_width, "height": game_height}))
    return binarize(screenshot)

def get_jump_delay_ratio(game_window, dz_mid):
    min_ratio = 0.4
    obstacle = (game_window[40:100, dz_mid:dz_mid+100] == 0).sum(axis=0)
    indices = np.where(obstacle > 0)[0]
    if not indices.size:
        return min_ratio
    obstacle_width = np.max(indices) - np.min(indices)
    result = obstacle_width / 35
    if result < min_ratio:
        return min_ratio
    return result

def main():
    game_window = {
        "top": 300,   # верхняя координата
        "left": 500,  # левая координата
        "width": 700, # ширина
        "height": 150 # высота
    }

    danger_zone = (135, 40, 200, 100)  # x1, y1, x2, y2
    dz_mid = (500 + 700) // 2

    game_x, game_y, game_width, game_height = game_window["left"], game_window["top"], game_window["width"], game_window["height"]

    pressed = False

    with mss() as sct:
        game = get_game_img(sct, game_x, game_y, game_width, game_height)
        print("game started")

        while True:
            game = get_game_img(sct, game_x, game_y, game_width, game_height)
            dangerous_zone = game[danger_zone[1]:danger_zone[3], danger_zone[0]:danger_zone[2]]

            if not np.all(dangerous_zone):
                if not pressed:
                    if np.all(dangerous_zone[-35:]):
                        pyautogui.keyDown("down")
                        time.sleep(0.3)
                        pyautogui.keyUp("down")
                    else:
                        jump_delay_ratio = get_jump_delay_ratio(game, dz_mid)
                        time.sleep(0.023 * jump_delay_ratio)
                        pyautogui.press('up')
                        time.sleep(0.225)
                        pyautogui.press("down")

                    pressed = True
            else:
                pressed = False

            screenshot = np.array(sct.grab(game_window))
            cv2.rectangle(screenshot, (danger_zone[0], danger_zone[1]), (danger_zone[2], danger_zone[3]), (0, 255, 0), 2)
            cv2.imshow("Game", cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    main()
