import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
import os

def get_centroid(labeled, label=1):
  pos = np.where(labeled == label)
  return (int)(pos[0].mean()), (int)(pos[1].mean())

def add_to_circle(pos, circ):
    xl, yl = get_centroid(labeled, pos)
    circ[0].append(xl)
    circ[1].append(yl)

circ1 = [[], []]
circ2 = [[], []]
circ3 = [[], []]
circles = [circ1, circ2, circ3]

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "out", "h_0.npy")

image = np.load(image_path)

labeled = label(image)
for i, circle in enumerate(circles, start=1):
  add_to_circle(i, circle)

for i in range(1, 100):
    image_path = os.path.join(script_dir, "out", f"h_{i}.npy")
    if os.path.exists(image_path):
        image = np.load(image_path)
    labeled = label(image)
    for lbl in range(1, 4):
        pos = np.where(labeled == lbl)
        last_positions = {i: (None, None) for i in range(3)}
        for x, y in zip(*pos):
            for idx, circ in enumerate(circles):
                last_x, last_y = last_positions[idx]
                if last_x == x and last_y == y:
                    add_to_circle(lbl, circ)
            for idx in range(3):
                if circles[idx][0]:
                    last_positions[idx] = (circles[idx][0][-1], circles[idx][1][-1])

for circ in circles:
    plt.plot(circ[0], circ[1])
plt.title("Траектории движения объектов")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()