import numpy as np
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv
from PIL import Image

im = Image.open("balls_and_rects.png")
im = np.array(im)

im_hsv = rgb2hsv(im)
binary = im.mean(2)
binary[binary > 0] = 1

h = im_hsv[:, :, 0]

labeled = label(binary)
regions = regionprops(labeled)

colors = []
circles = []
rectangles = []

for region in regions:
    pixels = h[region.coords]
    r = h[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]]
    unique_colors = np.unique(r)
    colors.extend(unique_colors)
    if np.min(r) == 0.0:
        circles.append(r)
    else:
        rectangles.append(r)
clusters = []
while colors:
    color1 = colors.pop(0)
    clusters.append([color1])
    for color2 in colors.copy():
        if abs(color1 - color2) < 0.05:
            clusters[-1].append(color2)
            colors.pop(colors.index(color2))

shades = {}
for cluster in clusters:
    mean_color = int(np.mean(cluster) * 255)
    shades[mean_color] = [int(np.min(cluster) * 255) - 1, int(np.max(cluster) * 255) + 1]

print(f"Всего фигур на рисунке: {np.max(labeled)}")
print(f"Прямоугольники: {len(rectangles)}")
print(f"Круги: {len(circles)}")
print("shade (color) | rectangles(count) | circles(count)")
for shade in sorted(shades.keys()):
    sum_rect = sum(1 for rect in rectangles if shades[shade][0] < np.max(rect) * 255 < shades[shade][1])
    sum_circ = sum(1 for circ in circles if shades[shade][0] < np.max(circ) * 255 < shades[shade][1])
    print(f"{shade}                   {sum_rect}                    {sum_circ}")
