import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from collections import defaultdict
from pathlib import Path

def add_bottom_line(region_image):
    image_with_line = region_image.copy()
    image_with_line[-1, :] = 1
    return image_with_line

def is_A(region):
    image_with_line = add_bottom_line(region.image)
    labeled_image = label(image_with_line)
    new_region = regionprops(labeled_image)[0]
    return new_region.euler_number == -1

def is_W(region):
    h, w = region.image.shape
    center_area = region.image[h // 4 : -h // 4, w // 4 : -w // 4]
    gaps = np.sum(center_area == 0)
    return gaps > 5

def extractor(region):
    area = np.sum(region.image) / region.image.size
    perimeter = region.perimeter / region.image.size
    cy, cx = region.local_centroid
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    euler = region.euler_number
    eccentricity = region.eccentricity
    have_vl = np.sum(np.mean(region.image, 0) == 1) > 3
    return np.array([area, perimeter, cy, cx, euler, eccentricity, have_vl])

def classificator(region, classes):
    det_class = None
    v = extractor(region)
    min_d = 10 ** 10
    for cls in classes:
        d = distance(v, classes[cls])
        if d < min_d:
            det_class = cls
            min_d = d
    
    if det_class == "8" or det_class == "B":
        left_strip = region.image[:, :2]
        if np.any(left_strip == 0):
            det_class = "8"
        else:
            det_class = "B"
    
    if det_class == "A" or det_class == "0":
        if is_A(region):
            det_class = "A"
        else:
            det_class = "0"

    if det_class == "*" or det_class == "W":
        if is_W(region):
            det_class = "W"
        else:
            det_class = "*"
    return det_class

def distance(v1, v2):
    return ((v1 - v2) ** 2).sum() ** 0.5

image = plt.imread("alphabet.png")[:, :, :3].mean(2)
image[image > 0] = 1

labeled_all_objects = label(image)
count_objects = np.max(labeled_all_objects)

template = plt.imread("alphabet-small.png")[:, :, :3].mean(2)
template[template < 1] = 0

template = np.logical_not(template)
labeled_all_template_objects = label(template)
count_template_objects = np.max(labeled_all_template_objects)

regions = regionprops(labeled_all_template_objects)

classes = { "8": extractor(regions[0]),
            "0": extractor(regions[1]),
            "A": extractor(regions[2]),
            "B": extractor(regions[3]),
            "1": extractor(regions[4]),
            "W": extractor(regions[5]),
            "X": extractor(regions[6]),
            "*": extractor(regions[7]),
            "/": extractor(regions[8]),
            "-": extractor(regions[9]), }

# print(count_objects)
# print(count_template_objects)

# plt.subplot(121)
# plt.imshow(labeled_all_objects)
# plt.axis('off')

# plt.subplot(122)
# plt.imshow(labeled_all_template_objects)
# plt.axis('off')

# plt.figure() #вывод картинок с символами

# for i, cls in enumerate(classes):
#     plt.subplot(2, 5, i+1)
#     plt.title(f'{cls} - {regions[i].label}')
#     plt.imshow(regions[i].image)
#     plt.axis('off')

symbols = defaultdict(lambda: 0)
path = Path("images")
path.mkdir(exist_ok=True)
plt.figure()
for i, region in enumerate(regionprops(labeled_all_objects)):
    print(i)
    symbol = classificator(region, classes)
    symbols[symbol] += 1
    plt.cla()
    plt.title(f"Symbol: {symbol}")
    plt.imshow(region.image)
    plt.savefig(path / f"image_{i:03d}.png")
    
print(symbols)
plt.show()