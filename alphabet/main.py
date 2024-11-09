import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops, euler_number
from pathlib import Path

def recognize(region):
    if region.image.mean() == 1.0:
        return "-"
    else:
        enumber = euler_number(region.image, 2)
        if enumber == -1: #B OR 8
            have_vl = np.sum(np.mean(region.image[:, :region.image.shape[1] // 2], 0) == 1) > 3
            if have_vl:
                return "B"
            else:
                return "8"
        elif enumber == 0: #P OR D
            image = region.image.copy()
            if 1 in region.image.mean(0)[:2]:
                image[len(image[:, 0])//2, :] = 1
                image_labeled = label(image)
                image_regions = regionprops(image_labeled)
                if image_regions[0].euler_number == -1:
                    return "D"
                elif image_regions[0].euler_number == 0:
                    return "P"
            image[-1, :] = 1
            enumber = euler_number(image)
            if enumber == -1: #A OR 0
                return "A"
            else:
                return "0"
        else: # 1 OR X OR W OR / OR *
            if 1 in region.image.mean(0):   
                return "1"
            image = region.image.copy()
            image[[0, -1], :] = 1
            image_labeled = label(image)
            image_regions = regionprops(image_labeled)
            euler = image_regions[0].euler_number 
            if euler == -1:
                return "X"
            elif euler == -2:
                return "W"
            if region.eccentricity > 0.5:
                return "/"
            else:
                return "*"  
    return "@"

image = plt.imread("symbols.png")[:, :, :3].mean(2)
image[image > 0] = 1

labeled_all_objects = label(image)

regions = regionprops(labeled_all_objects)

result = {}

for region in regions:
    symbol = recognize(region)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1

print("Частотный словарь: ", result)
print(f'Всего символов на картинке: {labeled_all_objects.max()}')
# plt.imshow(labeled_all_objects)
# plt.show()