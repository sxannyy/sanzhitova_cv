import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from scipy.ndimage import binary_erosion

def cut_image(B, struct):
    eroded_image = binary_erosion(B, struct)
    return eroded_image

struct1 =  [[0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]] #прямоугольник

struct2 =  [[0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]] #рога кверху

struct3 =  [[1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]] #рога книзу

struct4 =  [[1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0]] #рога вправо
 
struct5 =  [[0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1]] #рога влево

image = np.load("ps.npy")
labeled_all_objects = label(image)

structs = [struct1, struct2, struct3, struct4, struct5]
count_objects = np.max(labeled_all_objects)

for i in range(5):
    eroded_img = cut_image(image, structs[i])
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title(f"Всего объектов: {np.unique(labeled_all_objects).max()}")
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    if i == 1 or i == 2:
        count_parts = np.max(label(eroded_img)) - np.max(label(binary_erosion(image, struct1)))
    else:
        count_parts = np.max(label(eroded_img))
    plt.title(f"Всего объектов определенной структуры: {count_parts}")
    plt.imshow(eroded_img)

plt.show()
