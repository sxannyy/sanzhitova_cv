import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion

def check(B, y, x):
    if not 0 <= x < B.shape[0]:
        return False
    if not 0 <= y < B.shape[1]:
        return False
    if B[y, x] != 0:
        return True
    return False

def neighbors2(B, y, x):
    left = y, x-1
    top = y - 1, x
    if not check(B, *left):
        left = None
    if not check(B, *top):
        top = None
    return left, top

def exists(neighbors):
    return not all([n is None for n in neighbors])

def find(label, linked):
    j = label
    while linked[j] != 0:
        j = linked[j]
    return j

def union(label1, label2, linked):
    j = find(label1, linked)
    k = find(label2, linked)
    if j != k:
        linked[k] = j

def two_pass_labeling(B):
    labels = np.zeros_like(B)
    linked = np.zeros(256, dtype="uint32")
    label = 1

    for row in range(B.shape[0]):
        for col in range(B.shape[1]):
            if B[row, col] != 0:
                n = neighbors2(B, row, col)
                if not exists(n):
                    m = label
                    label += 1
                else:
                    lbs = [labels[i] for i in n if i is not None]
                    m = min(lbs)
                labels[row, col] = m
                for i in n:
                    if i is not None:
                        lb = labels[i]
                        if lb != m:
                            union(m, lb, linked)

    for row in range(B.shape[0]):
        for col in range(B.shape[1]):
            if B[row, col] != 0:
                new_label = find(labels[row, col], linked)
                if new_label != labels[row, col]:
                    labels[row, col] = new_label

    uniq_lb = np.unique(labels)
    for i in range(len(uniq_lb)):
        labels[labels == uniq_lb[i]] = i
    return labels

struct_plus = np.array([[0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0],
                         [1, 1, 1, 1, 1],
                         [0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0]])

struct_cross = np.array([[1, 0, 0, 0, 1],
                         [0, 1, 0, 1, 0],
                         [0, 0, 1, 0, 0],
                         [0, 1, 0, 1, 0],
                         [1, 0, 0, 0, 1]])

image = np.load("stars.npy")

image_without_plus = binary_erosion(image, struct_plus).astype("uint8")
image_without_cross = binary_erosion(image, struct_cross).astype("uint8")

count_plus = two_pass_labeling(image_without_plus)
count_cross = two_pass_labeling(image_without_cross)

num_objects_plus = len(np.unique(count_plus)) - 1
num_objects_cross = len(np.unique(count_cross)) - 1

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title(f'Количество объектов с плюсом: {num_objects_plus}')
plt.imshow(count_plus)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Количество объектов с крестом: {num_objects_cross}')
plt.imshow(count_cross)
plt.axis('off')

plt.show()