import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion
def check(B, y, x):
    if not 0 <= x < B.shape[1]:
        return False
    if not 0 <= y < B.shape[0]:
        return False
    return B[y, x] != 0

def neighbors2(B, y, x):
    left = (y, x-1)
    top = (y-1, x)
    return (left if check(B, *left) else None,
            top if check(B, *top) else None)

def exists(neighbors):
    return any(n is not None for n in neighbors)

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

def cut_image(B, struct):
    eroded_image = binary_erosion(B, struct)
    return eroded_image

struct = np.ones((3, 1))
image = np.load("wires6.npy").astype("uint8")

labeled_image = two_pass_labeling(image)

uniq_lb_first = np.unique(labeled_image)

for i in uniq_lb_first:
    if i == 0:
        continue
    cut_labeled_image = cut_image(labeled_image == i, struct)

    labeled_after_cutting = two_pass_labeling(cut_labeled_image.astype("int"))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title(f"Всего проводов в начале: {np.unique(labeled_image).max()}")
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    count_parts = np.unique(labeled_after_cutting).max()
    if count_parts > 0:
        plt.title(f"Всего частей: {count_parts}")
    else:
        plt.title(f"Этот провод не исправен")
    plt.imshow(labeled_after_cutting)

plt.axis("off")
plt.show()