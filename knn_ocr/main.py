import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
import matplotlib.pyplot as plt


def normalize_label(label: str) -> str:
    return label[1:] if label.startswith("s") and len(label) > 1 else label


def load_dataset(train_dir, img_size=(20, 20)):
    X, y = [], []
    for root, _, files in os.walk(train_dir):
        folder = os.path.basename(root)
        if folder == os.path.basename(train_dir):
            continue

        lbl = normalize_label(folder)

        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                img = cv2.imread(os.path.join(root, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                bin_img = cv2.resize(bin_img, img_size, cv2.INTER_AREA)

                feat = hog(
                    bin_img,
                    orientations=9,
                    pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys",
                )
                X.append(feat)
                y.append(lbl)
    return np.asarray(X, np.float32), np.asarray(y)


def fit_knn(X, y, k=3):
    model = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    model.fit(X, y)
    return model


def binarize(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    purple = cv2.inRange(hsv, (110, 40, 40), (170, 255, 255))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    local = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 31, 15)

    mask = cv2.bitwise_or(local, purple)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return mask


def split_by_lines(bboxes, gap=30):
    if not bboxes:
        return []
    bboxes.sort(key=lambda b: (b[1], b[0]))

    lines, cur = [], [bboxes[0]]
    for box in bboxes[1:]:
        if abs(box[1] - cur[-1][1]) <= gap:
            cur.append(box)
        else:
            lines.append(sorted(cur, key=lambda b: b[0]))
            cur = [box]
    lines.append(sorted(cur, key=lambda b: b[0]))
    return lines


def ocr_image(path, knn, img_size=(20, 20), debug=False):
    img = cv2.imread(path)
    if img is None:
        print(f"Не удалось открыть {path}")
        return ""

    bin_img = binarize(img)
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cnts if cv2.boundingRect(c)[2] > 2 and cv2.boundingRect(c)[3] > 2]

    if debug:
        vis = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        plt.imshow(vis[..., ::-1]); plt.axis("off"); plt.show()

    text_lines = []
    for line in split_by_lines(boxes):
        avg_gap = np.mean([line[i][0] - (line[i - 1][0] + line[i - 1][2])
                           for i in range(1, len(line))]) if len(line) > 1 else 0
        line_txt = ""
        for i, (x, y, w, h) in enumerate(line):
            if i and avg_gap and x - (line[i - 1][0] + line[i - 1][2]) > 1.8 * avg_gap:
                line_txt += " "

            roi = cv2.resize(bin_img[y:y + h, x:x + w], img_size, cv2.INTER_AREA)
            feat = hog(roi, orientations=9, pixels_per_cell=(4, 4),
                       cells_per_block=(2, 2), block_norm="L2-Hys")
            line_txt += knn.predict(feat.reshape(1, -1))[0]
        text_lines.append(line_txt)

    return "\n".join(text_lines)


def run(train_dir="./task/train", img_dir="./task", debug=False):
    X, y = load_dataset(train_dir)
    knn = fit_knn(X, y)

    out = {}
    for fname in sorted(os.listdir(img_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) and "train" not in fname:
            txt = ocr_image(os.path.join(img_dir, fname), knn, debug=debug)
            print(fname, "→\n", txt, "\n", "-" * 40)
            out[fname] = txt
    return out


results = run(debug=True)
