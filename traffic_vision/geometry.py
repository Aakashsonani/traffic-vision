"""Image geometry shared by inference and evaluation."""

import cv2
import numpy as np


def letterbox(image, size):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    left, top = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = cv2.resize(image, (new_w, new_h))
    # Return actual scales after integer resize: no coordinate drift on narrow images.
    return canvas, (new_w / width, new_h / height), (left, top)


def restore_boxes(boxes, scale, padding, width, height):
    boxes = boxes.astype(np.float32).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padding[0]) / scale[0]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padding[1]) / scale[1]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return boxes


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def nms(candidates, threshold=0.45, class_agnostic=False):
    kept = []
    for item in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if any((class_agnostic or item["id"] == prior["id"])
               and iou(item["bbox"], prior["bbox"]) > threshold for prior in kept):
            continue
        kept.append(item)
        if len(kept) >= 100:
            break
    return kept
