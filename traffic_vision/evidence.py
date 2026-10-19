"""Conservative geometric proposals and visible lamp evidence.

These checks supplement trained models; they do not infer unfamiliar sign meanings.
All thresholds operate on image geometry/colour, not specific input filenames or boxes.
"""

import cv2
import numpy as np

from traffic_vision.geometry import iou, nms


def crop_box(image, box, margin=.05):
    x1, y1, x2, y2 = box
    dx, dy = margin * (x2 - x1), margin * (y2 - y1)
    x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    x2, y2 = min(image.shape[1], int(np.ceil(x2 + dx))), min(image.shape[0], int(np.ceil(y2 + dy)))
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def overlaps_housing(box, housings):
    for housing in housings:
        other = housing["bbox"]
        overlap = max(0, min(box[2], other[2]) - max(box[0], other[0])) * max(
            0, min(box[3], other[3]) - max(box[1], other[1]))
        smaller = min((box[2] - box[0]) * (box[3] - box[1]),
                      (other[2] - other[0]) * (other[3] - other[1]))
        if smaller > 0 and overlap / smaller > .8:
            return True
    return False


def sign_regions(image, housings):
    """Propose compact coloured sign faces missed by the learned detectors."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    masks = {
        "red": cv2.inRange(hsv, (0, 90, 65), (12, 255, 255))
        | cv2.inRange(hsv, (168, 90, 65), (179, 255, 255)),
        "blue": cv2.inRange(hsv, (95, 90, 55), (130, 255, 255)),
        "yellow": cv2.inRange(hsv, (17, 100, 100), (38, 255, 255)),
    }
    height, width = image.shape[:2]
    proposals = []
    for color, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) < 16 or not .65 < w / h < 1.5 or w * h > height * width * .35:
                continue
            area, perimeter = cv2.contourArea(contour), cv2.arcLength(contour, True)
            if area < max(250, height * width * .0025) or perimeter == 0:
                continue
            polygon = cv2.approxPolyDP(contour, .04 * perimeter, True)
            circularity = 4 * np.pi * area / perimeter ** 2
            if circularity < .65 and not (3 <= len(polygon) <= 5 and cv2.isContourConvex(polygon)):
                continue
            if color in {"red", "blue"} and (circularity < .75 or not .8 < w / h < 1.25):
                if color != "red" or len(polygon) != 3:
                    continue
            box = [x, y, x + w, y + h]
            if overlaps_housing(box, housings):
                continue
            face = hsv[y:y + h, x:x + w]
            if color == "red" and (((face[:, :, 1] < 70) & (face[:, :, 2] > 150)).mean() < .12):
                continue
            proposals.append({"id": "sign_region", "score": float(area / (w * h)),
                              "bbox": box, "face_color": color})
    return nms(proposals)


def no_entry_bar(image):
    """Require the characteristic wide white bar across a red disk."""
    if image.size == 0:
        return False
    hsv = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_RGB2HSV)
    red = ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168)) & (hsv[:, :, 1] > 100)
    white = (hsv[:, :, 1] < 65) & (hsv[:, :, 2] > 175)
    return (white[32:48, 15:65].mean() > .65
            and red[15:29, 24:56].mean() > .65
            and red[51:65, 24:56].mean() > .65)


def exposure_reference(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturated = hsv[:, :, 2][hsv[:, :, 1] >= 50]
    return float(np.percentile(saturated, 98)) if saturated.size else 0.0


def visible_lamp_color(image, *, reference_brightness=None, whole_housing=False):
    """Read a bright, saturated lamp core; a dark tinted lens is not active evidence."""
    if not image.size:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, w = image.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    core = ((xx - w / 2) / max(1, w * .4)) ** 2 + ((yy - h / 2) / max(1, h * .4)) ** 2 < 1
    if whole_housing:
        core = np.ones((h, w), dtype=bool)
    reference = exposure_reference(image) if reference_brightness is None else reference_brightness
    brightness_threshold = max(45.0, reference * .6)
    bright = core & (hsv[:, :, 2] >= brightness_threshold) & (hsv[:, :, 1] >= 50)
    minimum = max(3, core.sum() * (.005 if whole_housing else .06))
    if bright.sum() < minimum:
        return None
    hue = hsv[:, :, 0][bright]
    counts = {"red": int(((hue <= 12) | (hue >= 168)).sum()),
              "yellow": int(((hue > 12) & (hue <= 38)).sum()),
              "green": int(((hue > 38) & (hue < 95)).sum())}
    color = max(counts, key=counts.get)
    agreement = counts[color] / len(hue)
    return {"color": color, "agreement": agreement,
            "brightness_threshold": brightness_threshold} if agreement >= .7 else None


def neutral_sign(item, reason):
    result = dict(item)
    result.update(id="unknown_sign", label="Unrecognised sign", status="uncertain",
                  category="Unverified", proposed_category=item.get("proposed_category", item["category"]),
                  proposed_id=item.get("proposed_id", item["id"]),
                  proposed_label=item.get("proposed_label", item["label"]),
                  meaning=reason)
    return result


def same_region(box, candidates):
    return any(iou(box, item["bbox"]) > .4 for item in candidates)
