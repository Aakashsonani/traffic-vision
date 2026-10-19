"""One-to-one object evaluation; abstentions never become true positives."""

import math
from collections import Counter, defaultdict

from traffic_vision.catalogue import catalogue
from traffic_vision.geometry import iou


def validate_dataset(data):
    if not isinstance(data.get("images"), list) or not data["images"]:
        raise ValueError("Dataset needs a non-empty images list")
    groups, paths = {}, set()
    known = {item["id"] for item in catalogue()}
    for image in data["images"]:
        for key in ("file", "group", "split", "objects"):
            if key not in image:
                raise ValueError(f"Image record missing {key}")
        if not image["group"] or image["split"] not in {"train", "val", "test"}:
            raise ValueError("Every image needs a recording/location group and train/val/test split")
        prior = groups.setdefault(image["group"], image["split"])
        if prior != image["split"]:
            raise ValueError(f"Data leakage: group {image['group']} appears in multiple splits")
        if image["file"] in paths:
            raise ValueError(f"Duplicate image path: {image['file']}")
        paths.add(image["file"])
        for obj in image["objects"]:
            label, box = obj.get("label"), obj.get("bbox", [])
            if label not in known and not valid_lamp_label(label):
                raise ValueError(f"Unsupported evaluation label: {label}")
            if (len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v)
                                         for v in box)
                    or min(box) < 0 or box[2] <= box[0] or box[3] <= box[1]):
                raise ValueError(f"Invalid bounding box for {label}: {box}")


def valid_lamp_label(label):
    if not isinstance(label, str):
        return False
    fields = label.split(":")
    return len(fields) == 3 and fields[0] == "light" and fields[1] in {
        "red", "yellow", "green"} and fields[2] in {
        "circle", "left", "right", "straight", "up_left", "up_right", "down", "down_left",
        "down_right", "u_turn", "pedestrian", "cross"}


class ObjectMetrics:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.counts = defaultdict(Counter)
        self.confusion = Counter()
        self.images = self.rejected = self.proposed = 0

    def add(self, truth, predictions):
        self.images += 1
        self.proposed += len(predictions)
        self.rejected += sum(pred.get("status") == "uncertain" for pred in predictions)
        accepted = sorted((pred for pred in predictions if pred.get("status") != "uncertain"),
                          key=lambda pred: pred["score"], reverse=True)
        matched, wrong_matched = set(), set()
        for obj in truth:
            self.counts[obj["label"]]["support"] += 1
        for pred in accepted:
            label = pred["id"]
            choices = [(iou(obj["bbox"], pred["bbox"]), index) for index, obj in enumerate(truth)
                       if index not in matched and obj["label"] == label]
            best_iou, best_index = max(choices, default=(0, -1))
            if best_iou >= self.iou_threshold:
                self.counts[label]["tp"] += 1
                matched.add(best_index)
                self.confusion[(label, label)] += 1
            else:
                self.counts[label]["fp"] += 1
                wrong = [(iou(obj["bbox"], pred["bbox"]), index) for index, obj in enumerate(truth)
                         if index not in matched and index not in wrong_matched
                         and obj["label"] != label]
                overlap, index = max(wrong, default=(0, -1))
                if overlap >= self.iou_threshold:
                    self.confusion[(truth[index]["label"], label)] += 1
                    wrong_matched.add(index)
                else:
                    self.confusion[("background", label)] += 1
        for index, obj in enumerate(truth):
            if index not in matched:
                self.counts[obj["label"]]["fn"] += 1
                if index not in wrong_matched:
                    self.confusion[(obj["label"], "missed")] += 1

    def report(self):
        def ratio(top, bottom):
            return top / bottom if bottom else None
        rows = []
        for label, counts in sorted(self.counts.items()):
            tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
            rows.append({"label": label, **{key: counts[key] for key in ("tp", "fp", "fn", "support")},
                         "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
                         "f1": ratio(2 * tp, 2 * tp + fp + fn)})
        tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
        precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
        return {"status": "evaluated", "images": self.images, "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": ratio(2 * tp, 2 * tp + fp + fn),
                "iou_threshold": self.iou_threshold, "rejected": self.rejected,
                "accepted_fraction": ratio(self.proposed - self.rejected, self.proposed),
                "per_class": rows, "targets": {"precision": .95, "recall": .90},
                "passed_targets": (precision or 0) >= .95 and (recall or 0) >= .90,
                "per_class_targets_met": bool(rows) and all(
                    (row["precision"] or 0) >= .95 and (row["recall"] or 0) >= .90
                    for row in rows if row["support"]),
                "classes_with_support": sum(row["support"] > 0 for row in rows),
                "confusion": [{"actual": a, "predicted": p, "count": count}
                              for (a, p), count in sorted(self.confusion.items())]}
