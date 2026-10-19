"""CPU inference. Runtime loads only ONNX graphs, never pickled checkpoints."""

import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from traffic_vision.catalogue import GTSRB_LABELS, LIGHT_LABELS, catalogue, sign_info
from traffic_vision.geometry import iou, letterbox, nms, restore_boxes

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("TRAFFIC_VISION_MODELS", str(ROOT / "models")))


class ModelsUnavailable(RuntimeError):
    pass


def arrow_direction(cosine, sine):
    if cosine * cosine + sine * sine < 1e-12:
        return "unknown"
    # Clockwise from up, following TIER IV's output convention.
    directions = ["straight", "up_right", "right", "down_right", "down", "down_left",
                  "left", "up_left"]
    angle = math.atan2(sine, cosine)
    index = math.floor((angle + math.pi / 8) / (math.pi / 4)) % 8
    return directions[index]


def decode_yolo(output, scale, padding, width, height, labels, threshold, model_name,
                selected=None):
    predictions = output[0].T  # YOLOv8: 1 x (4 + classes) x anchors; no objectness column.
    probabilities = predictions[:, 4:]
    if probabilities.shape[1] != len(labels):
        raise ValueError("Detector class count differs from its manifest")
    if selected is None:
        classes = probabilities.argmax(axis=1)
        scores = probabilities.max(axis=1)
    else:
        local = probabilities[:, selected]
        best = local.argmax(axis=1)
        classes = np.asarray(selected)[best]
        scores = local.max(axis=1)
    valid = np.isfinite(scores) & (scores >= threshold)
    rows, scores, classes = predictions[valid], scores[valid], classes[valid]
    if not len(rows):
        return []
    boxes = np.column_stack([rows[:, 0] - rows[:, 2] / 2, rows[:, 1] - rows[:, 3] / 2,
                             rows[:, 0] + rows[:, 2] / 2, rows[:, 1] + rows[:, 3] / 2])
    boxes = restore_boxes(boxes, scale, padding, width, height)
    found = []
    for box, score, class_id in zip(boxes, scores, classes):
        if not np.isfinite(box).all() or box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        raw = labels[int(class_id)]
        if raw in LIGHT_LABELS or model_name == "housing":
            info = {"id": "housing", "label": "Traffic light", "kind": "light",
                    "category": "Signal", "meaning": "Signal housing; state not read yet."}
        else:
            info = sign_info(raw)
            info["kind"] = "sign"
        found.append({**info, "bbox": box.round(2).tolist(), "score": float(score),
                      "model": model_name, "raw_label": raw, "status": "detected"})
    return nms(found)


def decode_lamps(output, width, height, input_width, input_height, threshold=0.55):
    """Decode TIER IV's 3-anchor / 16-channel head.

    Python port of the equations in Autoware cnn_lamp_recognizer.cpp (Apache-2.0,
    Copyright 2026 TIER IV, Inc.). See THIRD_PARTY_NOTICES.md for attribution.
    The graph already applies the activations; do not apply another sigmoid.
    """
    values = output[0]
    if values.shape[0] != 48:
        raise ValueError(f"Unexpected lamp head: {values.shape}")
    grid_h, grid_w = values.shape[1:]
    anchors = [7.0, 14.0, 42.0]
    colors = ["green", "yellow", "red"]
    shapes = ["circle", "arrow", "u_turn", "pedestrian", "number", "cross"]
    candidates = []
    for a, anchor in enumerate(anchors):
        head = values[a * 16:(a + 1) * 16]
        shape_ids, color_ids = head[8:14].argmax(axis=0), head[5:8].argmax(axis=0)
        shape_scores, color_scores = head[8:14].max(axis=0), head[5:8].max(axis=0)
        scores = head[4] * shape_scores
        for y, x in zip(*np.nonzero(np.isfinite(scores) & (scores >= threshold))):
            cx = (x + 2 * head[0, y, x] - 0.5) / grid_w
            cy = (y + 2 * head[1, y, x] - 0.5) / grid_h
            half_w = anchor * (2 * head[2, y, x]) ** 2 / input_width / 2
            half_h = anchor * (2 * head[3, y, x]) ** 2 / input_height / 2
            box = [max(0, cx - half_w) * width, max(0, cy - half_h) * height,
                   min(1, cx + half_w) * width, min(1, cy + half_h) * height]
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                continue
            color, shape = colors[int(color_ids[y, x])], shapes[int(shape_ids[y, x])]
            if color_scores[y, x] < 0.65:
                color = "unknown"
            if shape == "arrow":
                shape = arrow_direction(float(head[14, y, x]), float(head[15, y, x]))
            uncertain = color == "unknown" or shape in {"unknown", "number"}
            shape_label = {"circle": "circular light", "straight": "straight arrow",
                           "left": "left arrow", "right": "right arrow",
                           "u_turn": "U-turn arrow", "pedestrian": "pedestrian signal",
                           "up_left": "diagonal left arrow", "up_right": "diagonal right arrow",
                           "down": "down arrow", "down_left": "diagonal down-left arrow",
                           "down_right": "diagonal down-right arrow", "cross": "cross signal",
                           "number": "number signal", "unknown": "unclear shape"}[shape]
            label = f"{color.capitalize()} {shape_label}"
            candidates.append({
                "id": f"light:{color}:{shape}", "label": label,
                "kind": "light", "category": "Signal", "color": color, "shape": shape,
                "score": float(scores[y, x]), "color_score": float(color_scores[y, x]),
                "shape_score": float(shape_scores[y, x]), "bbox": [round(v, 2) for v in box],
                "meaning": "Observed lamp colour and shape. Lane relevance is not determined.",
                "model": "lamps", "status": "uncertain" if uncertain else "detected",
            })
    return nms(candidates, threshold=0.45, class_agnostic=True)


class Engine:
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
        self.manifest = json.loads((ROOT / "model-manifest.json").read_text(encoding="utf-8")) if (
            ROOT / "model-manifest.json").exists() else {}

    def status(self):
        files = {name: (MODEL_DIR / spec["file"]).is_file()
                 for name, spec in self.manifest.items()}
        return {"ready": len(files) == 6 and all(files.values()), "models": files,
                "runtime": "ONNX Runtime / CPU", "classes": len(catalogue()),
                "evaluation_status": "Not validated on local road scenes",
                "setup_command": "powershell -ExecutionPolicy Bypass -File setup.ps1"}

    def session(self, name):
        if name not in self.sessions:
            spec = self.manifest.get(name)
            if not spec or not (MODEL_DIR / spec["file"]).exists():
                raise ModelsUnavailable("Models are missing. Run setup.ps1 to download them.")
            path = MODEL_DIR / spec["file"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != spec["sha256"]:
                raise ModelsUnavailable(f"{name} model checksum failed. Run setup.ps1 again.")
            options = ort.SessionOptions()
            options.log_severity_level = 3
            options.intra_op_num_threads = min(4, os.cpu_count() or 2)
            options.inter_op_num_threads = 1
            self.sessions[name] = ort.InferenceSession(str(path), sess_options=options,
                                                       providers=["CPUExecutionProvider"])
        return self.sessions[name]

    def yolo(self, image, name, threshold):
        session = self.session(name)
        size = self.manifest[name]["input_size"]
        resized, scale, padding = letterbox(image, size)
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32) / 255
        output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        height, width = image.shape[:2]
        labels = self.manifest[name].get("labels", [str(i) for i in range(80)])
        return decode_yolo(output, scale, padding, width, height, labels, threshold, name,
                           selected=[9] if name == "housing" else None)

    def raw_lamps(self, image, threshold):
        session = self.session("lamps")
        shape = session.get_inputs()[0].shape
        in_h, in_w = int(shape[2]), int(shape[3])
        resized = cv2.resize(image, (in_w, in_h))
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32) / 255
        output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        height, width = image.shape[:2]
        return decode_lamps(output, width, height, in_w, in_h, threshold)

    def classify_sign(self, image):
        session = self.session("sign_classifier")
        resized = cv2.resize(image, (224, 224)).astype(np.float32) / 255
        resized = (resized - np.array([.485, .456, .406], dtype=np.float32)) / np.array(
            [.229, .224, .225], dtype=np.float32)
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None])
        logits = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        index = int(probabilities.argmax())
        class_id = self.manifest["sign_classifier"]["class_ids"][index]
        return sign_info(GTSRB_LABELS[class_id]), float(probabilities[index])

    def refine_signs(self, image, candidates, threshold):
        refined = []
        for item in candidates:
            if item["model"] != "signs":
                refined.append(item)
                continue
            x1, y1, x2, y2 = item["bbox"]
            dx, dy = .08 * (x2 - x1), .08 * (y2 - y1)
            crop = image[max(0, int(y1 - dy)):min(image.shape[0], math.ceil(y2 + dy)),
                         max(0, int(x1 - dx)):min(image.shape[1], math.ceil(x2 + dx))]
            if not crop.size:
                continue
            info, score = self.classify_sign(crop)
            item = dict(item)
            item["detector_label"] = item["label"]
            item["detector_score"] = item["score"]
            item["classifier_score"] = score
            if score >= max(.80, threshold):
                item.update(info)
                item["score"] = min(item["score"], score)
                item["verified"] = True
            else:
                item["status"] = "uncertain"
                item["meaning"] = "The sign classifier could not verify the detector's label."
            refined.append(item)
        # The specialist classifier resolves overlapping direction variants. Strong
        # urban prohibitions still raise a conflict rather than being overwritten.
        verified = [item for item in refined if item.get("verified")]
        return [item for item in refined if item["model"] != "urban" or not any(
            iou(item["bbox"], other["bbox"]) > .6
            and (item["category"] != "Prohibition" or item["score"] < .90
                 or item["id"] == other["id"])
            for other in verified)]

    def lamps(self, image, threshold):
        session = self.session("signal_classifier")
        resized = cv2.resize(image, (224, 224)).astype(np.float32)
        resized = (resized - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array(
            [58.395, 57.12, 57.375], dtype=np.float32)
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None])
        probabilities = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
        # This graph includes softmax. Applying softmax again destroys its scores.
        index = int(probabilities.argmax())
        score = float(probabilities[index])
        label = self.manifest["signal_classifier"]["labels"][index]
        height, width = image.shape[:2]
        if label == "unknown" or score < max(.60, threshold):
            return [{"id": "light:unknown:unknown", "label": "Traffic signal — uncertain",
                     "kind": "light", "category": "Signal", "color": "unknown", "shape": "unknown",
                     "score": score, "bbox": [0, 0, width, height], "status": "uncertain",
                     "model": "signal_classifier", "box_scope": "signal_housing",
                     "meaning": "The signal classifier could not read a known state reliably."}]
        localized = self.raw_lamps(image, threshold)
        result = []
        for token in label.split(","):
            # TIER IV's convention: a bare direction denotes a GREEN arrow; a bare
            # colour denotes a circular lamp. 'left,red' is not a red left arrow.
            color = token if token in {"red", "green", "yellow"} else "green"
            shape = "circle" if token in {"red", "green", "yellow"} else token
            matches = [lamp for lamp in localized if lamp["color"] == color and lamp["shape"] == shape]
            # Strong, localised arrows can refine a circular-only classifier class.
            if shape == "circle" and len(label.split(",")) == 1:
                arrows = [lamp for lamp in localized if lamp["color"] == color
                          and lamp["shape"] in {"left", "right", "straight", "up_left", "up_right"}
                          and lamp["shape_score"] >= .9 and lamp["score"] >= .85]
                if arrows:
                    best_arrow = max(arrows, key=lambda lamp: lamp["score"])
                    shape, matches = best_arrow["shape"], [best_arrow]
            matching = max(matches, key=lambda lamp: lamp["score"], default=None)
            shape_label = "circular light" if shape == "circle" else shape.replace("_", " ") + " arrow"
            result.append({"id": f"light:{color}:{shape}", "label": f"{color.capitalize()} {shape_label}",
                           "kind": "light", "category": "Signal", "color": color, "shape": shape,
                           "score": score, "classifier_score": score,
                           "bbox": matching["bbox"] if matching else [0, 0, width, height],
                           "box_scope": "lamp" if matching else "signal_housing",
                           "model": "signal_classifier+lamps", "status": "detected",
                           "meaning": "Observed signal state. Lane relevance is not determined."})
        return result

    def predict(self, image, mode="scene", threshold=0.55, detail=False):
        started = time.perf_counter()
        with self.lock:
            if mode == "signal":
                detections = self.lamps(image, threshold)
            else:
                # Specialist models have complementary class sets.
                signs = self.yolo(image, "signs", threshold)
                urban = self.yolo(image, "urban", threshold)
                housings = [item for item in urban if item["kind"] == "light"]
                signs = [item for item in signs + urban if item["kind"] == "sign"]
                housings += self.yolo(image, "housing", threshold)
                housings = nms(housings, class_agnostic=True)
                # Optional overlapping tiles recover small signs at the cost of extra CPU work.
                if detail and image.shape[1] >= 1000:
                    h, w = image.shape[:2]
                    for x in (0, w // 3):
                        crop = image[:max(1, int(h * 0.85)), x:x + int(w * 0.67)]
                        for model in ("signs", "urban"):
                            for found in self.yolo(crop, model, threshold):
                                found["bbox"][0] += x
                                found["bbox"][2] += x
                                if found["kind"] == "sign":
                                    signs.append(found)
                detections = resolve_sign_conflicts(self.refine_signs(image, signs, threshold))
                for housing in housings[:20]:
                    x1, y1, x2, y2 = housing["bbox"]
                    # A small margin keeps lamp edges when the housing box is slightly tight.
                    dx, dy = (x2 - x1) * 0.05, (y2 - y1) * 0.05
                    x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
                    x2, y2 = min(image.shape[1], math.ceil(x2 + dx)), min(image.shape[0], math.ceil(y2 + dy))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    lamps = self.lamps(image[y1:y2, x1:x2], threshold)
                    for lamp in lamps:
                        lamp["bbox"] = [lamp["bbox"][0] + x1, lamp["bbox"][1] + y1,
                                        lamp["bbox"][2] + x1, lamp["bbox"][3] + y1]
                        lamp["housing_score"] = housing["score"]
                    if not lamps:
                        housing.update(id="light:unknown:unknown", label="Traffic light — uncertain",
                                       status="uncertain", color="unknown", shape="unknown",
                                       meaning="A signal was located, but its lamp state is unclear.")
                    detections.extend(lamps or [housing])
                detections = nms(detections)
            detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
        return {"detections": detections, "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "width": image.shape[1], "height": image.shape[0], "mode": mode,
                "threshold": threshold, "detail": detail, "count": len(detections),
                "model_version": "pretrained-1.0", "score_note": "Model scores are not measured accuracy."}


def resolve_sign_conflicts(signs):
    """Keep contradictory labels visible as uncertain instead of choosing silently."""
    ordered = nms(signs)
    resolved, consumed = [], set()
    for index, item in enumerate(ordered):
        if index in consumed:
            continue
        conflicts = [(j, other) for j, other in enumerate(ordered) if j > index
                     and j not in consumed and other["id"] != item["id"]
                     and iou(item["bbox"], other["bbox"]) > 0.6]
        if conflicts:
            item = dict(item)
            item["status"] = "uncertain"
            item["alternatives"] = [row["label"] for _, row in conflicts]
            item["meaning"] = "Models disagree on this sign. Inspect the crop before using its label."
            consumed.update(j for j, _ in conflicts)
        resolved.append(item)
    return resolved
