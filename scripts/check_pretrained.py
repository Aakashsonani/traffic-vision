"""Reproducible public-sample checks, NOT a local-road accuracy benchmark.

Requires GTSRB_Final_Test_{Images,GT}.zip and traffic_light_sample_dataset.tar.gz
in data/. Sources and overlap limitations are documented in docs/VALIDATION.md.
"""

import csv
import io
import json
import random
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traffic_vision.evaluation import ObjectMetrics  # noqa: E402
from traffic_vision.inference import Engine  # noqa: E402


def main():
    data = ROOT / "data"
    engine = Engine()
    rng = random.Random(42)
    classes = {33: "turn_right", 34: "turn_left", 35: "straight", 36: "straight_right",
               37: "straight_left", 38: "keep_right", 39: "keep_left", 14: "stop", 17: "no_entry"}
    with zipfile.ZipFile(data / "GTSRB_Final_Test_GT.zip") as labels:
        csv_name = next(name for name in labels.namelist() if name.endswith(".csv"))
        rows = list(csv.DictReader(io.StringIO(labels.read(csv_name).decode()), delimiter=";"))
    archive = zipfile.ZipFile(data / "GTSRB_Final_Test_Images.zip")
    index = {Path(name).name: name for name in archive.namelist() if name.endswith(".ppm")}
    metrics, predictions = ObjectMetrics(), []
    for class_id, identity in classes.items():
        group = [row for row in rows if int(row["ClassId"]) == class_id]
        for row in rng.sample(group, 4):
            image = np.asarray(Image.open(io.BytesIO(archive.read(index[row["Filename"]]))).convert("RGB"))
            truth = [{"label": identity, "bbox": [int(row[key]) for key in (
                "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")]}]
            result = engine.predict(image)
            metrics.add(truth, result["detections"])
            predictions.append({"file": row["Filename"], "truth": truth,
                                "predictions": result["detections"]})
            print(row["Filename"], identity, [(d["id"], round(d["score"], 2), d["status"])
                                               for d in result["detections"]], flush=True)
    lamp_results = []
    with tarfile.open(data / "traffic_light_sample_dataset.tar.gz") as archive:
        for color in ("red", "yellow", "green"):
            names = [name for name in archive.getnames() if f"/val/{color}/" in name
                     and name.endswith(".jpg")]
            for name in rng.sample(names, 6):
                image = np.asarray(Image.open(archive.extractfile(name)).convert("RGB"))
                result = engine.predict(image, mode="signal")
                accepted = [d for d in result["detections"] if d["status"] == "detected"]
                best = max(accepted, key=lambda d: d["score"], default=None)
                lamp_results.append({"file": name, "expected_color": color,
                                     "correct": bool(best and best["color"] == color),
                                     "predictions": result["detections"]})
                print(name, [(d["id"], round(d["score"], 2)) for d in result["detections"]], flush=True)
    report = {"scope": "Public sample smoke checks. Upstream training overlap is unknown; not independent road-scene accuracy.",
              "seed": 42, "signs": metrics.report(),
              "lamp_color_checks": {"correct": sum(row["correct"] for row in lamp_results),
                                    "total": len(lamp_results)},
              "sign_predictions": predictions, "lamp_predictions": lamp_results}
    output = ROOT / "outputs" / "pretrained-check.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"sign_precision": report["signs"]["precision"],
                      "sign_recall": report["signs"]["recall"],
                      "lamps": report["lamp_color_checks"]}, indent=2))


if __name__ == "__main__":
    main()
