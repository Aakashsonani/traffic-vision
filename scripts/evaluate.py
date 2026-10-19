"""Evaluate full inference on a grouped, exhaustively annotated image dataset."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traffic_vision.evaluation import ObjectMetrics, validate_dataset  # noqa: E402
from traffic_vision.inference import Engine  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=.55)
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "evaluation.json")
    args = parser.parse_args()
    if not .25 <= args.threshold <= .95 or not 0 < args.iou <= 1:
        parser.error("Threshold must be 0.25–0.95 and IoU must be >0 and <=1")
    raw = args.dataset.read_bytes()
    data = json.loads(raw)
    validate_dataset(data)
    # Detect exact-content duplicates across splits as well as group/path leakage.
    seen = {}
    for item in data["images"]:
        path = args.dataset.parent / item["file"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in seen:
            raise ValueError(f"Duplicate image content: {item['file']} and {seen[digest]}")
        seen[digest] = item["file"]
    selected = [item for item in data["images"] if item["split"] == args.split]
    if not selected:
        raise ValueError(f"No images in {args.split} split")
    engine, metrics = Engine(), ObjectMetrics(args.iou)
    errors, latency = [], []
    for index, item in enumerate(selected):
        with Image.open(args.dataset.parent / item["file"]) as source:
            image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        for obj in item["objects"]:
            if obj["bbox"][2] > image.shape[1] or obj["bbox"][3] > image.shape[0]:
                raise ValueError(f"Bounding box exceeds image size: {item['file']}")
        result = engine.predict(image, mode=item.get("mode", "scene"), threshold=args.threshold,
                                detail=args.detail)
        metrics.add(item["objects"], result["detections"])
        latency.append(result["elapsed_ms"])
        errors.append({"file": item["file"], "truth": item["objects"],
                       "predictions": result["detections"]})
        print(f"{index + 1}/{len(selected)} {item['file']}: {result['count']} observations", flush=True)
    report = {**metrics.report(), "dataset_name": data.get("name", args.dataset.stem),
              "scope": data.get("scope", "User-provided dataset; representativeness unspecified"),
              "dataset_sha256": hashlib.sha256(raw).hexdigest(), "split": args.split,
              "threshold": args.threshold, "detail": args.detail,
              "generated_at": datetime.now(UTC).isoformat(),
              "latency_median_ms": float(np.median(latency)),
              "latency_p95_ms": float(np.percentile(latency, 95)),
              "model_hashes": {name: spec["sha256"] for name, spec in engine.manifest.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_name(args.output.stem + "-predictions.json").write_text(json.dumps(errors, indent=2))
    print(json.dumps({key: report[key] for key in ("images", "precision", "recall", "f1",
                                                  "rejected", "passed_targets")}, indent=2))


if __name__ == "__main__":
    main()
