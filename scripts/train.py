"""Fine-tune a detector on YOUR labelled data, with direction-preserving transforms."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("Dataset YAML does not exist")
    try:
        from ultralytics import YOLO
    except ImportError:
        parser.error("Install training dependencies: uv sync --extra training --extra dev")
    model = YOLO(args.model)
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                device=args.device, batch=args.batch, seed=42, deterministic=True,
                fliplr=0.0, flipud=0.0, degrees=5, hsv_h=0.0, hsv_s=0.15, hsv_v=0.25,
                patience=12, project="runs", name="traffic", workers=0, amp=False)
    print("Review validation errors and freeze thresholds before running the held-out test split.")
    print("Export best weights with: yolo export model=runs/traffic/weights/best.pt format=onnx")
    print("Custom taxonomies require a matching manifest and explicit catalogue mapping.")


if __name__ == "__main__":
    main()
