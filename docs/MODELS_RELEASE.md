# Converted inference models — models-v1

ONNX CPU inference exports for Traffic Vision. Exact sizes and SHA256 digests are
recorded in model-manifest.json. Original model creators and sources are credited in
THIRD_PARTY_NOTICES.md. These exports are pretrained models, not fresh local training.

- signs.onnx: Phearith YOLOv8m, 33 sign classes, input 960 x 960.
- urban.onnx: Mian Yahya Gul YOLOv8s, 21 urban sign/light classes, input 640 x 640.
- housing.onnx: Ultralytics COCO YOLOv8s, input 640 x 640, used for signal class 9.

Exported with Ultralytics 8.4.182, Torch 2.14.1+cpu and ONNX 1.23.2, opset 17,
static shapes, FP32, without export-time NMS or simplification. Raw detector scores
already include their class activation. Runtime uses the manifest's explicit labels.

The remaining ResNet and TIER IV classifiers are downloaded directly from their
immutable Hugging Face revisions. See scripts/download_models.py for setup.

The YOLO models/architecture are distributed under AGPL-3.0, retaining the model
cards' original MIT notices where declared. See the repository's third-party notices.
