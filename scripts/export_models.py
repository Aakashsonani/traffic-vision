"""Maintainer tool: fetch pinned source checkpoints and export CPU ONNX graphs.

End users run setup.ps1 or scripts/download_models.py instead; no PyTorch is needed
for inference. YOLO source checkpoints retain their upstream model licenses.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_models import download  # noqa: E402

SOURCES = [
    (
        "signs",
        "https://huggingface.co/Phearith/Traffic_Sign_Detection_Using_YOLOv8/resolve/"
        "946a0ecc14306ac6ea04073ea0f984964070851d/best_yolov8m.pt",
        "fd2b7646f31b5e1600e3d24ba27241925ed648908853d216a43b0892c1850392",
        960,
    ),
    (
        "urban",
        "https://huggingface.co/yahyagul/traffic-sign-yolov8/resolve/"
        "15be2681d3b6e666f45c744219aab84e97a2f1b4/best_roboflow.pt",
        "93793cc99ffda2a10e09860e3a7c6383f7c25d470dd604c4555b5b07253a6e2b",
        640,
    ),
]


def main():
    from ultralytics import YOLO

    folder = ROOT / "models"
    folder.mkdir(exist_ok=True)
    manifest_path = ROOT / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, url, digest, size in SOURCES:
        checkpoint = folder / f"{name}.pt"
        download(url, checkpoint, digest)
        model = YOLO(str(checkpoint))
        output = folder / f"{name}.onnx"
        if not output.exists():
            model.export(format="onnx", imgsz=size, opset=17, simplify=False, dynamic=False)
        labels = [model.names[i] for i in range(len(model.names))]
        (folder / f"{name}.labels.json").write_text(json.dumps(labels, indent=2) + "\n")
        manifest[name] = {
            **manifest.get(name, {}),
            "file": output.name, "sha256": hashlib.file_digest(output.open("rb"), "sha256").hexdigest(),
            "size": output.stat().st_size, "input_size": size, "labels": labels,
            "source": url,
        }
    # Official COCO detector is used only for traffic-light housings (class 9).
    generic = folder / "housing.onnx"
    if not generic.exists():
        model = YOLO("yolov8s.pt")
        exported = Path(model.export(format="onnx", imgsz=640, opset=17,
                                     simplify=False, dynamic=False))
        exported.replace(generic)
    manifest["housing"] = {
        **manifest.get("housing", {}),
        "file": generic.name, "sha256": hashlib.file_digest(generic.open("rb"), "sha256").hexdigest(),
        "size": generic.stat().st_size, "input_size": 640,
        "source": "https://github.com/ultralytics/assets/releases/tag/v8.4.0",
    }
    lamp = folder / "lamps.onnx"
    url = ("https://huggingface.co/AutowareFoundation/traffic_light_classifier/resolve/"
           "527c4905afd8f1bfcc03253529a105f7fa5a6d36/"
           "traffic_light_lamp_recognizer_comlops.onnx")
    digest = "300af04a5893961195a9b7ffd9d80364280f328a6dc549f9b1e05d428d96d2e5"
    download(url, lamp, digest)
    manifest["lamps"] = {"file": lamp.name, "sha256": digest,
                         "size": lamp.stat().st_size, "source": url}
    for name in ("sign_classifier", "signal_classifier"):
        spec = manifest[name]
        download(spec["source"], folder / spec["file"], spec["sha256"])
    (ROOT / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Exported all models; manifest saved.", flush=True)


if __name__ == "__main__":
    main()
