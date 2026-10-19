"""Local web application. Uploaded frames remain in memory."""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from traffic_vision.catalogue import GTSRB_LABELS, catalogue, sign_info
from traffic_vision.inference import Engine, ModelsUnavailable

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web"
MAX_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 36_000_000
engine = Engine()
app = FastAPI(title="Traffic Vision", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/samples", StaticFiles(directory=ROOT / "samples"), name="samples")
logger = logging.getLogger(__name__)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return engine.status()


@app.get("/api/catalogue")
def get_catalogue():
    verified_ids = {sign_info(label)["id"] for label in GTSRB_LABELS}
    signs = [{**item, "review_required": item["id"] not in verified_ids} for item in catalogue()]
    return {"signs": signs, "colors": ["red", "yellow", "green"],
            "lamp_shapes": ["circle", "left", "right", "straight", "up_left", "up_right",
                            "down", "down_left", "down_right", "u_turn", "pedestrian", "cross"],
            "scope": "European and urban sign designs; local country accuracy is unmeasured."}


@app.get("/api/samples")
def get_samples():
    path = ROOT / "samples" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


@app.get("/api/evaluation")
def get_evaluation():
    path = ROOT / "outputs" / "evaluation.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"status": "not_evaluated", "message": "No local road-scene benchmark has been run.",
            "targets": {"precision": 0.95, "recall": 0.90}}


def decode_image(payload):
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Image exceeds 36 megapixels. Resize it before uploading.")
            if image.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                raise ValueError("Use a JPEG, PNG, WebP or BMP image.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            # Keep original coordinates at ordinary image sizes; cap oversized photos for CPU use.
            image.thumbnail((2560, 2560))
            return np.asarray(image).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("This file could not be read as an image.") from exc


def annotate(image, detections):
    rendered = image.copy()
    palette = {"red": (200, 54, 54), "yellow": (180, 128, 24), "green": (29, 119, 82)}
    for index, item in enumerate(detections):
        x1, y1, x2, y2 = map(round, item["bbox"])
        color = (150, 110, 40) if item["status"] == "uncertain" else palette.get(
            item.get("color"), (39, 92, 166))
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        title = f"{index + 1}. {item['label']}"
        if item["id"] != "unknown_sign":
            title += f" {item['score']:.0%}"
        title = title.encode("ascii", errors="replace").decode()
        font_scale = max(0.4, min(0.7, image.shape[1] / 1600))
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_x = max(0, min(x1, image.shape[1] - tw - 8))
        label_y = max(th + 8, y1)
        cv2.rectangle(rendered, (label_x, label_y - th - 8),
                      (min(image.shape[1], label_x + tw + 8), label_y), color, -1)
        cv2.putText(rendered, title, (label_x + 4, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    success, buffer = cv2.imencode(".jpg", cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR),
                                 [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise RuntimeError("Could not encode the annotated image")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()


def analyse(payload, mode, threshold, detail):
    image = decode_image(payload)
    result = engine.predict(image, mode=mode, threshold=threshold, detail=detail)
    result["annotated_image"] = annotate(image, result["detections"])
    return result


@app.post("/api/analyze")
async def analyze(
    file: Annotated[UploadFile, File()],
    mode: Annotated[Literal["scene", "signal"], Form()] = "scene",
    threshold: Annotated[float, Form(ge=0.25, le=0.95)] = 0.55,
    detail: Annotated[bool, Form()] = False,
):
    payload = await file.read(MAX_BYTES + 1)
    await file.close()
    if len(payload) > MAX_BYTES:
        raise HTTPException(413, "Image exceeds the 16 MB upload limit.")
    if not payload:
        raise HTTPException(400, "Choose an image first.")
    try:
        return await run_in_threadpool(analyse, payload, mode, threshold, detail)
    except ModelsUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Recognition failed")
        raise HTTPException(500, "Recognition failed. See the local terminal for details.") from exc
