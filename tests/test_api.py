import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from traffic_vision.app import app, decode_image

client = TestClient(app)


def test_app_catalogue_and_health_are_available_without_loading_models():
    assert client.get("/").status_code == 200
    health = client.get("/api/health").json()
    assert isinstance(health["ready"], bool)
    assert health["classes"] > 40
    assert "no_left" in {row["id"] for row in client.get("/api/catalogue").json()["signs"]}


def test_corrupt_and_empty_uploads_return_clear_errors():
    response = client.post("/api/analyze", files={"file": ("bad.jpg", b"not an image", "image/jpeg")})
    assert response.status_code == 400
    assert "image" in response.json()["detail"]
    assert client.post("/api/analyze", files={"file": ("empty.jpg", b"", "image/jpeg")}).status_code == 400


def test_unsupported_mode_and_threshold_are_rejected():
    for fields in ({"mode": "invented"}, {"threshold": "0.01"}):
        assert client.post("/api/analyze", files={"file": ("x.jpg", b"bad")}, data=fields).status_code == 422


def test_image_preprocessing_preserves_rgb_channel_order():
    stream = io.BytesIO()
    Image.new("RGB", (30, 20), (255, 0, 0)).save(stream, format="PNG")
    image = decode_image(stream.getvalue())
    np.testing.assert_array_equal(image[0, 0], [255, 0, 0])
    assert image.shape == (20, 30, 3)
