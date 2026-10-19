import math

import numpy as np
import pytest

from traffic_vision.catalogue import LIGHT_LABELS, catalogue, sign_info
from traffic_vision.geometry import letterbox, nms, restore_boxes
from traffic_vision.inference import (
    Engine,
    arrow_direction,
    decode_lamps,
    decode_yolo,
    resolve_sign_conflicts,
)


def test_every_shipped_sign_class_has_an_explicit_unique_meaning():
    engine = Engine()
    for name in ("signs", "urban"):
        for raw in engine.manifest[name]["labels"]:
            if raw not in LIGHT_LABELS:
                assert sign_info(raw)["id"] in {row["id"] for row in catalogue()}
    identities = [sign_info(name)["id"] for name in ("TurnLeftAhead", "KeepLeft", "do_not_turn_l")]
    assert len(set(identities)) == 3


@pytest.mark.parametrize("shape", [(600, 1200), (1001, 334), (23, 1200)])
def test_letterbox_restores_original_pixel_coordinates(shape):
    image = np.zeros((*shape, 3), dtype=np.uint8)
    _, scales, padding = letterbox(image, 960)
    expected = np.array([[10, 3, shape[1] - 10, shape[0] - 3]], dtype=np.float32)
    transformed = expected.copy()
    transformed[:, [0, 2]] = transformed[:, [0, 2]] * scales[0] + padding[0]
    transformed[:, [1, 3]] = transformed[:, [1, 3]] * scales[1] + padding[1]
    np.testing.assert_allclose(restore_boxes(transformed, scales, padding, shape[1], shape[0]),
                               expected, atol=.001)


def test_yolo_decodes_box_and_class_without_extra_objectness_column():
    output = np.array([[[100], [200], [40], [60], [.15], [.9]]], dtype=np.float32)
    result = decode_yolo(output, (1, 1), (0, 0), 640, 640,
                         ["TurnLeftAhead", "TurnRightAhead"], .55, "signs")
    assert result[0]["id"] == "turn_right"
    assert result[0]["bbox"] == [80, 170, 120, 230]
    assert result[0]["score"] == pytest.approx(.9)


@pytest.mark.parametrize("angle,direction", [(0, "straight"), (math.pi / 2, "right"),
                                                (-math.pi / 2, "left"), (math.pi, "down")])
def test_arrow_orientation_matches_upstream_convention(angle, direction):
    assert arrow_direction(math.cos(angle), math.sin(angle)) == direction
    assert arrow_direction(0, 0) == "unknown"


def test_lamp_decoder_preserves_colour_and_direction_as_separate_attributes():
    output = np.zeros((1, 48, 4, 4), dtype=np.float32)
    output[0, 0:4, 2, 2] = .5
    output[0, 4, 2, 2] = .95
    output[0, 7, 2, 2] = .98  # red = color index 2, not green
    output[0, 9, 2, 2] = .9  # arrow = shape index 1
    output[0, 15, 2, 2] = -1  # negative angle = left
    result = decode_lamps(output, 256, 256, 256, 256)
    assert len(result) == 1
    assert result[0]["id"] == "light:red:left"
    assert result[0]["status"] == "detected"


def test_conflicting_direction_predictions_abstain():
    rows = [{"id": "turn_left", "label": "Turn left", "score": .9, "bbox": [0, 0, 50, 50]},
            {"id": "no_left", "label": "No left turn", "score": .8, "bbox": [1, 1, 51, 51]}]
    result = resolve_sign_conflicts(rows)
    assert len(result) == 1
    assert result[0]["status"] == "uncertain"
    assert result[0]["alternatives"] == ["No left turn"]


def test_nms_retains_separate_nearby_signs():
    rows = [{"id": "stop", "score": .9, "bbox": [0, 0, 50, 50]},
            {"id": "stop", "score": .8, "bbox": [1, 1, 51, 51]},
            {"id": "stop", "score": .7, "bbox": [60, 0, 110, 50]}]
    assert len(nms(rows)) == 2


def test_signal_classifier_does_not_apply_softmax_twice_or_misread_red_left(monkeypatch):
    engine = Engine()

    class FakeSession:
        def get_inputs(self):
            return [type("Input", (), {"name": "input"})()]

        def run(self, *_):
            probs = np.zeros((1, 11), dtype=np.float32)
            probs[0, 1] = .99  # 'left,red': GREEN left arrow alongside RED circular lamp.
            return [probs]

    monkeypatch.setattr(engine, "session", lambda _: FakeSession())
    monkeypatch.setattr(engine, "raw_lamps", lambda *_: [])
    result = engine.lamps(np.zeros((32, 96, 3), dtype=np.uint8), .55)
    assert {item["id"] for item in result} == {"light:green:left", "light:red:circle"}
    assert result[0]["score"] == pytest.approx(.99)


def test_classifier_uses_alphabetical_output_to_numeric_gtsrb_mapping(monkeypatch):
    engine = Engine()

    class FakeSession:
        def get_inputs(self):
            return [type("Input", (), {"name": "input"})()]

        def run(self, *_):
            logits = np.zeros((1, 43), dtype=np.float32)
            logits[0, 29] = 20  # output 29 -> official class 35, NOT class 29.
            return [logits]

    monkeypatch.setattr(engine, "session", lambda _: FakeSession())
    info, score = engine.classify_sign(np.zeros((32, 32, 3), dtype=np.uint8))
    assert info["id"] == "straight" and score > .99


def test_verified_direction_resolves_lane_variant_but_preserves_prohibition_conflict(monkeypatch):
    engine = Engine()
    monkeypatch.setattr(engine, "classify_sign", lambda _: (sign_info("TurnLeftAhead"), .93))
    primary = {**sign_info("TurnLeftAhead"), "kind": "sign", "model": "signs",
               "score": .95, "bbox": [5, 5, 45, 45], "status": "detected"}
    urban = {**sign_info("enter_left_lane"), "kind": "sign", "model": "urban",
             "score": .95, "bbox": [5, 5, 45, 45], "status": "detected"}
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    assert len(engine.refine_signs(image, [primary, urban], .55)) == 1
    prohibited = {**urban, **sign_info("do_not_turn_l")}
    result = resolve_sign_conflicts(engine.refine_signs(image, [primary, prohibited], .55))
    assert result[0]["status"] == "uncertain"
