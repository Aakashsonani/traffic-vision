import cv2
import numpy as np

from traffic_vision.catalogue import sign_info
from traffic_vision.evidence import neutral_sign, no_entry_bar, sign_regions, visible_lamp_color
from traffic_vision.inference import Engine


def test_missed_no_entry_has_geometric_proposal_and_correct_bar():
    image = np.full((160, 160, 3), 210, np.uint8)
    cv2.circle(image, (80, 80), 45, (240, 10, 10), -1)
    cv2.rectangle(image, (48, 72), (112, 88), (255, 255, 255), -1)
    proposals = sign_regions(image, [])
    assert len(proposals) == 1
    assert proposals[0]["face_color"] == "red"
    assert no_entry_bar(image[35:126, 35:126])
    # A blank red lens is not a sign, and a diagonal slash is not a no-entry bar.
    solid = np.zeros((100, 100, 3), np.uint8)
    cv2.circle(solid, (50, 50), 35, (255, 0, 0), -1)
    assert not sign_regions(solid, [])
    assert not no_entry_bar(solid)


def test_lamp_evidence_distinguishes_bright_yellow_from_dark_green_lens():
    yellow = np.zeros((60, 60, 3), np.uint8)
    cv2.circle(yellow, (30, 30), 22, (255, 180, 20), -1)
    assert visible_lamp_color(yellow)["color"] == "yellow"
    dark = np.zeros((60, 60, 3), np.uint8)
    cv2.circle(dark, (30, 30), 22, (10, 85, 65), -1)
    assert visible_lamp_color(dark, reference_brightness=255) is None
    # A genuinely underexposed signal can still be the brightest saturated lamp
    # in its housing; compare exposure within the housing rather than to a fixed value.
    assert visible_lamp_color(dark, reference_brightness=85)["color"] == "green"


def test_unsupported_symbol_does_not_keep_a_definite_turn_label():
    proposed = {**sign_info("do_not_turn_r"), "score": .9, "bbox": [1, 1, 50, 50]}
    result = neutral_sign(proposed, "Unverified symbol")
    assert result["id"] == "unknown_sign" and result["label"] == "Unrecognised sign"
    assert result["status"] == "uncertain" and result["category"] == "Unverified"
    assert result["proposed_label"] == "No right turn"


def test_grounded_red_and_yellow_override_invented_green_arrow(monkeypatch):
    engine = Engine()

    class FakeSession:
        def get_inputs(self):
            return [type("Input", (), {"name": "input"})()]

        def run(self, *_):
            probabilities = np.zeros((1, 11), np.float32)
            probabilities[0, 4] = .99  # red,right => red circle plus a green right arrow
            return [probabilities]

    monkeypatch.setattr(engine, "session", lambda _: FakeSession())
    boxes = [[5, 10, 35, 40], [5, 65, 35, 95]]
    raw = [{"id": "light:red:circle", "label": "Red circular light", "color": "red",
            "shape": "circle", "shape_score": .99, "score": .99, "status": "detected",
            "kind": "light", "category": "Signal", "bbox": box} for box in boxes]
    monkeypatch.setattr(engine, "raw_lamps", lambda *_: raw)
    image = np.zeros((110, 40, 3), np.uint8)
    cv2.circle(image, (20, 25), 14, (255, 15, 10), -1)
    cv2.circle(image, (20, 80), 14, (255, 180, 20), -1)
    result = engine.lamps(image, .55)
    assert {item["id"] for item in result} == {"light:red:circle", "light:yellow:circle"}
    assert all(item["box_scope"] == "lamp" for item in result)


def test_high_confidence_single_circle_can_fall_back_on_a_small_crop(monkeypatch):
    engine = Engine()

    class FakeSession:
        def get_inputs(self):
            return [type("Input", (), {"name": "input"})()]

        def run(self, *_):
            probabilities = np.zeros((1, 11), np.float32)
            probabilities[0, 3] = .99  # red circle only, not an arrow combination
            return [probabilities]

    monkeypatch.setattr(engine, "session", lambda _: FakeSession())
    monkeypatch.setattr(engine, "raw_lamps", lambda *_: [])
    image = np.zeros((12, 5, 3), np.uint8)
    image[0, 0] = (255, 10, 10)
    result = engine.lamps(image, .55)
    assert result[0]["id"] == "light:red:circle"
    assert result[0]["box_scope"] == "signal_housing"
