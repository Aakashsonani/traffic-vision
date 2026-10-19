import pytest

from traffic_vision.evaluation import ObjectMetrics, validate_dataset


def test_metrics_count_wrong_directions_misses_and_duplicate_detections():
    metrics = ObjectMetrics()
    metrics.add([{"label": "turn_left", "bbox": [0, 0, 20, 20]},
                 {"label": "stop", "bbox": [50, 0, 70, 20]}],
                [{"id": "turn_right", "bbox": [0, 0, 20, 20], "score": .99},
                 {"id": "stop", "bbox": [50, 0, 70, 20], "score": .9},
                 {"id": "stop", "bbox": [50, 0, 70, 20], "score": .8}])
    report = metrics.report()
    assert (report["tp"], report["fp"], report["fn"]) == (1, 2, 1)
    assert report["precision"] == pytest.approx(1 / 3)
    assert report["recall"] == .5
    assert not report["passed_targets"]


def test_uncertain_predictions_do_not_inflate_accuracy():
    metrics = ObjectMetrics()
    metrics.add([{"label": "stop", "bbox": [0, 0, 20, 20]}],
                [{"id": "stop", "bbox": [0, 0, 20, 20], "score": .99, "status": "uncertain"}])
    report = metrics.report()
    assert report["tp"] == 0 and report["fn"] == 1 and report["rejected"] == 1
    assert report["precision"] is None and report["recall"] == 0


def test_negative_scene_false_positive_counts():
    metrics = ObjectMetrics()
    metrics.add([], [{"id": "stop", "bbox": [0, 0, 20, 20], "score": .9}])
    assert metrics.report()["fp"] == 1


def test_recording_group_leakage_is_rejected():
    with pytest.raises(ValueError, match="leakage"):
        validate_dataset({"images": [
            {"file": "a.jpg", "group": "drive1", "split": "train", "objects": []},
            {"file": "b.jpg", "group": "drive1", "split": "test", "objects": []}]})


@pytest.mark.parametrize("label,box", [("invented_sign", [0, 0, 20, 20]),
                                       ("turn_left", [20, 20, 0, 0]),
                                       ("light:green:invented", [0, 0, 20, 20])])
def test_invalid_ground_truth_is_rejected(label, box):
    with pytest.raises(ValueError):
        validate_dataset({"images": [{"file": "a.jpg", "group": "drive1", "split": "test",
                                     "objects": [{"label": label, "bbox": box}]}]})
