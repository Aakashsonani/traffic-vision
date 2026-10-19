# Traffic Vision: Traffic Sign and Signal Recognition

## Abstract

Traffic Vision is a Python application that identifies road signs, traffic light
colours and directional signal states. It combines object detection with specialist
classifiers and runs locally on a CPU. A browser interface supports images, camera
frames and sampled video, with annotated results and downloadable observations.

## Problem and objectives

Traffic information appears as printed signs and illuminated lamps. Colour alone
cannot distinguish a mandatory left turn from a prohibited left turn, or determine
whether a lit signal is circular or directional. The project aims to locate visible
objects, assign explicit semantic labels, expose uncertainty and evaluate the complete
recognition pipeline reproducibly.

## Methodology

1. Decode the input into RGB and respect photo orientation.
2. Letterbox frames for two specialist YOLO sign detectors and a COCO housing detector.
3. Restore detected boxes to image coordinates and suppress duplicates.
4. Verify European sign crops with a 43-class ResNet classifier. Preserve the classifier's
   explicit output-to-GTSRB numeric mapping rather than assuming numeric folder order.
5. Classify signal housings with TIER IV's MobileNet model. Decode multi-lamp tokens:
   `left,red` means a green left arrow and a red circular lamp.
6. Use the lamp recognizer to refine geometry and strongly supported arrow shapes.
7. Mark uncertain or contradictory observations and display labels, boxes and scores.

## Architecture

FastAPI provides the local HTTP server. OpenCV and NumPy implement image transforms
and postprocessing. ONNX Runtime executes pretrained models without a GPU. A plain
HTML/CSS/JavaScript frontend captures webcam/video frames and submits them to the
same inference endpoint used for uploaded images. Images remain in server memory.

The repository separates inference, geometry, taxonomy, API, evaluation, training
and frontend code. Model versions, hashes and sources are recorded in a manifest.

## Evaluation

The evaluator matches an exact semantic class and IoU >= 0.5 with one-to-one object
matching. Precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is the harmonic mean
of precision and recall. An abstained observation cannot count as a true positive.
Negative scenes expose false positives. Recording/location groups and duplicate
image checks reduce leakage across splits.

A small v1.1 development check accepted 36/36 sign crops correctly. It accepted 14/18
signal colour crops correctly, with four uncertain results.
There were no incorrect accepted observations in that check. Upstream training
overlap is unknown and the samples were used during development, so these results
must not be presented as independent road-scene accuracy. See VALIDATION.md.

## Limitations

Pretrained weights were integrated; this release does not claim original model
training or country-specific validation. Detection of every catalogue class in road
scenes is unmeasured. Unknown signs can be misclassified. Blur, distance, occlusion,
night lighting and unfamiliar designs can reduce performance. Arrow-lamp accuracy
needs a dedicated labelled evaluation. Video is sampled, not guaranteed real-time.
Signal lane relevance and driving decisions are outside the project scope.

## Future work

Collect grouped local road images with exhaustive object annotations. Evaluate
direction and prohibition errors separately, then fine-tune on training groups and
freeze thresholds on validation groups. Test once on separate recordings. Expand
underrepresented classes, evaluate arrow lamps, and measure performance by lighting,
object size and camera type. Add tracking only after checking its state-change delay.

## References

Model and dataset references, revisions and licenses are listed in
THIRD_PARTY_NOTICES.md and model-manifest.json. Reproduction commands and detailed
results are documented in VALIDATION.md.
