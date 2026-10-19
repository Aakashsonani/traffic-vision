# Third-party sources and attribution

Source code in this repository is licensed under AGPL-3.0. Model files and sample
images retain their upstream terms; the repository's license does not replace them.
`model-manifest.json` records exact upstream revisions and SHA256 hashes.

| Component | Source | Terms / use |
|---|---|---|
| 33-class sign detector | [Phearith](https://huggingface.co/Phearith/Traffic_Sign_Detection_Using_YOLOv8) | Model card declares MIT; Ultralytics YOLO architecture/export is AGPL-3.0 |
| 21-class urban detector | [Mian Yahya Gul](https://huggingface.co/yahyagul/traffic-sign-yolov8) | Model card declares MIT; Ultralytics architecture/export is AGPL-3.0 |
| COCO traffic-light housing detector | [Ultralytics YOLOv8s](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| ResNet18 GTSRB classifier and output ordering | [sdklfjdsfs/gtsrb-resnet18](https://huggingface.co/sdklfjdsfs/gtsrb-resnet18) | MIT, per model card |
| MobileNetV2 signal classifier and lamp recognizer | [Autoware Foundation / TIER IV](https://huggingface.co/AutowareFoundation/traffic_light_classifier) | Apache-2.0 |
| Lamp decoding equations and arrow angle convention | [Autoware cnn_lamp_recognizer.cpp](https://github.com/autowarefoundation/autoware_universe/blob/main/perception/autoware_traffic_light_classifier/src/classifier/cnn_lamp_recognizer.cpp) | Copyright 2026 TIER IV, Inc.; Apache-2.0 |
| Signal token interpretation | [Autoware cnn_classifier.cpp](https://github.com/autowarefoundation/autoware_universe/blob/main/perception/autoware_traffic_light_classifier/src/classifier/cnn_classifier.cpp) | Copyright 2023 TIER IV, Inc.; Apache-2.0 |
| Sign photographs used in teaching boards | [GTSRB](https://benchmark.ini.rub.de/) | Original benchmark imagery; used here as attributed educational examples. Individual files recorded in samples/manifest.json. |
| Signal example crops | [TIER IV tutorial sample dataset](https://autoware-files.s3.us-west-2.amazonaws.com/dataset/traffic_light_sample_dataset.tar.gz) | Public Autoware training tutorial examples; upstream data rights retained. Individual files recorded in samples/manifest.json. |

The lamp equations are adapted from C++ into Python/NumPy, run on CPU ONNX Runtime,
and combined with a primary signal classifier. No Autoware ROS or TensorRT runtime
is required. The full signal classifier returns whole-housing states, while the
lamp recognizer can refine individual lamp geometry and strongly supported arrows.

Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

GTSRB citation: J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, *Man vs. computer:
Benchmarking machine learning algorithms for traffic sign recognition*, Neural
Networks 32 (2012), 323–332, https://doi.org/10.1016/j.neunet.2012.02.016.

The sidebar mark and workspace illustration were created for this project. Catalogue
glyphs are schematic UI cues, not official sign reproductions or training images.
