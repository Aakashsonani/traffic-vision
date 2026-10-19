TRAFFIC VISION - SAMPLE IMAGE PACK

104 real image crops: 86 signs (two per GTSRB class) and 18 signal crops.
Open index.html for a labelled gallery; expected-results.csv is the answer key.

HOW TO TEST
1. Open http://127.0.0.1:8765 and upload an image.
2. Use Full road scene for sign-*.png.
3. Use Close-up traffic signal for signal-*.jpg.
4. Start at the default 55% minimum score. Compare with the answer key.
5. Record incorrect, uncertain and missed observations, including high-score errors.

These images were sampled without running the app on them or selecting for success.
Previously used demo/check images were excluded. Upstream model training overlap
is unknown, so this is not an independent real-road accuracy benchmark. The set
contains small crops, not complete road scenes. Signal labels establish colour
only, not arrow direction. There are no verified arrow-lamp or urban no-turn
prohibition examples in this pack; collect separate labelled images for those.

OPTIONAL SIGN OBJECT EVALUATION (from the project folder):
python scripts/evaluate.py --dataset samples/validation/sign-evaluation.json --output outputs/sample-pack-evaluation.json
The bounding boxes in that file are from the original GTSRB annotations.
This does not replace the local-road Evaluation page report.

SOURCES / ATTRIBUTION
Signs: German Traffic Sign Recognition Benchmark, J. Stallkamp et al.
https://benchmark.ini.rub.de/
Signal crops: TIER IV / Autoware public tutorial dataset
https://autoware-files.s3.us-west-2.amazonaws.com/dataset/traffic_light_sample_dataset.tar.gz
Original image rights remain with their authors. Educational testing examples.
