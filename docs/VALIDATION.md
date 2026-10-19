# Validation record

The release uses existing pretrained weights. No custom model training or independent
local-road accuracy validation has been completed. Model scores displayed in the app
are not measured accuracy.

## Public sample check

`scripts/check_pretrained.py`, seed 42, default threshold 0.55. The run covers four
random official GTSRB test crops in each of nine selected sign classes: turn right,
turn left, straight, straight/right, straight/left, keep right, keep left, STOP and
no entry. Exact semantic class and IoU >= 0.5 are required for a true positive.

| Check | Result |
|---|---|
| Direction / regulatory sign crops | 36 accepted correctly, 0 uncertain, 0 incorrect accepted |
| Sign precision on these crops | 100% (36/36 accepted) |
| Sign recall on these crops | 100% (36/36 labelled objects) |
| Signal colour crops | 15 accepted correctly, 3 uncertain, 0 incorrect accepted |
| Signal colour acceptance on these crops | 83.3% (15/18) |

**These numbers are a smoke check, not an independent accuracy claim.** The third-party
weights' training overlap with these public datasets is unknown. The samples do not
establish performance on full road scenes, Indian signs, prohibitions, arrow lamps,
night conditions, or every catalogue class. The six example buttons use selected,
resized teaching images and must not be used for evaluation.

The initial detector-only check confused combined directions. Adding the separate
43-class ResNet sign classifier fixed those errors in this check. A primary MobileNet
signal classifier improved abstention and colour recognition compared with using the
lamp localizer alone. These changes were developed against the sample check, so it
cannot serve as a final held-out test.

## Reproduce

Download these sources into `data/` (approximately 87 MB):

- https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
- https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip
- https://autoware-files.s3.us-west-2.amazonaws.com/dataset/traffic_light_sample_dataset.tar.gz

Run `python scripts/check_pretrained.py`. The full observations and metrics go to
`outputs/pretrained-check.json`. Downloaded datasets are excluded from Git.

## Evaluate your own roads

Copy `docs/evaluation.example.json` into `data/evaluation.json`, replace its examples,
and annotate all supported visible objects with catalogue IDs and pixel bounding
boxes. Empty scenes need `objects: []`. Use recording/location IDs as groups; groups
and exact image contents cannot occur in multiple splits. Signal truth boxes should
match the returned `box_scope` (lamp or signal housing); report these scopes separately
when measuring localization quality. Use an agreed box convention before annotation.

```
python scripts/evaluate.py --dataset data/evaluation.json --split test
```

The Evaluation page reads `outputs/evaluation.json`. The report records model hashes,
dataset hash, precision/recall/F1, per-class counts, abstentions, confusion pairs and
latency. Missing classes cannot establish coverage. The proposed 95% precision and
90% recall targets must also be assessed per supported class on sufficient data.

## Software checks

```
uv sync --locked --extra dev
uv run pytest -q
uv run ruff check .
node --check web/app.js
uv run python scripts/check_browser.py
```

Tests cover coordinate restoration, class mappings, no extra objectness/sigmoid/
softmax, arrow orientation, conflicting predictions, unknown signal labels, missed
objects, negative scenes, abstention, data leakage, invalid uploads and API contracts.
