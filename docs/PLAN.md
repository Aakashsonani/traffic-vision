# Traffic Vision implementation plan

Goal: a Python college project that identifies road signs, traffic light colours and
directional lamps from local images, camera frames and videos. The student owns the
demo and its evaluation; the output is descriptive labels, not driving instructions.

Success means a working offline-after-setup app, real pretrained inference, repeatable
evaluation, a documented class catalogue, and a reproducible GitHub checkout.
95% precision and 90% recall are **targets**, not achieved results.

## Approach

1. CPU-compatible ONNX inference with two specialist sign detectors and a generic
   detector for signal housings; a ResNet classifier verifies sign crops, while TIER IV
   signal and lamp classifiers decode colour, shape and angle from cropped signals.
2. Explicitly preserve turn-left, keep-left and no-left-turn classes. Multiple lamps
   stay independent: red circular + green left arrow are two observations.
3. Image upload, browser webcam and locally decoded video; model scores, bounding
   boxes, class meanings, downloadable JSON, and an annotated image.
4. Pinned model sources and SHA256 downloads. No upload to a remote inference API.
5. Software checks plus actual model inference checks. Real-world quality requires
   representative labelled data from the target country, which has not been supplied.
6. Dataset-aware evaluation using exact class + IoU matches, including missed objects,
   false positives and rejected predictions. Split recordings/locations before training.
7. Publish source, model provenance and setup commands; keep downloaded weights and
   datasets out of Git history. Release converted ONNX weights as separate assets.

## Data contract

Evaluation JSON has an `images` list. Each image contains `file`, `group` (recording or
location), `split`, and `objects`: `{label, bbox: [x1,y1,x2,y2]}` in original pixels.
Every supported visible object must be annotated, including images with no objects.
Train, validation and test group IDs must not overlap. Labels use catalogue IDs for
signs and `light:<color>:<shape>` for lamps. Region-specific signs require their own
data and taxonomy review. Never horizontally flip directional signs without relabeling.

## Mistake budget and fallback

Confusing opposite directions or prohibited turns is unacceptable for claimed
coverage. Low-score/ambiguous objects can be uncertain; unknown classes are not
guaranteed to be rejected. Confidence is a model score, not measured accuracy.
If weights are missing, show setup instructions and fail inference clearly.

Version 1.1 tightens the fallback: detector-only classes remain unverified; neutral
labels replace uncertain semantic guesses. Geometric sign-face proposals can recover
missed signs, while lamp predictions require localised bright-colour evidence. This
does not expand the trained taxonomy or establish unknown-class rejection accuracy.

## Scope and remaining validation

The bundled sign models mix European and urban sign designs. Country-specific
generalisation is unmeasured. Initial installation uses pretrained weights, not a
claim of fresh custom training. Tiny, occluded, blurred or unfamiliar signs remain
hard. The app does not determine lane relevance or combine signs into a driving rule.

Next experiment: evaluate frozen weights on grouped local road images, inspect
direction/prohibition confusions, then fine-tune only when the error analysis supports it.
