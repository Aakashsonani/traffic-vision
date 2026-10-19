# Viva preparation

**Why not use only colour detection?**
Colour does not identify the printed symbol, speed value or arrow direction. Real
scenes also contain unrelated coloured objects and changing illumination.

**How do detection and classification differ?**
Detection locates an object's box. Classification assigns a semantic class to its crop.

**What did you implement versus reuse?**
The project integrates pretrained detector/classifier weights. Its application,
pre/postprocessing, taxonomy mapping, uncertainty handling, UI, evaluation, setup and
tests are implemented in this repository. Upstream sources are attributed.

**Why are two sign detectors used?**
Their class sets are complementary: one covers European signs including mandatory
directions, while the urban detector adds prohibitions and information signs.

**Why add a ResNet classifier?**
The first sample checks revealed high-score left/right-combination confusions.
An independent crop classifier improved those cases in the development check.

**What is letterboxing?**
Resize while preserving aspect ratio, then pad to the detector's input shape. Undo
the scale and padding to restore boxes to the displayed image coordinates.

**What is non-maximum suppression?**
Remove lower-score duplicate boxes with excessive overlap. Separate nearby signs
should survive suppression.

**Why can a red lamp and a green arrow appear together?**
They are separate lamps with separate meanings. The project preserves each rather
than compressing the whole housing into one colour/direction.

**Is a 95% score the same as 95% accuracy?**
No. A score is a model output. Accuracy and precision require labelled evaluation;
an uncalibrated model can be confidently wrong.

**What are precision, recall and IoU?**
Precision measures how many accepted predictions are correct. Recall measures how
many labelled objects are recognised. IoU measures bounding-box overlap.

**How do uncertain results affect evaluation?**
They cannot become true positives. Their corresponding labelled objects remain
misses, so abstaining cannot artificially improve recall.

**What is data leakage?**
Related frames, duplicate images or shared recordings appearing across training
and test sets can inflate results. Split by recording/location before training.

**Did you achieve 95% precision on local roads?**
That has not been measured. The recorded check uses small public samples with
unknown upstream training overlap; representative local-road evaluation remains necessary.

**What would you improve next?**
Create a representative grouped dataset, evaluate every class including arrow lamps,
inspect confusions and fine-tune using evidence from the validation split.
