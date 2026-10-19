# Comparable public projects

Checked 2026-10-19. These are references, not independently validated replacements.
Published metrics use different datasets and must not be compared as one benchmark.

| Repository | Coverage | Setup / weights |
|---|---|---|
| [Harisawan115/Traffic-Light-Detection-and-Sign-Board-Recognition-System](https://github.com/Harisawan115/Traffic-Light-Detection-and-Sign-Board-Recognition-System) | Closest application scope: dual YOLOv8 models, light colours and road signs, Streamlit image/video/webcam UI | README instructs downloading datasets and training both models; usable trained weights are not supplied in the listed files |
| [pedramebd/traffic-sign-detection-recognition](https://github.com/pedramebd/traffic-sign-detection-recognition) | YOLOv8 detector plus CNN classification of 43 German sign classes | README explicitly says trained weights are excluded; training notebooks provided. No traffic-light state recognizer |
| [Satish-Vennapu/Traffic-Light-Detection-using-YOLOv8](https://github.com/Satish-Vennapu/Traffic-Light-Detection-using-YOLOv8) | Traffic-light detection and red/yellow/green classification | Includes trained-weight archives and download links; does not cover the complete directional road-sign task |
| [nineRishav/Traffic_Light_Sign](https://github.com/nineRishav/Traffic_Light_Sign) | Traffic-light states plus speed-limit OCR and temporal confirmation | Different sign scope: speed limits, not a broad directional-sign catalogue |

Harisawan115's application is the closest architectural reference; Satish-Vennapu's
project is useful for evaluating an alternate traffic-light model. None has been run
on the user's uploaded image in this review. Unfamiliar sign designs still need
appropriate labels, trained classes, and representative evaluation.
