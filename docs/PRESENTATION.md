# Presentation outline

1. **Title:** Traffic Vision — Traffic Sign and Signal Recognition using Python.
2. **Problem:** identify colours, directional arrows, prohibitions and other signs
   without confusing signs that share a colour or shape.
3. **Objectives:** locate objects; identify explicit meanings; report uncertainty;
   run locally; export and evaluate observations.
4. **Architecture:** frame → detectors → sign/signal classifiers → postprocessing
   → annotated image and JSON. Explain why detection and classification differ.
5. **Technology:** Python, FastAPI, OpenCV, NumPy, ONNX Runtime and browser capture.
6. **Supported classes:** demonstrate left/right/straight, combined arrows,
   keep-left/right, no-turn restrictions, STOP, no entry and signal colours.
7. **Demonstration:** start.bat → load direction example → run recognition → inspect
   labels → toggle original → export JSON → try a signal crop in close-up mode.
8. **Evaluation:** precision, recall, IoU, confusion pairs, recording-level splits
   and abstentions. Present the small public-sample results with their limitations.
9. **Challenges:** directional confusions, tiny objects, multiple lamps, class
   ordering, RGB preprocessing, and model confidence versus measured accuracy.
10. **Future work:** representative local dataset, dedicated arrow-lamp tests,
    fine-tuning, per-class quality targets, temporal evaluation and CPU optimisation.

State clearly that the project integrates pretrained models. Do not present the
public-sample check or the curated examples as independent real-road accuracy.
