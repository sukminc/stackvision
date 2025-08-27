
# StackVision — Python-first ML Pipeline (MVP)

This scaffold contains everything you need to label a small venue-specific dataset, train an instance segmentation model (YOLOv8-seg), evaluate it, and export to Core ML for the iOS shell.

**Quick start**
1) Create & activate a virtualenv
2) `pip install ultralytics coremltools onnx onnxruntime opencv-python rich`
3) Put your labeled data into `ml-pipeline/datasets/pokerplace_v1` (YOLO format)
4) Update `chips.yaml` classes if your denominations differ
5) `python ml-pipeline/training/train_yolov8_seg.py`
6) Exported `.mlmodel` appears under `ml-pipeline/models/coreml/`
