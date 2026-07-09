# Emotix — Emotion Detection with YOLOv12

A simple, end-to-end project that trains a **YOLOv12** object detector to recognise facial
emotions, plus a small web demo (`Testing_web/`).

## Emotion classes

`anger` · `fear` · `happy` · `neutral` · `sad` · `surprise`

## Project layout

```
dataset/                          Roboflow YOLOv12 export (train / valid / test)
yolov12_emotion_detection.ipynb   Train + validate + predict + export notebook
Models/                           Trained models: best.pt, last.pt, best.onnx
predictions/                      Sample annotated predictions from the valid set
runs/emotion_yolov12/             Training outputs (metrics, plots, confusion matrix)
Testing_web/                      React web demo
requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
```

Then open **`yolov12_emotion_detection.ipynb`** and run the cells top to bottom. The notebook:

1. Writes a corrected dataset config (`dataset/data_fixed.yaml`).
2. Fine-tunes `yolo12n.pt` on the emotion dataset.
3. Validates the model and reports mAP.
4. Defines two prediction helpers and runs them on sample validation images:
   - `predict_single(model, image_path)` — one image.
   - `predict_batch(model, image_paths)` — a list of images shown in a grid.
5. Exports the model to ONNX and collects `best.pt`, `last.pt` and `best.onnx` into `Models/`.

Trained weights are saved to `runs/emotion_yolov12/weights/` and copied into `Models/`.
