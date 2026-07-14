from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "Models" / "best.onnx"
CLASS_NAMES = ["anger", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_SIZE = 320
MODEL_VERSION = "100-epoch retrain (mAP50=0.567)"  # updated 2026-07-12 commit 75bcaec


app = FastAPI(title="Emotix emotion detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_session() -> ort.InferenceSession:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def letterbox(image: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    ratio = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    padded = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 114, dtype=np.uint8)
    pad_x = (IMAGE_SIZE - new_width) / 2
    pad_y = (IMAGE_SIZE - new_height) / 2

    top = int(round(pad_y - 0.1))
    left = int(round(pad_x - 0.1))
    padded[top : top + new_height, left : left + new_width] = resized

    return padded, ratio, (pad_x, pad_y)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)

        xx1 = np.maximum(x1[current], x1[order[1:]])
        yy1 = np.maximum(y1[current], y1[order[1:]])
        xx2 = np.minimum(x2[current], x2[order[1:]])
        yy2 = np.minimum(y2[current], y2[order[1:]])

        inter_width = np.maximum(0, xx2 - xx1)
        inter_height = np.maximum(0, yy2 - yy1)
        intersection = inter_width * inter_height
        union = areas[current] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-6)

        order = order[1:][iou <= iou_threshold]

    return keep


def run_prediction(image: Image.Image, confidence: float, iou: float) -> list[dict]:
    original = np.array(image.convert("RGB"))
    model_image, ratio, (pad_x, pad_y) = letterbox(original)
    tensor = model_image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    session = get_session()
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0][0].T

    boxes = xywh_to_xyxy(output[:, :4])
    class_scores = output[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    selected = scores >= confidence
    boxes = boxes[selected]
    scores = scores[selected]
    class_ids = class_ids[selected]

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio

    height, width = original.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)

    keep = nms(boxes, scores, iou)
    detections = []

    for index in keep:
        class_id = int(class_ids[index])
        detections.append(
            {
                "label": CLASS_NAMES[class_id],
                "class_id": class_id,
                "confidence": float(scores[index]),
                "box": [float(value) for value in boxes[index].tolist()],
            }
        )

    return detections


@app.get("/health")
def health() -> dict:
    try:
        session = get_session()
        input_shape = session.get_inputs()[0].shape
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "model": str(MODEL_PATH),
        "model_version": MODEL_VERSION,
        "classes": CLASS_NAMES,
        "input_shape": input_shape,
    }


@app.post("/predict")
async def predict(
    file: Annotated[UploadFile, File(...)],
    confidence: Annotated[float, Query(ge=0.01, le=1.0)] = 0.10,
    iou: Annotated[float, Query(ge=0.01, le=1.0)] = 0.45,
) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read image") from exc

    try:
        detections = run_prediction(image, confidence, iou)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "image": {"width": image.width, "height": image.height},
        "detections": detections,
    }
