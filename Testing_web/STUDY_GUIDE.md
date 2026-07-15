# Emotix frontend/backend connection study guide

This guide documents what was built, why it was built this way, and how the React frontend connects to the emotion detection model.

## 1. Goal

The project already had trained model files:

- `Models/best.pt`
- `Models/last.pt`
- `Models/best.onnx`

The missing part was a working application flow:

1. Open camera in the browser.
2. Capture a frame from the camera.
3. Send that frame to the model.
4. Run emotion prediction.
5. Return prediction results to the frontend.
6. Draw emotion labels and bounding boxes on the video.

The final solution uses React for the frontend and FastAPI for the backend.

## 2. Why frontend and backend are separate

The browser cannot directly run the original PyTorch `.pt` model in a normal React app. React runs in the browser, while YOLO/PyTorch style model inference usually runs in Python.

So we use this architecture:

```text
Browser camera
    |
    v
React frontend at http://localhost:5173
    |
    | POST image frame
    v
FastAPI backend at http://localhost:8000
    |
    | ONNX Runtime inference
    v
Models/best.onnx
    |
    v
JSON predictions returned to React
```

This keeps the frontend simple and lets Python handle model inference.

## 3. Frontend technologies installed

The frontend uses:

- `react`: UI library.
- `react-dom`: renders React into the browser DOM.
- `vite`: development server and build tool.
- `@vitejs/plugin-react`: React support for Vite.

These are listed in:

```text
Testing_web/package.json
```

Install them with:

```powershell
cd C:\Users\Hariom\Desktop\yolo\Testing_web
npm install
```

Run the frontend with:

```powershell
npm run dev
```

The frontend runs at:

```text
http://localhost:5173
```

## 4. Backend technologies installed

The backend uses:

- `fastapi`: creates the HTTP API.
- `uvicorn`: runs the FastAPI server.
- `python-multipart`: lets FastAPI receive uploaded image files.
- `pillow`: reads uploaded images.
- `opencv-python`: resizes and pads images before model inference.
- `numpy`: array operations for image/model data.
- `onnxruntime`: runs `Models/best.onnx`.

These are listed in:

```text
Testing_web/requirements-web.txt
```

Install them with:

```powershell
cd C:\Users\Hariom\Desktop\yolo
pip install -r Testing_web\requirements-web.txt
```

Run the backend with:

```powershell
cd C:\Users\Hariom\Desktop\yolo
python -m uvicorn Testing_web.api:app --host 127.0.0.1 --port 8000
```

The backend runs at:

```text
http://localhost:8000
```

## 5. Why we switched from `.pt` to `.onnx`

The first backend version loaded:

```text
Models/best.pt
```

That requires `ultralytics` and `torch`. On Windows, installing `torch` failed because of a long-path package installation problem.

The repo already had:

```text
Models/best.onnx
```

So the backend was changed to use ONNX Runtime instead. This is lighter and easier to run for a web API demo.

The active model path is now in `Testing_web/api.py`:

```python
MODEL_PATH = ROOT_DIR / "Models" / "best.onnx"
```

## 6. Backend API endpoints

The backend file is:

```text
Testing_web/api.py
```

It exposes two endpoints.

### Health endpoint

```text
GET /health
```

Purpose:

- Confirms the backend is running.
- Confirms the ONNX model can be loaded.
- Returns model path and input shape.

Example URL:

```text
http://localhost:8000/health
```

### Prediction endpoint

```text
POST /predict
```

Purpose:

- Receives an uploaded image under the form field name `file`.
- Runs emotion detection.
- Returns JSON detections.

The frontend sends the request like this:

```js
const formData = new FormData();
formData.append("file", blob, "camera-frame.jpg");

const response = await fetch(`${API_URL}/predict`, {
  method: "POST",
  body: formData,
});
```

Example response:

```json
{
  "image": { "width": 1280, "height": 720 },
  "detections": [
    {
      "label": "happy",
      "class_id": 2,
      "confidence": 0.82,
      "box": [120.5, 80.2, 260.7, 240.9]
    }
  ]
}
```

## 7. Backend prediction flow

The backend prediction pipeline is:

1. Read uploaded image with Pillow.
2. Convert image to RGB.
3. Resize/pad image to `320 x 320` using letterbox preprocessing.
4. Convert image into model tensor format:

```text
[1, 3, 320, 320]
```

5. Run ONNX Runtime inference.
6. Decode YOLO output.
7. Convert boxes from center format to corner format.
8. Filter low-confidence detections.
9. Scale boxes back to the original image size.
10. Apply non-max suppression.
11. Return detections as JSON.

Important backend constants:

```python
CLASS_NAMES = ["anger", "fear", "happy", "neutral", "sad", "surprise"]
IMAGE_SIZE = 320
```

## 8. What letterbox preprocessing does

The model expects a square `320 x 320` image, but camera frames are usually rectangular.

Letterbox preprocessing:

- Keeps the original aspect ratio.
- Resizes the image to fit inside `320 x 320`.
- Adds gray padding where needed.

This avoids stretching faces, which helps the model predict better.

In code:

```python
model_image, ratio, (pad_x, pad_y) = letterbox(original)
```

Later, `ratio`, `pad_x`, and `pad_y` are used to convert predicted boxes back to the original camera image size.

## 9. Frontend main files

The React app lives in:

```text
Testing_web/src
```

Important files:

- `main.jsx`: starts React and displays fallback errors.
- `App.jsx`: contains camera, capture, API calls, and drawing logic.
- `styles.css`: layout and UI styling.

## 10. Frontend camera flow

The camera starts with:

```js
navigator.mediaDevices.getUserMedia({
  video: { width: { ideal: 1280 }, height: { ideal: 720 } },
  audio: false,
});
```

The stream is assigned to the video element:

```js
videoRef.current.srcObject = stream;
```

React refs are used because video and canvas are real browser elements:

```js
const videoRef = useRef(null);
const overlayRef = useRef(null);
```

## 11. Capturing a camera frame

When the user clicks `Capture`, the frontend:

1. Creates a temporary canvas.
2. Draws the current video frame into it.
3. Converts that canvas to a JPEG blob.
4. Sends the blob to the backend.

Core logic:

```js
const canvas = document.createElement("canvas");
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;
canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
```

## 12. Drawing prediction boxes

The backend returns boxes in original image coordinates.

The frontend draws those boxes on a canvas overlay placed above the video:

```jsx
<video ref={videoRef} />
<canvas ref={overlayRef} className="overlay" />
```

The drawing function:

- Resizes the overlay canvas to match the displayed video size.
- Scales model coordinates to screen coordinates.
- Draws a rectangle.
- Draws a label like `happy 82%`.

This happens in:

```text
drawDetections()
```

## 13. API online/offline status

The frontend checks:

```text
http://localhost:8000/health
```

If the backend is reachable, the UI shows:

```text
API online
```

If it is not reachable, the UI shows:

```text
API offline
```

This was added because the browser error `Failed to fetch` was not clear enough.

## 14. CORS setup

The frontend and backend run on different ports:

- Frontend: `localhost:5173`
- Backend: `localhost:8000`

Browsers block cross-port requests unless the backend allows them. That is why `api.py` includes CORS middleware:

```python
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
```

## 15. Errors we fixed

### Black screen

Problem:

The app page was black and did not show UI.

Fix:

- Added visible fallback content in `index.html`.
- Added error handling in `main.jsx`.
- Made the startup camera screen visible.

### `React is not defined`

Problem:

The browser showed:

```text
Uncaught ReferenceError: React is not defined
```

Fix:

Imported React in `App.jsx`:

```js
import React, { Component, useCallback, useEffect, useRef, useState } from "react";
```

### `Prediction error: Failed to fetch`

Problem:

The frontend was trying to call:

```text
http://localhost:8000/predict
```

But the backend server was not running.

Fix:

- Added `/health` check.
- Added `API online/offline` indicator.
- Documented that FastAPI must run in a second terminal.

### PyTorch install issue

Problem:

Installing `torch` for the `.pt` model hit a Windows long-path error.

Fix:

Changed backend inference to use:

```text
Models/best.onnx
```

with:

```text
onnxruntime
```

## 16. How to run everything from zero

Open terminal 1:

```powershell
cd C:\Users\Hariom\Desktop\yolo
pip install -r Testing_web\requirements-web.txt
python -m uvicorn Testing_web.api:app --host 127.0.0.1 --port 8000
```

Open terminal 2:

```powershell
cd C:\Users\Hariom\Desktop\yolo\Testing_web
npm install
npm run dev
```

Open browser:

```text
http://localhost:5173
```

Click:

```text
Start -> Capture
```

Or enable:

```text
Auto
```

## 17. How to test backend alone

Open:

```text
http://localhost:8000/health
```

You should see JSON with:

```json
{
  "status": "ok"
}
```

If that does not work, the frontend cannot predict.

## 18. Common troubleshooting

If the page says `API offline`:

- Start the FastAPI backend.
- Make sure it is running on port `8000`.
- Open `http://localhost:8000/health`.

If the camera does not open:

- Use `localhost` or `127.0.0.1`, not a random file path.
- Allow camera permission in the browser.
- Check if another app is using the camera.

If capture says `No emotion detected`:

- Put your face closer and well lit.
- Try lowering backend confidence in the frontend request later if needed.
- Make sure the model was trained for the face angle and lighting condition.

If boxes appear in the wrong place:

- Check the letterbox scaling logic in `api.py`.
- Check the canvas scaling logic in `drawDetections()`.

## 19. What to study next

Useful concepts to learn from this implementation:

- React refs for video and canvas.
- Browser camera API: `navigator.mediaDevices.getUserMedia`.
- Sending files with `FormData`.
- FastAPI file uploads.
- CORS between frontend and backend.
- ONNX Runtime inference.
- YOLO output decoding.
- Non-max suppression.
- Drawing on HTML canvas.
