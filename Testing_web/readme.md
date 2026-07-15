# Emotix camera web demo

This folder contains the frontend and backend bridge that connects the trained Emotix
emotion model to a browser camera.

The app is split into two parts:

- React frontend: opens the webcam, captures a frame, sends it to the backend, and draws prediction boxes.
- FastAPI backend: receives the captured image, runs `Models/best.onnx`, and returns detected emotions.

Read [STUDY_GUIDE.md](./STUDY_GUIDE.md) for the full explanation of how everything was connected.

## Required installs

Install Node.js for the React frontend.

Install Python packages for the backend:

```powershell
cd C:\Users\Hariom\Desktop\yolo
pip install -r Testing_web\requirements-web.txt
```

If you use the bundled Python from Codex, run:

```powershell
cd C:\Users\Hariom\Desktop\yolo
C:\Users\Hariom\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m pip install -r Testing_web\requirements-web.txt
```

Install frontend packages:

```powershell
cd C:\Users\Hariom\Desktop\yolo\Testing_web
npm install
```

## Run backend

Keep this terminal open:

```powershell
cd C:\Users\Hariom\Desktop\yolo
python -m uvicorn Testing_web.api:app --host 127.0.0.1 --port 8000
```

Or with the bundled Codex Python:

```powershell
cd C:\Users\Hariom\Desktop\yolo
C:\Users\Hariom\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m uvicorn Testing_web.api:app --host 127.0.0.1 --port 8000
```

Check backend health:

```text
http://localhost:8000/health
```

## Run frontend

Keep this terminal open too:

```powershell
cd C:\Users\Hariom\Desktop\yolo\Testing_web
npm run dev
```

Open:

```text
http://localhost:5173
```

The page should show `API online` when the backend is running.

## Main files

- `Testing_web/src/App.jsx`: camera UI, capture logic, API calls, drawing prediction boxes.
- `Testing_web/src/main.jsx`: React app bootstrapping and error display.
- `Testing_web/src/styles.css`: frontend styling.
- `Testing_web/api.py`: FastAPI + ONNX Runtime emotion prediction backend.
- `Testing_web/requirements-web.txt`: lightweight backend dependencies.
- `Testing_web/package.json`: React/Vite frontend dependencies and scripts.
