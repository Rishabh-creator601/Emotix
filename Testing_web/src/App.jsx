import React, { Component, useCallback, useEffect, useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <main className="boot-screen">
          <h1>Emotix</h1>
          <p>{this.state.error.message}</p>
        </main>
      );
    }

    return this.props.children;
  }
}

function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  const [cameraReady, setCameraReady] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [autoCapture, setAutoCapture] = useState(false);
  const [detections, setDetections] = useState([]);
  const [message, setMessage] = useState("Camera is not started");
  const [apiReady, setApiReady] = useState(false);

  const checkApi = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }
      setApiReady(true);
      return true;
    } catch {
      setApiReady(false);
      return false;
    }
  }, []);

  const drawDetections = useCallback((items) => {
    const video = videoRef.current;
    const canvas = overlayRef.current;
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) return;

    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 3;
    ctx.font = "14px Arial";

    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;

    items.forEach((item) => {
      const [x1, y1, x2, y2] = item.box;
      const x = x1 * scaleX;
      const y = y1 * scaleY;
      const width = (x2 - x1) * scaleX;
      const height = (y2 - y1) * scaleY;
      const label = `${item.label} ${(item.confidence * 100).toFixed(0)}%`;

      ctx.strokeStyle = "#13c296";
      ctx.fillStyle = "#13c296";
      ctx.strokeRect(x, y, width, height);

      const textWidth = ctx.measureText(label).width + 12;
      ctx.fillRect(x, Math.max(0, y - 24), textWidth, 24);
      ctx.fillStyle = "#071111";
      ctx.fillText(label, x + 6, Math.max(16, y - 7));
    });
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });

      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setCameraReady(true);
      setMessage("Camera ready");
    } catch (error) {
      setMessage(`Camera error: ${error.message}`);
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraReady(false);
    setAutoCapture(false);
    setDetections([]);
    drawDetections([]);
    setMessage("Camera stopped");
  };

  const captureFrame = useCallback(async () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || isPredicting) return;

    setIsPredicting(true);
    setMessage("Running emotion detection...");

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      const backendOnline = await checkApi();
      if (!backendOnline) {
        throw new Error("Backend is not running. Start the FastAPI server on port 8000.");
      }

      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
      if (!blob) {
        throw new Error("Could not capture camera frame");
      }
      const formData = new FormData();
      formData.append("file", blob, "camera-frame.jpg");

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }

      const result = await response.json();
      setDetections(result.detections);
      drawDetections(result.detections);
      setMessage(result.detections.length ? "Prediction complete" : "No emotion detected");
    } catch (error) {
      setMessage(`Prediction error: ${error.message}`);
    } finally {
      setIsPredicting(false);
    }
  }, [checkApi, drawDetections, isPredicting]);

  useEffect(() => {
    checkApi();
  }, [checkApi]);

  useEffect(() => {
    if (autoCapture && cameraReady) {
      intervalRef.current = window.setInterval(captureFrame, 1500);
    }

    return () => {
      window.clearInterval(intervalRef.current);
    };
  }, [autoCapture, cameraReady, captureFrame]);

  useEffect(() => {
    const handleResize = () => drawDetections(detections);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [detections, drawDetections]);

  return (
    <main className="app-shell">
      <section className="camera-panel">
        <div className="video-wrap">
          <video
            ref={videoRef}
            className={cameraReady ? "camera-on" : "camera-off"}
            autoPlay
            playsInline
            muted
            onLoadedMetadata={() => drawDetections(detections)}
          />
          <canvas ref={overlayRef} className="overlay" />
          {!cameraReady && (
            <div className="empty-state">
              <h1>Emotix</h1>
              <p>Camera emotion detection is ready.</p>
              <button onClick={startCamera}>Start Camera</button>
            </div>
          )}
        </div>

        <div className="toolbar">
          {!cameraReady ? (
            <button onClick={startCamera}>Start</button>
          ) : (
            <button onClick={stopCamera}>Stop</button>
          )}
          <button onClick={captureFrame} disabled={!cameraReady || isPredicting}>
            Capture
          </button>
          <label className="toggle">
            <input
              type="checkbox"
              checked={autoCapture}
              disabled={!cameraReady}
              onChange={(event) => setAutoCapture(event.target.checked)}
            />
            Auto
          </label>
          <span className={apiReady ? "api-pill online" : "api-pill offline"}>
            API {apiReady ? "online" : "offline"}
          </span>
          <span className="status">{message}</span>
        </div>
      </section>

      <aside className="results-panel">
        <h1>Emotix</h1>
        <p>Live camera emotion detection</p>

        <div className="result-list">
          {detections.length === 0 ? (
            <div className="result-empty">No detections yet</div>
          ) : (
            detections.map((item) => (
              <div className="result-row" key={`${item.label}-${item.confidence}-${item.box.join("-")}`}>
                <span>{item.label}</span>
                <strong>{(item.confidence * 100).toFixed(1)}%</strong>
              </div>
            ))
          )}
        </div>
      </aside>
    </main>
  );
}

export default App;
export { ErrorBoundary };
