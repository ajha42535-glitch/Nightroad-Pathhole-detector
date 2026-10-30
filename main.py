import base64
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

app = FastAPI(title="NightRoad Guardian API", version="1.0.0")
origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins if origin.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_PATH = os.getenv("MODEL_PATH", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "NightRoad pothole model")
_model = None
_model_error = None


def resolve_model_path() -> Optional[Path]:
    if MODEL_PATH:
        candidate = Path(MODEL_PATH).expanduser()
        return candidate if candidate.is_absolute() else (PROJECT_DIR / candidate).resolve()
    for candidate in (
        BASE_DIR / "best.pt",
        BASE_DIR / "yolo26_best.pt",
        PROJECT_DIR / "best.pt",
        PROJECT_DIR / "yolo26_best.pt",
    ):
        if candidate.is_file():
            return candidate
    return None


def get_model():
    global _model, _model_error
    if _model is not None:
        return _model
    path = resolve_model_path()
    if path is None or not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=("Pothole model weights not found. Copy your trained .pt file to "
                    "backend/best.pt or backend/yolo26_best.pt, or set MODEL_PATH."),
        )
    try:
        _model = YOLO(str(path))
        _model_error = None
        return _model
    except Exception as exc:
        _model_error = str(exc)
        raise HTTPException(status_code=503, detail=f"Could not load model weights: {exc}") from exc


@app.get("/")
def home():
    return {"message": "NightRoad Guardian Backend is running!"}


@app.get("/api/health")
def health():
    path = resolve_model_path()
    return {
        "status": "online",
        "service": "NightRoad Guardian API",
        "model_configured": bool(path and path.is_file()),
        "model_name": MODEL_NAME,
    }


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum size is 15 MB.")

    image_array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read this image. Try JPG or PNG.")

    model = get_model()
    try:
        results = model.predict(source=image, conf=0.25, verbose=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    result = results[0]
    names = result.names
    detections = []
    confidences = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            detections.append({
                "class_id": class_id,
                "label": str(class_name),
                "confidence": round(confidence, 4),
                "box": xyxy,
            })
            confidences.append(confidence)

    annotated = result.plot()
    ok, encoded = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode annotated image.")

    return JSONResponse({
        "success": True,
        "model_name": MODEL_NAME,
        "pothole_count": len(detections),
        "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0,
        "detections": detections,
        "annotated_image": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "latitude": latitude,
        "longitude": longitude,
        "note": "Model predictions may be wrong; review detections manually.",
    })
