"""
main.py — FastAPI Obstacle Detection REST API
----------------------------------------------
Exposes YOLOv8 obstacle detection as a production-ready REST API.

Endpoints:
    POST /detect-image   — Detect obstacles in an uploaded image.
    POST /detect-video   — Detect obstacles in an uploaded video.
    GET  /health         — Health check.
    GET  /model-info     — Model metadata.

Usage:
    uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000
    # or
    python deployment/api/main.py
"""

import io
import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from ultralytics import YOLO

from src.utils.config import load_config
from src.utils.helper_functions import draw_detections, generate_color_palette
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

CONFIG_PATH  = "configs/training_config.yaml"
cfg          = load_config(CONFIG_PATH)
inf_cfg      = cfg.inference
deploy_cfg   = cfg.deployment.get("api", {})

WEIGHTS_PATH = inf_cfg.get("model_path", "models/weights/best.pt")
CLASS_NAMES  = [
    "pedestrian", "bicycle", "car", "motorcycle",
    "bus", "truck", "traffic_light", "stop_sign",
]
COLORS = generate_color_palette(len(CLASS_NAMES))

MAX_FILE_MB  = deploy_cfg.get("max_file_size_mb", 50)
MAX_FILE_B   = MAX_FILE_MB * 1024 * 1024

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Autonomous Obstacle Detection API",
    description=(
        "YOLOv8-powered REST API for real-time obstacle detection "
        "in autonomous driving scenarios."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Singleton ─────────────────────────────────────────────────────────

_model: Optional[YOLO] = None


def get_model() -> YOLO:
    """Lazy-load model (singleton)."""
    global _model
    if _model is None:
        logger.info(f"Loading model: {WEIGHTS_PATH}")
        _model = YOLO(WEIGHTS_PATH)
        logger.info("Model loaded.")
    return _model


# ─── Schemas ─────────────────────────────────────────────────────────────────

class Detection(BaseModel):
    class_id:   int   = Field(..., description="Class index.")
    class_name: str   = Field(..., description="Class label string.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox:       List[float] = Field(..., description="[x1, y1, x2, y2] in pixels.")


class ImageDetectionResponse(BaseModel):
    request_id:    str
    filename:      str
    image_shape:   List[int]
    detections:    List[Detection]
    num_detections: int
    inference_ms:  float


class VideoDetectionResponse(BaseModel):
    request_id:     str
    filename:       str
    total_frames:   int
    total_detections: int
    avg_inference_ms: float
    frames:         List[Dict[str, Any]]


class ModelInfoResponse(BaseModel):
    model_path:  str
    class_names: List[str]
    num_classes: int
    input_size:  int
    conf_threshold: float
    iou_threshold:  float


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
async def health_check() -> dict:
    """Check API and model availability."""
    model_ready = Path(WEIGHTS_PATH).exists()
    return {
        "status":      "healthy" if model_ready else "degraded",
        "model_ready": model_ready,
        "timestamp":   time.time(),
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Utility"])
async def model_info() -> ModelInfoResponse:
    """Return model configuration metadata."""
    return ModelInfoResponse(
        model_path=WEIGHTS_PATH,
        class_names=CLASS_NAMES,
        num_classes=len(CLASS_NAMES),
        input_size=cfg.model.get("input_size", 640),
        conf_threshold=inf_cfg.get("conf_threshold", 0.35),
        iou_threshold=inf_cfg.get("iou_threshold", 0.45),
    )


@app.post(
    "/detect-image",
    response_model=ImageDetectionResponse,
    tags=["Detection"],
    summary="Detect obstacles in an uploaded image.",
)
async def detect_image(
    file: UploadFile = File(..., description="Image file (JPG, PNG, BMP)."),
    conf: float = inf_cfg.get("conf_threshold", 0.35),
    iou: float  = inf_cfg.get("iou_threshold", 0.45),
    return_annotated: bool = False,
) -> Any:
    """
    Perform obstacle detection on an uploaded image.

    - **file**: Image file upload (JPEG / PNG / BMP).
    - **conf**: Confidence threshold (0–1).
    - **iou**:  NMS IoU threshold (0–1).
    - **return_annotated**: If true, returns the annotated image as PNG instead of JSON.
    """
    _validate_file_size(file)
    contents = await file.read()
    img = _decode_image(contents, file.filename)

    request_id = str(uuid.uuid4())[:8]
    model = get_model()

    t0 = time.perf_counter()
    results = model.predict(
        img,
        conf=conf,
        iou=iou,
        verbose=False,
        device=inf_cfg.get("device", "cpu"),
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    detections = _parse_detections(results)

    if return_annotated:
        annotated = _annotate_image(img, detections)
        _, buf = cv2.imencode(".png", annotated)
        return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")

    return ImageDetectionResponse(
        request_id=request_id,
        filename=file.filename,
        image_shape=list(img.shape),
        detections=[Detection(**d) for d in detections],
        num_detections=len(detections),
        inference_ms=round(inference_ms, 2),
    )


@app.post(
    "/detect-video",
    response_model=VideoDetectionResponse,
    tags=["Detection"],
    summary="Detect obstacles in an uploaded video.",
)
async def detect_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV)."),
    conf: float = inf_cfg.get("conf_threshold", 0.35),
    iou: float  = inf_cfg.get("iou_threshold", 0.45),
    frame_skip: int = 1,
    max_frames: int = 500,
) -> VideoDetectionResponse:
    """
    Perform obstacle detection on each frame of an uploaded video.

    - **file**:       Video file upload.
    - **conf**:       Confidence threshold.
    - **iou**:        NMS IoU threshold.
    - **frame_skip**: Process every N-th frame (1 = all frames).
    - **max_frames**: Maximum frames to process (to limit latency).
    """
    contents = await file.read()
    request_id = str(uuid.uuid4())[:8]

    with tempfile.NamedTemporaryFile(
        suffix=Path(file.filename).suffix, delete=False
    ) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        frame_results = _process_video_frames(tmp_path, conf, iou, frame_skip, max_frames)
    finally:
        tmp_path.unlink(missing_ok=True)

    total_dets = sum(len(f["detections"]) for f in frame_results)
    avg_ms = (
        sum(f["inference_ms"] for f in frame_results) / max(1, len(frame_results))
    )

    return VideoDetectionResponse(
        request_id=request_id,
        filename=file.filename,
        total_frames=len(frame_results),
        total_detections=total_dets,
        avg_inference_ms=round(avg_ms, 2),
        frames=frame_results,
    )


# ─── Internal Helpers ─────────────────────────────────────────────────────────

def _validate_file_size(file: UploadFile) -> None:
    """Raise 413 if the uploaded file exceeds the size limit."""
    # Content-Length header check (best-effort)
    if hasattr(file, "size") and file.size and file.size > MAX_FILE_B:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum allowed: {MAX_FILE_MB} MB.",
        )


def _decode_image(contents: bytes, filename: str) -> np.ndarray:
    """Decode bytes to BGR numpy array."""
    arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cannot decode image: {filename}",
        )
    return img


def _parse_detections(results) -> List[Dict[str, Any]]:
    """Extract detection dicts from Ultralytics results."""
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id  = int(box.cls.item())
            conf_v  = float(box.conf.item())
            xyxy    = [round(v, 2) for v in box.xyxy[0].tolist()]
            cls_name = (CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES)
                        else result.names.get(cls_id, str(cls_id)))
            detections.append({
                "class_id":   cls_id,
                "class_name": cls_name,
                "confidence": round(conf_v, 4),
                "bbox":       xyxy,
            })
    return detections


def _annotate_image(img: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw detection boxes onto a BGR image."""
    boxes  = [d["bbox"] for d in detections]
    labels = [d["class_name"] for d in detections]
    confs  = [d["confidence"] for d in detections]
    colors = [COLORS[d["class_id"] % len(COLORS)] for d in detections]
    return draw_detections(img, boxes, labels, confs, colors)


def _process_video_frames(
    video_path: Path,
    conf: float,
    iou: float,
    frame_skip: int,
    max_frames: int,
) -> List[Dict]:
    """Run detection on video frames and return per-frame results."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot open video file.",
        )

    model = get_model()
    frame_results = []
    frame_idx = 0
    processed = 0

    while processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % max(1, frame_skip) == 0:
            t0 = time.perf_counter()
            results = model.predict(frame, conf=conf, iou=iou, verbose=False)
            inf_ms  = (time.perf_counter() - t0) * 1000

            dets = _parse_detections(results)
            frame_results.append({
                "frame":        frame_idx,
                "detections":   dets,
                "inference_ms": round(inf_ms, 2),
            })
            processed += 1

        frame_idx += 1

    cap.release()
    return frame_results


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "deployment.api.main:app",
        host=deploy_cfg.get("host", "0.0.0.0"),
        port=deploy_cfg.get("port", 8000),
        workers=deploy_cfg.get("workers", 1),
        reload=deploy_cfg.get("reload", False),
    )
