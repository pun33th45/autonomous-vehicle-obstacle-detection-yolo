"""
test_api.py
-----------
Integration tests for the FastAPI obstacle detection API.

Tests are designed to work without a real model by patching the YOLO
prediction call.
"""

import io
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    """Return a mock YOLO model that returns empty detections."""
    mock = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = None
    mock.predict.return_value = [mock_result]
    return mock


@pytest.fixture
def client(mock_model, tmp_path, monkeypatch):
    """Create a TestClient with a mocked YOLO model and fake weights file."""
    # Create a fake weights file so the API doesn't raise FileNotFoundError
    fake_weights = tmp_path / "best.pt"
    fake_weights.write_bytes(b"fake")

    # Patch the weights path and model loader
    monkeypatch.setenv("WEIGHTS_PATH", str(fake_weights))

    with patch("deployment.api.main.WEIGHTS_PATH", str(fake_weights)), \
         patch("deployment.api.main._model", mock_model), \
         patch("deployment.api.main.get_model", return_value=mock_model):
        from deployment.api.main import app
        yield TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Generate a small test image as JPEG bytes."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (50, 50), (0, 255, 0), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ─── Health Check ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "timestamp" in data


# ─── Model Info ──────────────────────────────────────────────────────────────

class TestModelInfoEndpoint:
    def test_returns_class_names(self, client):
        data = client.get("/model-info").json()
        assert "class_names" in data
        assert len(data["class_names"]) == 8
        assert "car" in data["class_names"]

    def test_returns_num_classes(self, client):
        data = client.get("/model-info").json()
        assert data["num_classes"] == 8


# ─── Detect Image ─────────────────────────────────────────────────────────────

class TestDetectImageEndpoint:
    def test_valid_image_returns_200(self, client, sample_image_bytes):
        resp = client.post(
            "/detect-image",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_response_schema(self, client, sample_image_bytes):
        data = client.post(
            "/detect-image",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        ).json()
        assert "request_id" in data
        assert "detections" in data
        assert "num_detections" in data
        assert "inference_ms" in data
        assert isinstance(data["detections"], list)

    def test_invalid_file_raises_422(self, client):
        resp = client.post(
            "/detect-image",
            files={"file": ("bad.jpg", b"not_an_image", "image/jpeg")},
        )
        assert resp.status_code == 422


# ─── Detect Video ────────────────────────────────────────────────────────────

class TestDetectVideoEndpoint:
    @pytest.fixture
    def tiny_video_bytes(self, tmp_path):
        """Create a 10-frame dummy video."""
        path = tmp_path / "test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 10, (64, 64))
        for _ in range(10):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        return path.read_bytes()

    def test_video_endpoint_returns_200(self, client, tiny_video_bytes):
        resp = client.post(
            "/detect-video",
            files={"file": ("test.mp4", tiny_video_bytes, "video/mp4")},
        )
        assert resp.status_code == 200

    def test_video_response_has_frames(self, client, tiny_video_bytes):
        data = client.post(
            "/detect-video",
            files={"file": ("test.mp4", tiny_video_bytes, "video/mp4")},
        ).json()
        assert "frames" in data
        assert "total_frames" in data
