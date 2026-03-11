# ============================================================
# Autonomous Vehicle Obstacle Detection — Production Dockerfile
# ============================================================
# Base image: CUDA 12.1 + cuDNN 8 + Python 3.10
# ============================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ── System packages ──────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3.10-venv \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        ffmpeg \
        wget \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ── Working directory ────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project ─────────────────────────────────────────────
COPY . .

# Create runtime directories
RUN mkdir -p data/raw data/processed data/annotations \
             models/weights models/checkpoints \
             logs runs

# ── Environment ──────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Expose API port ──────────────────────────────────────────
EXPOSE 8000

# ── Default command: start FastAPI server ────────────────────
CMD ["uvicorn", "deployment.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
