# 🚀 Deployment Guide

Complete guide for deploying the Autonomous Obstacle Detection system.

---

## 1. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run app.py

# Run FastAPI (separate terminal)
uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 2. Streamlit Community Cloud (Free)

### Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit dashboard"
   git push origin main
   ```

2. **Deploy**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **New app**
   - Repository: `yourname/autonomous-obstacle-detection-yolo`
   - Branch: `main`
   - Main file path: `app.py`
   - Click **Deploy!**

3. **Notes for cloud deployment**
   - `packages.txt` installs system deps (`libgl1`, `ffmpeg`, etc.)
   - Webcam tab is **disabled** on cloud (no camera access)
   - Upload file size limit: 200 MB (set in `.streamlit/config.toml`)
   - Model weights must be accessible — either committed (if < 100 MB) or downloaded on startup

### Auto-download weights on startup

Add to top of `app.py` if weights are not in the repo:
```python
import gdown
weights_path = "models/weights/best.pt"
if not Path(weights_path).exists():
    Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
    gdown.download(
        "https://drive.google.com/file/d/YOUR_FILE_ID/view",
        weights_path, fuzzy=True
    )
```

---

## 3. Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Streamlit**
   - Visibility: Public

2. Clone and push:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USER/obstacle-detection
   cp -r . obstacle-detection/
   # Copy huggingface_spaces_README.md → README.md
   cp huggingface_spaces_README.md obstacle-detection/README.md
   cd obstacle-detection
   git add . && git commit -m "Initial deploy"
   git push
   ```

3. HuggingFace will auto-build and deploy. Monitor at:
   `https://huggingface.co/spaces/YOUR_USER/obstacle-detection`

---

## 4. Docker (Self-Hosted)

### Build and Run API

```bash
# Build
docker build -t obstacle-detection:latest .

# Run with GPU
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    obstacle-detection:latest

# Run without GPU
docker run -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    obstacle-detection:latest
```

### Docker Compose (API + Training)

```bash
# Start API
docker compose up api

# Run training
docker compose --profile training up train

# View logs
docker compose logs -f api
```

### Add Streamlit to Docker

Add to `docker-compose.yml`:
```yaml
  streamlit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
    command: streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

---

## 5. Cloud Platforms (AWS / GCP / Azure)

### AWS ECS (Fargate)

```bash
# Tag and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker tag obstacle-detection:latest ACCOUNT.dkr.ecr.REGION.amazonaws.com/obstacle-detection:latest
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/obstacle-detection:latest

# Create ECS task definition and service via console or CDK
```

### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/obstacle-detection

# Deploy
gcloud run deploy obstacle-detection \
    --image gcr.io/PROJECT_ID/obstacle-detection \
    --port 8000 \
    --memory 4Gi \
    --cpu 2 \
    --allow-unauthenticated
```

### Azure Container Instances

```bash
az container create \
    --resource-group myResourceGroup \
    --name obstacle-detection \
    --image your-registry/obstacle-detection:latest \
    --ports 8000 \
    --memory 4 \
    --cpu 2
```

---

## 6. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WEIGHTS_PATH` | `models/weights/best.pt` | Path to YOLO weights |
| `CONF_THRESHOLD` | `0.35` | Default confidence |
| `IOU_THRESHOLD` | `0.45` | Default NMS IoU |
| `API_PORT` | `8000` | FastAPI port |
| `STREAMLIT_PORT` | `8501` | Streamlit port |

---

## 7. Performance Tips

- Use **FP16** (`--half`) for 2× faster GPU inference
- Export to **TensorRT** for 3–4× speedup on NVIDIA GPUs
- Use `frame_skip=2` or `3` for video to trade accuracy for speed
- Set `cache=ram` in training config if RAM allows for faster training
