FROM python:3.11-slim

# uv: drop-in for pip with much faster installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# torchvision is needed only for the EfficientNet export step below.
# ultralytics already pulled torch; this adds ~100 MB for the model arch.
RUN uv pip install --system --no-cache torchvision

# ── YOLO: export INT8 OpenVINO IR ──────────────────────────────────────────
# Output to /opt/models/ (NOT /app/) because dev-mode compose bind-mounts
# ./detector over /app, which would mask anything baked into /app at runtime.
RUN mkdir -p /opt/models \
 && cd /opt/models \
 && python -c "from ultralytics import YOLO; \
    m = YOLO('yolov8n.pt'); \
    m.export(format='openvino', int8=True, data='coco128.yaml')"

# ── Classifier: PT → OpenVINO IR + classes.json ────────────────────────────
COPY models/cat_classifier.pt /app/models/cat_classifier.pt
# classifier.py is copied here (not with the rest below) because the export
# step's parity gate imports its runtime _preprocess to prove the exported IR
# matches torch on the real preprocessing.
COPY export_classifier.py classifier.py ./
RUN python export_classifier.py \
    --pt /app/models/cat_classifier.pt \
    --out /opt/models/cat_classifier_openvino/

COPY storage.py detectors.py main.py ./

CMD ["python", "-u", "main.py"]
