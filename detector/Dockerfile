FROM python:3.11-slim

# uv: drop-in for pip with much faster installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# ── YOLO: export INT8 OpenVINO IR ──────────────────────────────────────────
# Output to /opt/models/ (NOT /app/) because dev-mode compose bind-mounts
# ./detector over /app, which would mask anything baked into /app at runtime.
RUN mkdir -p /opt/models \
 && cd /opt/models \
 && python -c "from ultralytics import YOLO; \
    m = YOLO('yolov8n.pt'); \
    m.export(format='openvino', int8=True, data='coco128.yaml')"

# ── Classifier ─────────────────────────────────────────────────────────────
# The runtime classifier model is NOT baked into the image. It is delivered via
# the shared read-only volume at /opt/models/classifier
# (CLASSIFIER_WEIGHTS=/opt/models/classifier/current), switched by
# `just classifier-promote` + `just classifier-restart` — no image rebuild to
# change the model. export_classifier.py stays in the image so promotion (run
# inside this container) can export a checkpoint to the OpenVINO runtime format;
# its parity gate imports classifier.py's runtime _preprocess.
COPY export_classifier.py classifier.py ./

COPY storage.py recordings_index.py detectors.py bowl_monitor.py main.py ./

CMD ["python", "-u", "main.py"]
