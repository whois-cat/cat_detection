FROM python:3.11-slim

# uv: drop-in for pip with much faster installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Pre-download default YOLO weights and pre-export an INT8 OpenVINO IR.
# OpenVINO INT8 runs ~3-4× faster than torch CPU on Intel hardware (and is
# still significantly faster on AMD CPUs via the OpenVINO runtime).
# Calibration uses Ultralytics' coco128 (auto-downloaded, ~7 MB) — fine for
# the cat class we care about.
# Output goes to /opt/models/ (NOT /app/) because in dev mode the compose
# override bind-mounts ./detector over /app, which would mask anything baked
# into /app at runtime. YOLO_WEIGHTS points at the absolute /opt/models path.
RUN mkdir -p /opt/models \
 && cd /opt/models \
 && python -c "from ultralytics import YOLO; \
    m = YOLO('yolov8n.pt'); \
    m.export(format='openvino', int8=True, data='coco128.yaml')"

COPY storage.py detectors.py main.py ./

CMD ["python", "-u", "main.py"]
