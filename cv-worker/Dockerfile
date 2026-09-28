# Build context: repo root (needs cv-worker/ and hubclient/).
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1

WORKDIR /app
# cv-worker depends on ../hubclient.
COPY hubclient /hubclient
COPY cv-worker/pyproject.toml cv-worker/uv.lock ./
RUN uv sync --frozen --no-dev --extra yolo --extra export --no-install-project
ENV PATH=/app/.venv/bin:$PATH
# ultralytics pulls in full opencv-python, which needs libGL.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# YOLO as an INT8 OpenVINO IR. Outside /app so a dev bind mount can't hide it.
RUN mkdir -p /opt/models && cd /opt/models \
 && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='openvino', int8=True, data='coco128.yaml')" \
 && rm -rf yolov8n.pt /root/.config/Ultralytics /app/datasets /opt/models/datasets

# The identity classifier is not baked in: it comes from the shared read-only
# models/classifier volume (CLASSIFIER_DIR), so switching models needs no rebuild.
COPY cv-worker/cv_worker ./cv_worker
RUN uv sync --frozen --no-dev --extra yolo --extra export
CMD ["cv-worker"]
