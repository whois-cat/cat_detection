# Build context is the repo root (see docker-compose.yml) so we can share the
# single source of truth for the index logic: detector/recordings_index.py.
FROM python:3.11-slim
WORKDIR /app
COPY detector/recordings_index.py .
COPY indexer/indexer.py .
CMD ["python", "-u", "indexer.py"]
