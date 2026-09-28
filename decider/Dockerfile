# Build context: repo root (needs decider/ and hubclient/).
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app
COPY hubclient /hubclient
COPY decider/pyproject.toml decider/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY decider/decider ./decider
RUN uv sync --frozen --no-dev
ENV PATH=/app/.venv/bin:$PATH
CMD ["decider", "--config", "/config.yaml"]
