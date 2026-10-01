FROM python:3.12-slim

# System libraries Docling needs for PDF and image handling
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# Dependencies first so this layer is cached; no dev group (no pytest or ruff).
# uv's download cache lives in a build cache mount, not in the image.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models/huggingface \
    FASTEMBED_CACHE_PATH=/models/fastembed \
    DOCLING_ARTIFACTS_PATH=/models/docling

# Download every local model now (about 1.5 GB), so the server never downloads at start-up.
RUN docling-tools models download layout tableformer rapidocr -o /models/docling \
    && python scripts/download_models.py
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
CMD ["python", "-m", "rag_engine.mcp_server"]
