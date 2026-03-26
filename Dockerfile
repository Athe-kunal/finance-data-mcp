# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:$PATH" \
    OLMOCR_WORKSPACE=/app/localworkspace \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv

# Keep system packages minimal and avoid recommended extras.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --group ocr-md

RUN uv run playwright install chromium --with-deps

COPY . .

RUN chmod +x /app/entrypoint.sh \
    && mkdir -p /app/sec_data /app/localworkspace /app/earnings_transcripts_data

VOLUME ["/app/sec_data", "/app/localworkspace", "/app/earnings_transcripts_data"]

EXPOSE 8000 8081

ENTRYPOINT ["/app/entrypoint.sh"]
