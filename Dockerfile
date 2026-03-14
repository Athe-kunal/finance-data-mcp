# syntax=docker/dockerfile:1.7

# Use CUDA runtime (not devel) to significantly reduce base image size.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:$PATH" \
    OLMOCR_WORKSPACE=/app/localworkspace \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv

# Keep system packages minimal and avoid recommended extras.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    make \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install browser binaries by default for SEC scraping workflows.
ARG INSTALL_PLAYWRIGHT_BROWSER=1

# Install only runtime dependencies into a dedicated venv and aggressively clean caches.
RUN uv sync --frozen --no-dev \
    && if [ "$INSTALL_PLAYWRIGHT_BROWSER" = "1" ]; then uv run playwright install chromium; fi \
    && rm -rf /root/.cache /tmp/*

COPY . .

RUN chmod +x /app/entrypoint.sh \
    && mkdir -p /app/sec_data /app/localworkspace

VOLUME ["/app/sec_data", "/app/localworkspace"]

EXPOSE 8000 8081

ENTRYPOINT ["/app/entrypoint.sh"]
