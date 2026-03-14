FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/root/.local/bin:$PATH" \
    OLMOCR_WORKSPACE=/app/localworkspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    make \
    ca-certificates \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Increase ulimit to avoid "Too many open files" during bytecode compilation of many packages
RUN ulimit -n 65536 && uv sync --frozen && uv run playwright install chromium --with-deps

COPY . .

RUN chmod +x /app/entrypoint.sh

RUN mkdir -p /app/sec_data /app/localworkspace

VOLUME ["/app/sec_data", "/app/localworkspace"]

EXPOSE 8000 8081

ENTRYPOINT ["/app/entrypoint.sh"]
