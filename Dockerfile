FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/root/.local/bin:$PATH" \
    OLMOCR_WORKSPACE=/app/localworkspace \
    CUDA_VISIBLE_DEVICES=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        make \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Copy dependency files first to leverage layer caching
COPY pyproject.toml uv.lock ./

# uv reads requires-python = ">=3.13" from pyproject.toml and manages the interpreter
RUN uv sync --frozen

# Copy the rest of the source (filtered by .dockerignore)
COPY . .

RUN chmod +x /app/entrypoint.sh

RUN mkdir -p /app/sec_data /app/localworkspace

VOLUME ["/app/sec_data", "/app/localworkspace"]

EXPOSE 8000 8081

ENTRYPOINT ["/app/entrypoint.sh"]
