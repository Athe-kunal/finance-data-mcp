# SEC-filings-Markdown

## Configuration

Settings are loaded via Pydantic Settings from environment variables or a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `SEC_API_ORGANIZATION` | Organization name for SEC API User-Agent | `Your-Organization` |
| `SEC_API_EMAIL` | Contact email for SEC API User-Agent | `your-email@example.com` |
| `OLMOCR_SERVER` | vLLM server URL for olmOCR | `http://localhost:8000/v1` |
| `OLMOCR_MODEL` | Model name for olmOCR | `allenai/olmOCR-2-7B-1025-FP8` |
| `OLMOCR_WORKSPACE` | Workspace directory for OCR output | `./localworkspace` |

## Docker

### Build

```bash
docker build -t sec-filings-md .
```

The image now defaults to a smaller footprint by using the CUDA runtime base while still preinstalling Playwright Chromium for scraping.
If you want to skip Playwright browser installation (to reduce image size further), build with:

```bash
docker build --build-arg INSTALL_PLAYWRIGHT_BROWSER=0 -t sec-filings-md .
```

Or via Makefile:

```bash
make docker-build
```

### Run

```bash
GPU_DEVICE=${GPU_DEVICE:-3}
docker run --gpus device=${GPU_DEVICE} \
  -e SEC_API_ORGANIZATION="Your-Organization" \
  -e SEC_API_EMAIL="your-email@example.com" \
  -v ./sec_data:/app/sec_data \
  -v ./localworkspace:/app/localworkspace \
  -p 8081:8081 \
  sec-filings-md
```

Or via Makefile (build + run in one step):

```bash
make docker-start
```

Makefile overrides:

| Variable | Description | Default |
|----------|-------------|---------|
| `IMAGE_NAME` | Docker image name | `sec-filings-md` |
| `GPU_DEVICE` | GPU device index | `0` |
| `API_PORT` | Host port for API | `8081` |
| `SEC_API_ORGANIZATION` | SEC API User-Agent org | `Your-Organization` |
| `SEC_API_EMAIL` | SEC API contact email | `your-email@example.com` |

Example with overrides:

```bash
make docker-start GPU_DEVICE=3 SEC_API_EMAIL="you@example.com"
```

The two volumes persist data across container restarts:

| Volume | Container path | Purpose |
|--------|---------------|---------|
| `sec_data` | `/app/sec_data` | Downloaded SEC filing PDFs |
| `localworkspace` | `/app/localworkspace` | OCR workspace and output markdown |

Override the workspace path at runtime with `-e OLMOCR_WORKSPACE=/custom/path`.

## Installation

```bash
uv sync
playwright install chromium
```

## Usage

Start vLLM server:
```bash
make vllm-olmocr-serve
```

Benchmark vLLM with guidellm (start the vLLM server first, then in another terminal):
```bash
make guidellm-benchmark
```

Fetch SEC filings:
```bash
uv run python -m filings.sec_data --ticker AMZN --year 2025
```

Run OCR pipeline:
```bash
uv run python ocr/olmocr_pipeline.py --pdf-dir sec_data/AMZN-2025
```
