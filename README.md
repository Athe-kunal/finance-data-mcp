# olmocr-sec-filings

## Configuration

Settings are loaded via Pydantic Settings from environment variables or a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `SEC_API_ORGANIZATION` | Organization name for SEC API User-Agent | `Your-Organization` |
| `SEC_API_EMAIL` | Contact email for SEC API User-Agent | `your-email@example.com` |
| `OLMOCR_SERVER` | vLLM server URL for olmOCR | `http://localhost:8000/v1` |
| `OLMOCR_MODEL` | Model name for olmOCR | `allenai/olmOCR-2-7B-1025-FP8` |
| `OLMOCR_WORKSPACE` | Workspace directory for OCR output | `./localworkspace` |

Import the singleton `settings` instance:

```python
from settings import settings

company = settings.sec_api_organization
email = settings.sec_api_email
server = settings.olmocr_server
```

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

Fetch SEC filings:
```bash
uv run python -m filings.sec_data --ticker AMZN --year 2025
```

Run OCR pipeline:
```bash
uv run python ocr/olmocr_pipeline.py --pdf-dir sec_data/AMZN-2025
```
