"""CLI entrypoint for serving the finance_data FastAPI app."""

import uvicorn

from finance_data.settings import sec_settings


def main() -> None:
    """Start the finance_data API server."""
    uvicorn.run(
        "server:app",
        host=sec_settings.api_host,
        port=sec_settings.api_port,
    )


if __name__ == "__main__":
    main()
