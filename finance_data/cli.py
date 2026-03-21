"""CLI entrypoint for serving the finance_data FastAPI app."""

import uvicorn


def main() -> None:
    """Start the finance_data API server."""
    uvicorn.run("server:app", host="0.0.0.0", port=8081)


if __name__ == "__main__":
    main()
