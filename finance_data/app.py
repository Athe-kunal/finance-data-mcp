"""FastAPI app accessor.

Importing this module still avoids importing `server` at package import time.
"""


def get_app():
    """Return the FastAPI app defined in server.py."""
    from server import app

    return app
