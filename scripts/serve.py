"""CLI execution script for booting the FastAPI server via Uvicorn.

Serves the stateless RFC Atlas HTTP boundary.
"""

import argparse
import logging
import sys

import uvicorn

logger = logging.getLogger(__name__)


def main() -> None:
    """Parses arguments and starts the Uvicorn ASGI server."""
    parser = argparse.ArgumentParser(description="RFC Atlas API Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",  # noqa: S104
        help="Bind socket to this host.",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Bind socket to this port."
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development."
    )
    args = parser.parse_args()

    print("=========================================================")
    print(f" 🚀 BOOTING RFC ATLAS API SERVER (Port {args.port})")
    print("=========================================================")

    try:
        uvicorn.run(
            "rfc_atlas.api.factory:create_app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            factory=True,
        )
    except Exception:
        logger.exception("FATAL: API Server crashed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
