"""FastAPI application factory and lifecycle management.

Configures CORS, sets up Hugging Face health checks, and initializes the heavy
ML synthesis pipeline exactly once during server boot.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rfc_atlas.api.router import router
from rfc_atlas.synthesis.orchestrator import SynthesisOrchestrator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages the startup and shutdown lifecycle of the FastAPI server.

    Ensures the stateless Synthesis Orchestrator (and its underlying Retrieval Engine)
    is instantiated and its models are loaded into memory before accepting HTTP traffic.

    Args:
        app (FastAPI): The running FastAPI application instance.

    Yields:
        None
    """
    logger.info("🚀 Booting RFC Atlas API Lifespan...")

    try:
        engine = SynthesisOrchestrator()
        app.state.engine = engine
        logger.info("✅ Synthesis Engine successfully mounted to app state.")

        yield

    except Exception as e:
        logger.critical("❌ FATAL: API Lifespan encountered an error: %s", e)
        raise
    finally:
        logger.info("🛑 RFC Atlas API shutting down gracefully...")
        app.state.engine = None


def create_app() -> FastAPI:
    """Initializes and configures the FastAPI application boundary.

    Returns:
        FastAPI: The strictly configured web application instance.
    """
    app = FastAPI(
        title="RFC Atlas API",
        description="Stateless RAG API boundary for the RFC Atlas Intelligence System.",
        version="0.1.0",
        lifespan=app_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/health", tags=["Lifecycle"])
    async def health_check() -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Authoritative health check for Hugging Face Spaces lifecycle management.

        Returns:
            JSONResponse: A simple 200 OK status payload.
        """
        return JSONResponse(content={"status": "healthy", "service": "rfc-atlas-api"})

    logger.info("FastAPI application factory configured successfully.")
    return app
