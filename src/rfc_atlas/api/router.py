"""FastAPI routing endpoints for the RFC Atlas.

Handles incoming chat requests, maps them to the Synthesis Engine, and
streams the generative response back to the client.
"""

import logging
from collections.abc import Generator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from rfc_atlas.api.schema import ChatRequest
from rfc_atlas.utils.exceptions import SynthesisError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.post("/chat", tags=["Synthesis"])
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    """Streaming endpoint for the Next.js frontend (Vercel AI SDK).

    Extracts the latest user query from the conversation history, processes it
    through the local LanceDB RAG pipeline, and streams the Gemini response back.

    Args:
        request (Request): The incoming HTTP request.
        payload (ChatRequest): The validated Pydantic payload from the client.

    Returns:
        StreamingResponse: A text/plain SSE stream yielding the generative answer.

    Raises:
        HTTPException: If the Synthesis Engine is unavailable or the query is missing.
    """
    engine = getattr(request.app.state, "engine", None)
    if not engine:
        logger.critical("Synthesis Engine is not initialized in the app state.")
        raise HTTPException(status_code=503, detail="Synthesis Engine unavailable.")

    last_user_message = next(
        (msg.content for msg in reversed(payload.messages) if msg.role == "user"),
        None,
    )

    if not last_user_message:
        raise HTTPException(
            status_code=400,
            detail="No user message found in the conversation history.",
        )

    def stream_generator() -> Generator[str, None, None]:
        try:
            yield from engine.stream_query(
                query_text=last_user_message,
                top_k=payload.top_k,
                temperature=payload.temperature,
            )
        except SynthesisError as e:
            logger.exception("Synthesis stream interrupted by Gemini API")
            yield f"\n\n[ERROR: Synthesis failed - {e}]"
        except Exception:
            logger.exception("Unexpected error during stream generation.")
            yield "\n\n[FATAL ERROR: Internal system failure.]"

    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
