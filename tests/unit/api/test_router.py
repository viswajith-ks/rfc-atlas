from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rfc_atlas.api.router import router
from rfc_atlas.utils.exceptions import SynthesisError


@pytest.fixture
def app_with_mock_engine() -> FastAPI:
    """Creates a lightweight FastAPI instance with a mocked synthesis engine.

    Returns:
        FastAPI: The test application instance.
    """
    app = FastAPI()
    app.include_router(router)

    mock_engine = MagicMock()
    app.state.engine = mock_engine
    return app


def test_chat_stream_success(app_with_mock_engine: FastAPI) -> None:
    mock_engine = app_with_mock_engine.state.engine

    def fake_stream(
        query_text: str, top_k: int, temperature: float
    ) -> Generator[str, None, None]:
        _ = (query_text, top_k, temperature)
        yield "Chunk 1 "
        yield "Chunk 2"

    mock_engine.stream_query.side_effect = fake_stream

    client = TestClient(app_with_mock_engine)
    response = client.post(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "How does TCP work?"}],
            "top_k": 5,
            "temperature": 0.1,
        },
    )

    assert response.status_code == 200  # pyright: ignore[reportUnknownMemberType]
    assert response.text == "Chunk 1 Chunk 2"  # pyright: ignore[reportUnknownMemberType]
    mock_engine.stream_query.assert_called_once_with(
        query_text="How does TCP work?", top_k=5, temperature=0.1
    )


def test_chat_stream_missing_engine() -> None:
    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)
    response = client.post(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 503  # pyright: ignore[reportUnknownMemberType]
    assert response.json()["detail"] == "Synthesis Engine unavailable."  # pyright: ignore[reportUnknownMemberType]


def test_chat_stream_no_user_message(app_with_mock_engine: FastAPI) -> None:
    client = TestClient(app_with_mock_engine)
    response = client.post(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        "/api/chat",
        json={"messages": [{"role": "system", "content": "You are a helpful bot."}]},
    )

    assert response.status_code == 400  # pyright: ignore[reportUnknownMemberType]
    assert "No user message found" in response.json()["detail"]  # pyright: ignore[reportUnknownMemberType]


def test_chat_stream_synthesis_error(app_with_mock_engine: FastAPI) -> None:
    mock_engine = app_with_mock_engine.state.engine

    def fake_stream(
        query_text: str, top_k: int, temperature: float
    ) -> Generator[str, None, None]:
        _ = (query_text, top_k, temperature)
        yield "Starting stream... "
        err_msg = "API rate limit exceeded"
        raise SynthesisError(err_msg)

    mock_engine.stream_query.side_effect = fake_stream

    client = TestClient(app_with_mock_engine)
    response = client.post(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Crash it."}]},
    )

    assert response.status_code == 200  # pyright: ignore[reportUnknownMemberType]
    assert "Starting stream..." in response.text  # pyright: ignore[reportUnknownMemberType]
    assert (
        "[ERROR: Synthesis failed - LLM Synthesis failed: API rate limit exceeded]"
        in response.text  # pyright: ignore[reportUnknownMemberType]
    )
