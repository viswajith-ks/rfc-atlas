import pytest
from pydantic import ValidationError

from rfc_atlas.api.schema import ChatMessage, ChatRequest


def test_valid_chat_request() -> None:
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="How does TCP work?")],
        top_k=15,
        temperature=0.5,
    )
    assert req.top_k == 15
    assert req.temperature == pytest.approx(0.5)
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"


def test_invalid_top_k() -> None:
    with pytest.raises(
        ValidationError, match="Input should be less than or equal to 50"
    ):
        ChatRequest(messages=[ChatMessage(role="user", content="Hi")], top_k=100)

    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 1"
    ):
        ChatRequest(messages=[ChatMessage(role="user", content="Hi")], top_k=0)


def test_invalid_temperature() -> None:
    # Use raw string (r"") and escape the dot (\.0) so regex doesn't interpret it as "any character"
    with pytest.raises(
        ValidationError, match="Input should be less than or equal to 1"
    ):
        ChatRequest(messages=[ChatMessage(role="user", content="Hi")], temperature=1.5)


def test_empty_messages() -> None:
    with pytest.raises(ValidationError, match="List should have at least 1 item"):
        ChatRequest(messages=[])
