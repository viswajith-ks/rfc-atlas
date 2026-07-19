"""Pydantic schemas for the RFC Atlas FastAPI boundary.

Enforces strict typing for incoming client requests, ensuring compatibility
with the frontend.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message within a chat conversation history."""

    role: Literal["user", "assistant", "system", "data"] = Field(
        description="The author or type of the message."
    )
    content: str = Field(description="The textual content of the message.")


class ChatRequest(BaseModel):
    """The incoming payload from the frontend."""

    messages: list[ChatMessage] = Field(
        description="The conversation history. The final message is the active query.",
        min_length=1,
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="The number of context chunks to retrieve from LanceDB.",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="The sampling randomness for the LLM synthesis.",
    )
