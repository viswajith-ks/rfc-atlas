"""Data contracts and routing schemas for the LanceDB vector database."""

from typing import Literal

from pydantic import BaseModel, Field

from normalization.schema import BlockType, NormativeKeyword

LanceTableRoute = Literal[
    "prose", "security", "references", "abnf", "sourcecode", "artwork", "table"
]

TABLE_ROUTING_MAP: dict[str, LanceTableRoute] = {
    "paragraph": "prose",
    "list": "prose",
    "security": "security",
    "references": "references",
    "abnf": "abnf",
    "sourcecode": "sourcecode",
    "artwork": "artwork",
    "table": "table",
}


class ChunkNormativeStatement(BaseModel):
    """Schema for normative statements preserved inside a chunk."""

    keyword: NormativeKeyword
    statement_text: str
    actor: str | None = None
    referenced_rfcs: list[int] = Field(default=[])


class ChunkRecord(BaseModel):
    """Final embedding-ready text chunk with fully denormalized relational metadata."""

    chunk_id: str
    rfc_number: str
    rfc_title: str | None = None
    status: str | None = None

    rfc_year: int | None = None
    stream: str | None = None
    obsoletes: list[int] = Field(default=[])
    updated_by: list[int] = Field(default=[])

    block_type: BlockType
    table_route: str
    hierarchy_path: str
    text_payload: str
    sourcecode_type: str | None = None
    parsing_confidence: float = Field(ge=0.0, le=1.0)
    normative_statements: list[ChunkNormativeStatement] = Field(default=[])
