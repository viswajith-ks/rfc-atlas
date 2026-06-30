"""Data contracts and routing schemas for the LanceDB vector database."""

from typing import Literal

from pydantic import BaseModel, Field

from rfc_atlas.normalization.schema import (
    BlockType,
    NormativeStatement,
    SourcecodeFormat,
)

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


class ChunkRecord(BaseModel):
    """Final embedding-ready text chunk with fully denormalized relational metadata."""

    chunk_id: str
    rfc_number: int
    rfc_title: str | None = None
    status: str | None = None

    rfc_year: int | None = Field(
        default=None, description="Flattened publication year for vector filtering."
    )
    rfc_month: int | None = Field(
        default=None, description="Flattened publication month."
    )
    stream: str | None = None
    obsoletes: list[int] = Field(default=[])
    updated_by: list[int] = Field(default=[])

    block_type: BlockType
    table_route: LanceTableRoute
    hierarchy_path: str
    text_payload: str
    sourcecode_type: SourcecodeFormat | None = None
    parsing_confidence: float = Field(ge=0.0, le=1.0)
    normative_statements: list[NormativeStatement] = Field(default=[])
