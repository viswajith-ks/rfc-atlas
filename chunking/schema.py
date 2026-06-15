"""Data contracts and routing schemas for the LanceDB vector database."""

from pydantic import BaseModel, Field

TABLE_ROUTING_MAP: dict[str, str] = {
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
    """Schema defining a structured chunk prior to LanceDB ingestion."""

    chunk_id: str
    rfc_number: str
    block_type: str
    table_route: str
    hierarchy_path: str
    text_payload: str

    sourcecode_type: str | None = None
    rfc_title: str | None = None
    status: str | None = None

    parsing_confidence: float = Field(ge=0.0, le=1.0)
    normative_statements: list[dict[str, str]] = Field(default=[])
