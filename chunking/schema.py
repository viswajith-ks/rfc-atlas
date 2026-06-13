"""Data contracts and routing schemas for the LanceDB vector database."""

from typing import Any, TypedDict

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


class ChunkRecord(TypedDict):
    """Schema defining a structured chunk prior to LanceDB ingestion."""

    chunk_id: str
    rfc_number: str
    block_type: str
    table_route: str
    hierarchy_path: str
    text_payload: str
    sourcecode_type: str | None
    parsing_confidence: float
    normative_statements: list[dict[str, Any]]
    rfc_title: str | None
    status: str | None
