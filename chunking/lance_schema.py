"""Strict PyArrow schema definitions and adapters for LanceDB columnar storage."""

import json
from typing import Any

import pyarrow as pa

from chunking.schema import ChunkRecord

VECTOR_DIMENSIONS: int = 256

LANCE_CHUNK_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string(), nullable=False),
    pa.field("rfc_number", pa.int32(), nullable=False),
    pa.field("rfc_title", pa.string(), nullable=True),
    pa.field("status", pa.string(), nullable=True),
    pa.field("rfc_year", pa.int16(), nullable=True),
    pa.field("rfc_month", pa.int8(), nullable=True),
    pa.field("stream", pa.string(), nullable=True),
    pa.field("obsoletes", pa.list_(pa.int32()), nullable=False),
    pa.field("updated_by", pa.list_(pa.int32()), nullable=False),
    pa.field("block_type", pa.string(), nullable=False),
    pa.field("table_route", pa.string(), nullable=False),
    pa.field("hierarchy_path", pa.string(), nullable=False),
    pa.field("text_payload", pa.string(), nullable=False),
    pa.field("sourcecode_type", pa.string(), nullable=True),
    pa.field("parsing_confidence", pa.float32(), nullable=False),
    pa.field("normative_statements_json", pa.string(), nullable=False),
    pa.field("vector", pa.list_(pa.float32(), VECTOR_DIMENSIONS), nullable=False),
])


def record_to_lance_row(record: ChunkRecord, vector: list[float]) -> dict[str, Any]:
    """Converts a validated Pydantic ChunkRecord into a PyArrow-compliant dictionary.

    Serializes complex nested Pydantic objects to safe JSON strings and enforces
    exact mathematical dimension constraints on the embedding array.

    Args:
        record (ChunkRecord): The pristine chunk record from Phase 2.
        vector (list[float]): The 256-dimensional float array emitted by Nomic.

    Returns:
        dict[str, Any]: A flat dictionary ready for zero-copy PyArrow ingestion.

    Raises:
        ValueError: If the vector length violates the 256-dimension contract.
    """
    if len(vector) != VECTOR_DIMENSIONS:
        msg = (
            f"Vector dimension mismatch for '{record.chunk_id}': "
            f"expected {VECTOR_DIMENSIONS}, got {len(vector)}"
        )
        raise ValueError(msg)

    statements_dump = [
        stmt.model_dump(mode="json") for stmt in record.normative_statements
    ]

    return {
        "chunk_id": record.chunk_id,
        "rfc_number": record.rfc_number,
        "rfc_title": record.rfc_title,
        "status": record.status,
        "rfc_year": record.rfc_year,
        "rfc_month": record.rfc_month,
        "stream": record.stream,
        "obsoletes": record.obsoletes,
        "updated_by": record.updated_by,
        "block_type": record.block_type,
        "table_route": record.table_route,
        "hierarchy_path": record.hierarchy_path,
        "text_payload": record.text_payload,
        "sourcecode_type": record.sourcecode_type,
        "parsing_confidence": record.parsing_confidence,
        "normative_statements_json": json.dumps(statements_dump),
        "vector": vector,
    }
