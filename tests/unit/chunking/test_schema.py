"""Unit tests for chunking data contracts and LanceDB routing logic."""

from typing import get_args

import pytest
from pydantic import ValidationError

from chunking.schema import (
    TABLE_ROUTING_MAP,
    ChunkNormativeStatement,
    ChunkRecord,
    LanceTableRoute,
)
from normalization.schema import BlockType

EXPECTED_BLOCKS: frozenset[str] = frozenset(get_args(BlockType))
VALID_TABLES: frozenset[str] = frozenset(get_args(LanceTableRoute))


def test_table_routing_map_completeness() -> None:
    """Ensure all normalization BlockTypes have a mapped LanceDB destination."""
    for block in EXPECTED_BLOCKS:
        assert block in TABLE_ROUTING_MAP, f"Missing route for block type: {block}"


def test_table_routing_destinations_are_valid() -> None:
    """Ensure we aren't routing blocks to non-existent LanceDB tables."""
    for block_type, destination in TABLE_ROUTING_MAP.items():
        assert destination in VALID_TABLES, (
            f"Invalid destination table '{destination}' for block '{block_type}'"
        )


def test_chunk_record_validation_success() -> None:
    """Verify that a valid ChunkRecord parses correctly with strictly typed fields."""
    record = ChunkRecord(
        chunk_id="1234-sec1-01",
        rfc_number="1234",
        block_type="paragraph",
        table_route="prose",
        hierarchy_path="1. Introduction",
        text_payload="This is a test chunk.",
        parsing_confidence=0.95,
        normative_statements=[
            ChunkNormativeStatement(
                keyword="MUST", statement_text="This is a test chunk."
            )
        ],
    )

    assert record.chunk_id == "1234-sec1-01"
    assert record.sourcecode_type is None


def test_chunk_record_validation_failure_bad_types() -> None:
    """Verify that Pydantic rejects invalid data types during chunk instantiation."""
    with pytest.raises(ValidationError):
        ChunkRecord(
            chunk_id=12345,  # type: ignore
            rfc_number="1234",
            block_type="paragraph",
            table_route="prose",
            hierarchy_path="1. Introduction",
            text_payload=["Not", "a", "string"],  # type: ignore
            parsing_confidence=0.95,
        )


def test_chunk_record_validation_failure_bad_confidence() -> None:
    """Verify that parsing_confidence strictly bounds between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        ChunkRecord(
            chunk_id="1234-sec1-01",
            rfc_number="1234",
            block_type="paragraph",
            table_route="prose",
            hierarchy_path="1. Introduction",
            text_payload="Testing confidence bound.",
            parsing_confidence=1.5,
        )
