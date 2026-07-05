from typing import Any, get_args

import pytest
from pydantic import ValidationError

from rfc_atlas.chunking.schema import (
    TABLE_ROUTING_MAP,
    ChunkRecord,
    LanceTableRoute,
    NormativeStatement,
)
from rfc_atlas.normalization.schema import BlockType

EXPECTED_BLOCKS: frozenset[str] = frozenset(get_args(BlockType))
VALID_TABLES: frozenset[str] = frozenset(get_args(LanceTableRoute))


def test_table_routing_map_completeness() -> None:
    for block in EXPECTED_BLOCKS:
        assert block in TABLE_ROUTING_MAP, f"Missing route for block type: {block}"


def test_table_routing_destinations_are_valid() -> None:
    for block_type, destination in TABLE_ROUTING_MAP.items():
        assert destination in VALID_TABLES, (
            f"Invalid destination table '{destination}' for block '{block_type}'"
        )


def test_chunk_record_validation_success() -> None:
    record = ChunkRecord(
        chunk_id="1234-sec1-01",
        rfc_number=1234,
        block_type="paragraph",
        table_route="prose",
        hierarchy_path="1. Introduction",
        text_payload="This is a test chunk.",
        parsing_confidence=0.95,
        normative_statements=[
            NormativeStatement(keyword="MUST", statement_text="This is a test chunk.")
        ],
    )

    assert record.chunk_id == "1234-sec1-01"
    assert record.sourcecode_type is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_id": 12345, "text_payload": ["Not", "a", "string"]},
        {"parsing_confidence": 1.5},
        {"parsing_confidence": -0.1},
    ],
    ids=["bad_types", "confidence_too_high", "confidence_too_low"],
)
def test_chunk_record_validation_failures(overrides: dict[str, Any]) -> None:
    base_data: dict[str, Any] = {
        "chunk_id": "1234-sec1-01",
        "rfc_number": 1234,
        "block_type": "paragraph",
        "table_route": "prose",
        "hierarchy_path": "1. Introduction",
        "text_payload": "This is a test chunk.",
        "parsing_confidence": 0.95,
    }
    base_data.update(overrides)

    with pytest.raises(ValidationError):
        ChunkRecord(**base_data)
