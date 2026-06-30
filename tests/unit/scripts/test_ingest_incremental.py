import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import lancedb
import numpy as np
import pyarrow as pa
import pytest

from scripts.ingest_incremental import (
    _get_existing_ids,  # pyright: ignore[reportPrivateUsage]
    _sync_single_table,  # pyright: ignore[reportPrivateUsage]
)
from vector_store.schema import LANCE_CHUNK_SCHEMA, VECTOR_DIMENSIONS, build_lance_table


@pytest.fixture
def mock_lancedb(tmp_path: Path) -> lancedb.DBConnection:
    db_dir = tmp_path / "lancedb"
    db = lancedb.connect(str(db_dir))

    records: list[dict[str, Any]] = [
        {
            "chunk_id": f"chunk-00{i}",
            "rfc_number": 1000,
            "block_type": "paragraph",
            "table_route": "prose",
            "hierarchy_path": "Root",
            "text_payload": f"Existing payload {i}",
            "parsing_confidence": 1.0,
            "obsoletes": [],
            "updated_by": [],
            "normative_statements": [],
        }
        for i in range(1, 3)  # Creates chunk-001 and chunk-002
    ]

    fake_vectors = np.random.rand(2, VECTOR_DIMENSIONS).astype(np.float32)
    vector_array = pa.FixedSizeListArray.from_arrays(
        pa.array(fake_vectors.ravel(), type=pa.float32()), VECTOR_DIMENSIONS
    )

    table = build_lance_table(records, vector_array)
    db.create_table("prose", data=table, schema=LANCE_CHUNK_SCHEMA)  # pyright: ignore[reportUnknownMemberType]

    return db


@pytest.fixture
def mock_jsonl_file(tmp_path: Path) -> Path:
    jsonl_path = tmp_path / "prose.jsonl"

    records: list[dict[str, Any]] = [
        {
            "chunk_id": "chunk-002",
            "rfc_number": 1000,
            "block_type": "paragraph",
            "table_route": "prose",
            "hierarchy_path": "Root",
            "text_payload": "Duplicate",
            "parsing_confidence": 1.0,
            "obsoletes": [],
            "updated_by": [],
            "normative_statements": [],
        },
        {
            "chunk_id": "chunk-003",
            "rfc_number": 1000,
            "block_type": "paragraph",
            "table_route": "prose",
            "hierarchy_path": "Root",
            "text_payload": "New chunk!",
            "parsing_confidence": 1.0,
            "obsoletes": [],
            "updated_by": [],
            "normative_statements": [],
        },
    ]

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return jsonl_path


def test_columnar_set_difference(mock_lancedb: lancedb.DBConnection) -> None:
    """Asserts that _get_existing_ids successfully extracts all IDs via PyArrow."""
    table = mock_lancedb.open_table("prose")
    existing_ids = _get_existing_ids(table)

    assert len(existing_ids) == 2
    assert "chunk-001" in existing_ids
    assert "chunk-002" in existing_ids


def test_sync_single_table_skips_duplicates(
    mock_lancedb: lancedb.DBConnection,
    mock_jsonl_file: Path,
) -> None:
    """Asserts that the sync logic drops duplicates and only embeds new chunks."""
    # 1. Setup the mocked SentenceTransformer instance directly
    mock_model_instance = MagicMock()

    # Nomic natively outputs 768 dimensions before we truncate it to 256.
    # We must mock it to return an array of shape (N, 768)
    def fake_encode(texts: list[str], **_: list[Any]) -> np.ndarray:
        return np.random.rand(len(texts), 768).astype(np.float32)

    mock_model_instance.encode.side_effect = fake_encode

    # 2. Run the sync (which will pass the mock jsonl containing chunk 2 and 3)
    new_chunks, _ = _sync_single_table(
        mock_jsonl_file, mock_lancedb, mock_model_instance
    )

    # 3. Assertions
    # It should have skipped chunk-002 and ONLY processed chunk-003
    assert new_chunks == 1

    # Verify the database now contains exactly 3 rows (chunk 1, 2, 3)
    table = mock_lancedb.open_table("prose")
    assert table.count_rows() == 3

    # Verify the encode method was only called ONCE with exactly ONE text payload
    mock_model_instance.encode.assert_called_once()
    passed_texts = mock_model_instance.encode.call_args[0][0]
    assert len(passed_texts) == 1
    assert "search_document: New chunk!" in passed_texts[0]
