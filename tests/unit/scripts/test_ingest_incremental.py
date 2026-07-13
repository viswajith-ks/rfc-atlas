import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import lancedb
import numpy as np
import pyarrow as pa
import pytest

from rfc_atlas.vector_store.schema import (
    LANCE_CHUNK_SCHEMA,
    VECTOR_DIMENSIONS,
    build_lance_table,
)
from scripts.ingest_incremental import (
    _get_all_existing_ids,  # pyright: ignore[reportPrivateUsage]
    _process_jsonl_stream,  # pyright: ignore[reportPrivateUsage]
    run_incremental_sync,
)


@pytest.fixture
def mock_lancedb(tmp_path: Path) -> lancedb.DBConnection:
    db_dir = tmp_path / "lancedb"
    db = lancedb.connect(str(db_dir))

    def create_fake_table(name: str, ids: list[str]) -> None:
        records: list[dict[str, Any]] = [
            {
                "chunk_id": cid,
                "rfc_number": 1000,
                "block_type": "paragraph" if name == "prose" else "abnf",
                "table_route": name,
                "hierarchy_path": "Root",
                "text_payload": f"Existing payload {cid}",
                "parsing_confidence": 1.0,
                "obsoletes": [],
                "updated_by": [],
                "normative_statements": [],
            }
            for cid in ids
        ]

        fake_vectors = np.random.rand(len(ids), VECTOR_DIMENSIONS).astype(np.float32)
        vector_array = pa.FixedSizeListArray.from_arrays(
            pa.array(fake_vectors.ravel(), type=pa.float32()), VECTOR_DIMENSIONS
        )

        table = build_lance_table(records, vector_array)
        db.create_table(name, data=table, schema=LANCE_CHUNK_SCHEMA)  # pyright: ignore[reportUnknownMemberType]

    # Create multiple tables to test global extraction and dynamic routing
    create_fake_table("prose", ["chunk-001", "chunk-002"])
    create_fake_table("abnf", ["chunk-a01"])

    return db


@pytest.fixture
def mock_master_jsonl(tmp_path: Path) -> Path:
    jsonl_path = tmp_path / "master_chunks.jsonl"

    records: list[dict[str, Any]] = [
        {
            "chunk_id": "chunk-002",
            "rfc_number": 1000,
            "block_type": "paragraph",
            "table_route": "prose",
            "hierarchy_path": "Root",
            "text_payload": "Duplicate prose",
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
            "text_payload": "New prose chunk!",
            "parsing_confidence": 1.0,
            "obsoletes": [],
            "updated_by": [],
            "normative_statements": [],
        },
        {
            "chunk_id": "chunk-a02",
            "rfc_number": 1000,
            "block_type": "abnf",
            "table_route": "abnf",
            "hierarchy_path": "Root",
            "text_payload": "New abnf chunk!",
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


def test_global_set_difference(mock_lancedb: lancedb.DBConnection) -> None:
    existing_ids = _get_all_existing_ids(mock_lancedb)

    assert len(existing_ids) == 3
    assert "chunk-001" in existing_ids
    assert "chunk-002" in existing_ids
    assert "chunk-a01" in existing_ids


@patch("scripts.ingest_incremental.SentenceTransformer")
def test_process_jsonl_stream_skips_duplicates_and_routes(
    mock_st_class: MagicMock,
    mock_lancedb: lancedb.DBConnection,
    mock_master_jsonl: Path,
) -> None:
    mock_model_instance = MagicMock()
    mock_st_class.return_value = mock_model_instance

    def fake_encode(texts: list[str], **_: list[Any]) -> np.ndarray:
        return np.random.rand(len(texts), 768).astype(np.float32)

    mock_model_instance.encode.side_effect = fake_encode

    existing_ids = _get_all_existing_ids(mock_lancedb)
    new_chunks, tables_to_optimize = _process_jsonl_stream(
        mock_master_jsonl, existing_ids, mock_lancedb
    )

    assert new_chunks == 2
    assert "prose" in tables_to_optimize
    assert "abnf" in tables_to_optimize

    prose_table = mock_lancedb.open_table("prose")
    assert prose_table.count_rows() == 3

    abnf_table = mock_lancedb.open_table("abnf")
    assert abnf_table.count_rows() == 2

    mock_model_instance.encode.assert_called_once()
    passed_texts = mock_model_instance.encode.call_args[0][0]

    assert len(passed_texts) == 2
    assert "search_document: New prose chunk!" in passed_texts

    # Assert progress bar is disabled so stdout isn't polluted in production
    assert mock_model_instance.encode.call_args[1].get("show_progress_bar") is False


@patch("scripts.ingest_incremental.lancedb.connect")
@patch("scripts.ingest_incremental._validate_environment")
@patch("scripts.ingest_incremental._get_all_existing_ids")
@patch("scripts.ingest_incremental._process_jsonl_stream")
def test_run_incremental_sync_rebuilds_fts(
    mock_process: MagicMock,
    mock_get_ids: MagicMock,
    mock_validate: MagicMock,
    mock_connect: MagicMock,
) -> None:
    mock_validate.return_value = True
    mock_get_ids.return_value = set()

    # Simulate the jsonl stream successfully parsing 5 chunks and updating 2 tables
    mock_process.return_value = (5, {"prose", "abnf"})

    mock_db = MagicMock()
    mock_connect.return_value = mock_db

    mock_prose = MagicMock()
    mock_abnf = MagicMock()

    def mock_open(name: str) -> MagicMock:
        return mock_prose if name == "prose" else mock_abnf

    mock_db.open_table.side_effect = mock_open

    # Execute the orchestrator function
    run_incremental_sync()

    # 1. Assert prose table was healed and the FTS index was rebuilt
    mock_prose.optimize.assert_called_once()
    mock_prose.create_fts_index.assert_called_once_with("text_payload", replace=True)

    # 2. Assert abnf table was healed and the FTS index was rebuilt
    mock_abnf.optimize.assert_called_once()
    mock_abnf.create_fts_index.assert_called_once_with("text_payload", replace=True)
