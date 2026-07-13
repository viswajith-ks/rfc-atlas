from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rfc_atlas.vector_store.lancedb_builder import construct_database
from rfc_atlas.vector_store.schema import VECTOR_DIMENSIONS, build_lance_table


@pytest.fixture
def mock_parquet_dir(tmp_path: Path) -> Path:
    pq_dir = tmp_path / "parquet_vectors"
    pq_dir.mkdir()

    records: list[dict[str, Any]] = [
        {
            "chunk_id": f"chunk-00{i}",
            "rfc_number": 1000,
            "block_type": "paragraph" if i < 5 else "abnf",
            "table_route": "prose" if i < 5 else "abnf",
            "hierarchy_path": "Root",
            "text_payload": f"This is chunk {i} detailing the Transmission Control Protocol.",
            "parsing_confidence": 1.0,
            "obsoletes": [999] if i == 1 else [],
            "updated_by": [2000] if i == 1 else [],
            "normative_statements": [],
        }
        for i in range(1, 6)
    ]

    fake_vectors = np.random.rand(len(records), VECTOR_DIMENSIONS).astype(np.float32)
    vector_array = pa.FixedSizeListArray.from_arrays(
        pa.array(fake_vectors.ravel(), type=pa.float32()), VECTOR_DIMENSIONS
    )

    table = build_lance_table(records, vector_array)
    pq.write_table(table, pq_dir / "atlas_worker_0_shard_0000.parquet")  # pyright: ignore[reportUnknownMemberType]

    return pq_dir


def test_lancedb_bootstrapping_and_indexing(
    mock_parquet_dir: Path, tmp_path: Path
) -> None:
    db_dir = tmp_path / "lancedb"

    # 1. Execute the builder
    construct_database(mock_parquet_dir, db_dir)

    # 2. Verify Database & Table Creation (Testing Zero-Copy Routing)
    db = lancedb.connect(str(db_dir))

    # It should have dynamically created both tables from the monolithic shard
    assert "prose" in db.list_tables().tables
    assert "abnf" in db.list_tables().tables

    prose_table = db.open_table("prose")
    assert prose_table.count_rows() == 4

    abnf_table = db.open_table("abnf")
    assert abnf_table.count_rows() == 1

    # 3. Verify BM25 Tantivy Index Creation
    # If the index wasn't built, query_type="fts" will raise a ValueError.
    search_results = (  # pyright: ignore[reportUnknownVariableType]
        prose_table
        .search("Transmission Control", query_type="fts")  # pyright: ignore[reportUnknownMemberType]
        .select(["chunk_id"])
        .to_list()
    )

    # It should successfully find the matching chunks via full-text search
    assert len(search_results) > 0  # pyright: ignore[reportUnknownArgumentType]
    assert search_results[0]["chunk_id"].startswith("chunk-")  # pyright: ignore[reportUnknownMemberType]


def test_construct_database_exit_paths(tmp_path: Path) -> None:
    db_dir = tmp_path / "lancedb"
    missing_dir = tmp_path / "does_not_exist"
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    # 1. Test missing source directory
    with pytest.raises(SystemExit) as exc1:
        construct_database(missing_dir, db_dir)
    assert exc1.value.code == 1

    # 2. Test empty source directory (exists but no .parquet files)
    with pytest.raises(SystemExit) as exc2:
        construct_database(empty_dir, db_dir)
    assert exc2.value.code == 1
