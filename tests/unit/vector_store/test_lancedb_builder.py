from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vector_store.lancedb_builder import construct_database
from vector_store.schema import VECTOR_DIMENSIONS, build_lance_table


@pytest.fixture
def mock_parquet_dir(tmp_path: Path) -> Path:
    pq_dir = tmp_path / "parquet_vectors"
    pq_dir.mkdir()

    # Generate 5 fake records
    records: list[dict[str, Any]] = [
        {
            "chunk_id": f"chunk-00{i}",
            "rfc_number": 1000,
            "block_type": "paragraph",
            "table_route": "prose",
            "hierarchy_path": "Root",
            "text_payload": f"This is chunk {i} detailing the Transmission Control Protocol.",
            "parsing_confidence": 1.0,
            "obsoletes": [],
            "updated_by": [],
            "normative_statements": [],
        }
        for i in range(1, 6)
    ]

    # Generate random 256D vectors matching the records
    fake_vectors = np.random.rand(len(records), VECTOR_DIMENSIONS).astype(np.float32)
    vector_array = pa.FixedSizeListArray.from_arrays(
        pa.array(fake_vectors.ravel(), type=pa.float32()), VECTOR_DIMENSIONS
    )

    # Write the PyArrow table to a Parquet file matching the naming convention
    table = build_lance_table(records, vector_array)
    pq.write_table(table, pq_dir / "prose_shard_0000.parquet")  # pyright: ignore[reportUnknownMemberType]

    return pq_dir


def test_lancedb_bootstrapping_and_indexing(
    mock_parquet_dir: Path, tmp_path: Path
) -> None:
    """Asserts that LanceDB constructs tables and applies BM25 indices correctly."""
    db_dir = tmp_path / "lancedb"

    # 1. Execute the builder
    construct_database(mock_parquet_dir, db_dir)

    # 2. Verify Database & Table Creation
    db = lancedb.connect(str(db_dir))
    assert "prose" in db.list_tables().tables

    table = db.open_table("prose")
    assert table.count_rows() == 5

    # 3. Verify BM25 Tantivy Index Creation
    # If the index wasn't built, query_type="fts" will raise a ValueError.
    search_results = (  # pyright: ignore[reportUnknownVariableType]
        table
        .search("Transmission Control", query_type="fts")  # pyright: ignore[reportUnknownMemberType]
        .select(["chunk_id"])
        .to_list()
    )

    # It should successfully find the matching chunks via full-text search
    assert len(search_results) > 0  # pyright: ignore[reportUnknownArgumentType]
    assert search_results[0]["chunk_id"].startswith("chunk-")  # pyright: ignore[reportUnknownMemberType]
