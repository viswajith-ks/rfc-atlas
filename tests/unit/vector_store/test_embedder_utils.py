from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rfc_atlas.vector_store.kaggle_embedder import get_final_chunk_id_of_parquet


def test_get_final_chunk_id_success(tmp_path: Path) -> None:
    """Asserts that O(1) seeking correctly extracts the final chunk_id."""
    parquet_path = tmp_path / "valid_shard.parquet"

    # 1. Create a synthetic PyArrow table
    schema = pa.schema([pa.field("chunk_id", pa.string())])
    table = pa.Table.from_arrays(
        [pa.array(["chunk-001", "chunk-002", "chunk-003"])], schema=schema
    )

    # 2. Write it to disk
    pq.write_table(table, parquet_path)  # pyright: ignore[reportUnknownMemberType]

    # 3. Test extraction
    final_id = get_final_chunk_id_of_parquet(parquet_path)
    assert final_id == "chunk-003"


def test_get_final_chunk_id_empty_file(tmp_path: Path) -> None:
    """Asserts that a structurally valid but empty Parquet file returns None."""
    parquet_path = tmp_path / "empty_shard.parquet"

    schema = pa.schema([pa.field("chunk_id", pa.string())])
    table = pa.Table.from_arrays([pa.array([], type=pa.string())], schema=schema)
    pq.write_table(table, parquet_path)  # pyright: ignore[reportUnknownMemberType]

    assert get_final_chunk_id_of_parquet(parquet_path) is None


def test_get_final_chunk_id_missing_or_corrupt(tmp_path: Path) -> None:
    """Asserts that I/O exceptions are safely trapped without crashing."""
    # 1. Missing File
    missing_path = tmp_path / "does_not_exist.parquet"
    assert get_final_chunk_id_of_parquet(missing_path) is None

    # 2. Corrupt File (Not a Parquet)
    corrupt_path = tmp_path / "corrupt.parquet"
    corrupt_path.write_text("This is just a standard text file.", encoding="utf-8")
    assert get_final_chunk_id_of_parquet(corrupt_path) is None
