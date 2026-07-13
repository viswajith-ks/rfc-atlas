from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rfc_atlas.vector_store.kaggle_embedder import (
    _get_existing_chunk_ids,  # pyright: ignore[reportPrivateUsage]
)


def test_get_existing_chunk_ids(tmp_path: Path) -> None:
    out_dir = tmp_path / "parquet_vectors"
    out_dir.mkdir()

    # 1. Verify genuinely empty directory behaves safely
    assert _get_existing_chunk_ids(out_dir) == set()

    # 2. Valid Parquet 1
    schema1 = pa.schema([pa.field("chunk_id", pa.string())])
    table1 = pa.Table.from_arrays(
        [pa.array(["chunk-001", "chunk-002"])], schema=schema1
    )
    pq.write_table(table1, out_dir / "valid_1.parquet")  # pyright: ignore[reportUnknownMemberType]

    # 3. Empty Parquet
    table_empty = pa.Table.from_arrays([pa.array([], type=pa.string())], schema=schema1)
    pq.write_table(table_empty, out_dir / "empty.parquet")  # pyright: ignore[reportUnknownMemberType]

    # 4. Corrupt Parquet (Text file)
    (out_dir / "corrupt.parquet").write_text(
        "This is not a parquet file.", encoding="utf-8"
    )

    # 5. Missing chunk_id column (Schema mismatch)
    schema_wrong = pa.schema([pa.field("other_column", pa.string())])
    table_wrong = pa.Table.from_arrays([pa.array(["data"])], schema=schema_wrong)
    pq.write_table(table_wrong, out_dir / "wrong_schema.parquet")  # pyright: ignore[reportUnknownMemberType]

    existing_ids = _get_existing_chunk_ids(out_dir)

    assert len(existing_ids) == 2
    assert "chunk-001" in existing_ids
    assert "chunk-002" in existing_ids
