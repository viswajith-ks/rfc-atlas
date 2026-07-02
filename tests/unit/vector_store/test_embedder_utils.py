from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rfc_atlas.vector_store.kaggle_embedder import get_final_chunk_id_of_parquet


@pytest.mark.parametrize(
    ("scenario", "expected_id"),
    [
        ("valid", "chunk-003"),
        ("empty", None),
        ("missing", None),
        ("corrupt", None),
    ],
)
def test_get_final_chunk_id(
    tmp_path: Path, scenario: str, expected_id: str | None
) -> None:
    target_path = tmp_path / f"{scenario}_shard.parquet"

    if scenario == "valid":
        schema = pa.schema([pa.field("chunk_id", pa.string())])
        table = pa.Table.from_arrays(
            [pa.array(["chunk-001", "chunk-002", "chunk-003"])], schema=schema
        )
        pq.write_table(table, target_path)  # pyright: ignore[reportUnknownMemberType]
    elif scenario == "empty":
        schema = pa.schema([pa.field("chunk_id", pa.string())])
        table = pa.Table.from_arrays([pa.array([], type=pa.string())], schema=schema)
        pq.write_table(table, target_path)  # pyright: ignore[reportUnknownMemberType]
    elif scenario == "corrupt":
        target_path.write_text("This is just a standard text file.", encoding="utf-8")

    assert get_final_chunk_id_of_parquet(target_path) == expected_id
