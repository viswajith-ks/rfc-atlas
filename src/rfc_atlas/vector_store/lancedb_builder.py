"""LanceDB Table Construction & Indexing Engine.

Executes strictly typed PyArrow ingestion of Gold Parquet chunks into LanceDB.
Compiles Tantivy BM25 sparse lexical indices and trains IVF-PQ dense vector clusters.
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import lancedb
import pyarrow.parquet as pq
from lancedb.table import Table as LanceTable

from rfc_atlas.vector_store.schema import LANCE_CHUNK_SCHEMA

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_IN_DIR = _PROJECT_ROOT / "data" / "embeddings" / "parquet_vectors"
LOCAL_DB_DIR = _PROJECT_ROOT / "data" / "lancedb"

HNSW_ROW_THRESHOLD = 50_000


def _build_indices(lance_table: LanceTable, total_rows: int) -> None:
    """Compiles Tantivy FTS and HNSW-SQ dense vector indices for a table.

    Args:
        lance_table (LanceTable): The active LanceDB table connection.
        total_rows (int): The total number of rows present in the table.
    """
    logger.info("Compiling Tantivy Sparse Lexical Index (BM25)...")
    start_time = time.time()
    lance_table.create_fts_index("text_payload", replace=True)
    elapsed = time.time() - start_time
    logger.info("Tantivy index compiled in %.2f seconds.", elapsed)

    if total_rows > HNSW_ROW_THRESHOLD:
        logger.info("Training HNSW Graph Index (Max Quality)...")
        start_time = time.time()
        calculated_partitions = max(16, int(total_rows // 4096))

        lance_table.create_index(
            vector_column_name="vector",
            # We strictly L2-normalize vectors during ingestion.
            # On unit vectors, L2 yields identical ranking to Cosine
            # but benefits from faster SIMD hardware optimizations.
            metric="l2",
            index_type="IVF_HNSW_SQ",
            num_partitions=calculated_partitions,
            m=32,
            ef_construction=200,
            replace=True,
        )

        elapsed = time.time() - start_time
        logger.info("HNSW training completed in %.2f seconds.", elapsed)
    else:
        logger.info("⏭️ Table < 50k rows. Bypassing HNSW Graph (Flat Search is faster).")


def construct_database(source_dir: Path, db_dir: Path) -> None:
    """Ingests Parquet shards into LanceDB using streaming appends.

    Iterates over partitioned Parquet shard files within the source directory,
    creates corresponding LanceDB tables, and sequentially streams the data in.
    Subsequently compiles indices for dense and sparse vector retrieval.

    Args:
        source_dir (Path): The directory containing the input `.parquet` shard files.
        db_dir (Path): The target directory to initialize and build the LanceDB tables.

    Raises:
        RuntimeError: If a LanceDB table fails to initialize during streaming.
    """
    logger.info("Initializing LanceDB Serverless Instance at: %s", db_dir)
    db = lancedb.connect(str(db_dir))
    schema = LANCE_CHUNK_SCHEMA

    if not source_dir.exists():
        logger.error("FATAL: Source directory does not exist: %s", source_dir)
        sys.exit(1)

    parquets = list(source_dir.glob("*.parquet"))
    if not parquets:
        logger.error("FATAL: No Parquet files found in %s", source_dir)
        sys.exit(1)

    table_groups: dict[str, list[Path]] = {}
    for p in parquets:
        table_name = p.stem.split("_shard_")[0]
        table_groups.setdefault(table_name, []).append(p)

    for table_name, file_paths in table_groups.items():
        file_paths.sort()

        logger.info("\n==================================================")
        logger.info("BUILDING TABLE: [%s]", table_name.upper())
        logger.info("==================================================")

        logger.info("Streaming %s shards into LanceDB...", len(file_paths))

        pf = pq.ParquetFile(file_paths[0])
        parquet_meta: dict[bytes, bytes] = pf.schema_arrow.metadata or {}  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        enriched_schema = LANCE_CHUNK_SCHEMA.with_metadata(parquet_meta)

        total_rows: int = 0
        lance_table: LanceTable | None = None

        for i, shard in enumerate(file_paths):
            logger.info(
                "Loading shard %d/%d: %s",
                i + 1,
                len(file_paths),
                shard.name,
            )

            tbl = pq.read_table(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                shard,
                schema=schema,
                memory_map=True,
            )

            total_rows += tbl.num_rows  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

            if i == 0:
                lance_table = db.create_table(  # pyright: ignore[reportUnknownMemberType]
                    table_name,
                    data=tbl,  # pyright: ignore[reportUnknownArgumentType]
                    schema=enriched_schema,
                    mode="overwrite",
                )
            elif lance_table is not None:
                lance_table.add(tbl)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            else:
                msg = f"Failed to initialize LanceDB table: {table_name}"
                raise RuntimeError(msg)

            del tbl

        if lance_table is None:
            logger.warning("No data processed for table: %s. Skipping.", table_name)
            continue

        logger.info(
            "Total vectors in %s: %s",
            table_name,
            f"{total_rows:,}",
        )

        _build_indices(lance_table, total_rows)  # pyright: ignore[reportUnknownArgumentType]

    logger.info("\nLANCEDB CONSTRUCTION COMPLETE!")


def main() -> None:
    """Parses arguments and executes the LanceDB table construction."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="RFC Atlas LanceDB Construction Engine."
    )
    parser.add_argument("--in-dir", type=Path, default=LOCAL_IN_DIR)
    parser.add_argument("--db-dir", type=Path, default=LOCAL_DB_DIR)
    args = parser.parse_args()

    try:
        if args.db_dir.exists():
            shutil.rmtree(args.db_dir)
        construct_database(args.in_dir, args.db_dir)
    except Exception:
        logger.exception("CRITICAL FAILURE: LanceDB construction aborted abnormally.")
        sys.exit(1)


if __name__ == "__main__":
    main()
