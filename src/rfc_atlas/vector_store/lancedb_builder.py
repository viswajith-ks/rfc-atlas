"""LanceDB Table Construction & Indexing Engine.

Executes strictly typed PyArrow ingestion of unified Parquet chunks into LanceDB.
Leverages PyArrow Dataset push-down filters for zero-copy table segregation.
Compiles Tantivy BM25 sparse lexical indices and trains IVF-PQ dense vector clusters.
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import get_args

import lancedb
import pyarrow.dataset as ds
from lancedb.table import Table as LanceTable

from rfc_atlas.chunking.schema import LanceTableRoute
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
    """Ingests monolithic Parquet shards into segregated LanceDB tables.

    Mounts the entire folder of parquet shards as a single PyArrow Dataset.
    Uses zero-copy push-down filters to stream route-specific slices
    into their respective independent LanceDB tables.

    Args:
        source_dir (Path): The directory containing the monolithic `.parquet` shards.
        db_dir (Path): The target directory to initialize and build the LanceDB tables.
    """
    logger.info("Initializing LanceDB Serverless Instance at: %s", db_dir)
    db = lancedb.connect(str(db_dir))

    if not source_dir.exists():
        logger.error("FATAL: Source directory does not exist: %s", source_dir)
        sys.exit(1)

    parquets = list(source_dir.glob("*.parquet"))
    if not parquets:
        logger.error("FATAL: No Parquet files found in %s", source_dir)
        sys.exit(1)

    logger.info("Mounting unified PyArrow Dataset from %s shards...", len(parquets))
    dataset = ds.dataset(source_dir, format="parquet")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    parquet_meta: dict[bytes, bytes] = dataset.schema.metadata or {}  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    enriched_schema = LANCE_CHUNK_SCHEMA.with_metadata(parquet_meta)

    for table_name in sorted(get_args(LanceTableRoute)):
        logger.info("\n==================================================")
        logger.info("BUILDING TABLE: [%s]", table_name.upper())
        logger.info("==================================================")

        condition = ds.field("table_route") == table_name  # pyright: ignore[reportUnknownVariableType, reportPrivateImportUsage, reportUnknownMemberType]
        scanner = dataset.scanner(filter=condition)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        total_rows: int = scanner.count_rows()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        if total_rows == 0:
            logger.warning("No data found for table route: %s. Skipping.", table_name)
            continue

        logger.info("Streaming %s matching chunks into LanceDB...", f"{total_rows:,}")

        reader = scanner.to_reader()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        lance_table = db.create_table(  # pyright: ignore[reportUnknownMemberType]
            table_name,
            data=reader,  # pyright: ignore[reportUnknownArgumentType]
            schema=enriched_schema,
            mode="overwrite",
        )

        logger.info("Total vectors in %s: %s", table_name, f"{total_rows:,}")
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
