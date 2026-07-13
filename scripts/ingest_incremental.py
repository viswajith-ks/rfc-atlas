"""RFC Atlas — Autonomous Incremental Vector Forge.

Scans the unified master JSONL chunk file, performs a global set-difference
against the existing LanceDB tables to isolate new records, embeds them using
local CPU compute, and dynamically routes them to their specific tables.
Idempotent and safe to run continuously.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import lancedb
from lancedb import DBConnection
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from rfc_atlas.vector_store.schema import (
    LANCE_CHUNK_SCHEMA,
    build_lance_table,
    normalize_and_convert_vectors,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONNode: TypeAlias = JSONPrimitive | list["JSONNode"] | dict[str, "JSONNode"]
JSONDict: TypeAlias = dict[str, JSONNode]

logger = logging.getLogger(__name__)

BATCH_SIZE: int = 8
FLUSH_BUFFER_ROWS: int = 2_000

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = _PROJECT_ROOT / "data" / "chunks"
DB_DIR = _PROJECT_ROOT / "data" / "lancedb"


def _get_all_existing_ids(db: DBConnection) -> set[str]:
    """Extracts all existing chunk_ids across all LanceDB tables via fast columnar read.

    Args:
        db (DBConnection): The active LanceDB connection.

    Returns:
        set[str]: A set of all chunk_id strings currently present in the database.
    """
    logger.info("Extracting existing ID manifest from all LanceDB tables...")
    existing_ids: set[str] = set()

    for table_name in db.list_tables().tables:
        table = db.open_table(table_name)
        if table.count_rows() > 0:
            lance_ds = table.to_lance()  # pyright: ignore[reportUnknownMemberType]
            scanner = lance_ds.scanner(columns=["chunk_id"])  # pyright: ignore[reportUnknownMemberType]
            for batch in scanner.to_batches():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                existing_ids.update(batch["chunk_id"].to_pylist())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    return existing_ids


def _flush_and_route(
    records: list[JSONDict],
    db: DBConnection,
    model: SentenceTransformer,
) -> set[str]:
    """Embeds a buffer of new records and routes them to their specific LanceDB tables.

    Args:
        records (list[JSONDict]): The buffer of parsed JSONL chunk records.
        db (DBConnection): Active LanceDB connection.
        model (SentenceTransformer): The loaded embedding model instance.

    Returns:
        set[str]: A set of table names that received new chunks during this flush.
    """
    updated_tables: set[str] = set()
    if not records:
        return updated_tables

    texts = [f"search_document: {r['text_payload']}" for r in records]

    raw_embeddings: npt.NDArray[np.float32] = model.encode(  # pyright: ignore[reportUnknownMemberType]
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    table_groups: defaultdict[str, list[tuple[JSONDict, int]]] = defaultdict(list)
    for idx, r in enumerate(records):
        table_groups[str(r.get("table_route", "prose"))].append((r, idx))

    available_tables = set(db.list_tables().tables)

    for table_name, items in table_groups.items():
        if table_name not in available_tables:
            logger.warning(
                "⚠️ Table '%s' does not exist in LanceDB. Skipping.", table_name
            )
            continue

        group_records = [x[0] for x in items]
        group_indices = [x[1] for x in items]

        group_embeddings = raw_embeddings[group_indices]
        vector_arrow_array = normalize_and_convert_vectors(group_embeddings)

        lance_table = db.open_table(table_name)
        schema = LANCE_CHUNK_SCHEMA.with_metadata(lance_table.schema.metadata or {})

        pa_table = build_lance_table(group_records, vector_arrow_array, schema=schema)
        lance_table.add(pa_table)  # pyright: ignore[reportUnknownMemberType]

        updated_tables.add(table_name)

    return updated_tables


def _ensure_model(model: SentenceTransformer | None) -> SentenceTransformer:
    """Lazily loads the embedding model if not already initialized.

    Args:
        model (SentenceTransformer | None): The existing model instance or None.

    Returns:
        SentenceTransformer: A ready-to-use embedding model instance.
    """
    if model is None:
        logger.info("📥 Loading Nomic embedding model into CPU memory...")
        return SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
    return model


def _validate_environment(master_jsonl_path: Path) -> bool:
    """Validates that necessary directories and files exist before syncing.

    Args:
        master_jsonl_path (Path): Path to the master JSONL chunk file.

    Returns:
        bool: True if all required directories and the master JSONL file exist.
    """
    if not CHUNKS_DIR.exists() or not DB_DIR.exists():
        logger.error(
            "❌ Required directories missing. Ensure %s and %s exist.",
            CHUNKS_DIR,
            DB_DIR,
        )
        return False

    if not master_jsonl_path.exists():
        logger.info("No master_chunks.jsonl file found. Exiting.")
        return False

    return True


def _process_jsonl_stream(
    master_jsonl_path: Path, existing_ids: set[str], db: DBConnection
) -> tuple[int, set[str]]:
    """Reads the master chunk JSONL, batches new records, and flushes to LanceDB.

    Args:
        master_jsonl_path (Path): Path to the master JSONL chunk file.
        existing_ids (set[str]): A set of chunk IDs already present in the database.
        db (DBConnection): The active LanceDB connection.

    Returns:
        tuple[int, set[str]]: A tuple containing the total number of newly inserted
             chunks and a set of LanceDB table names that were modified and require
             optimization.
    """
    records_buffer: list[JSONDict] = []
    total_new_chunks: int = 0
    tables_to_optimize: set[str] = set()
    model: SentenceTransformer | None = None

    with master_jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(
            f, desc="Checking master_chunks.jsonl", unit=" lines", leave=False
        ):
            if not line.strip():
                continue

            try:
                record: JSONDict = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", master_jsonl_path.name)
                continue

            if str(record.get("chunk_id")) not in existing_ids:
                records_buffer.append(record)
                total_new_chunks += 1

                if len(records_buffer) >= FLUSH_BUFFER_ROWS:
                    model = _ensure_model(model)
                    updated = _flush_and_route(records_buffer, db, model)
                    tables_to_optimize.update(updated)
                    records_buffer.clear()

    if records_buffer:
        model = _ensure_model(model)
        updated = _flush_and_route(records_buffer, db, model)
        tables_to_optimize.update(updated)

    return total_new_chunks, tables_to_optimize


def run_incremental_sync() -> None:
    """Scans the master JSONL file, isolates deltas, and appends them to LanceDB."""
    logger.info("==================================================")
    logger.info("🔄 INITIATING AUTONOMOUS INCREMENTAL SYNC")
    logger.info("==================================================")

    master_jsonl_path = CHUNKS_DIR / "master_chunks.jsonl"
    if not _validate_environment(master_jsonl_path):
        return

    db: DBConnection = lancedb.connect(str(DB_DIR))
    existing_ids = _get_all_existing_ids(db)

    logger.info(
        "📂 Scanning %s (Found %s existing IDs)...",
        master_jsonl_path.name,
        f"{len(existing_ids):,}",
    )

    total_new_chunks, tables_to_optimize = _process_jsonl_stream(
        master_jsonl_path, existing_ids, db
    )

    logger.info("==================================================")
    if total_new_chunks == 0:
        logger.info("🎉 Database is already completely up to date.")
    else:
        logger.info("🧹 Compacting storage fragments & healing indices...")
        for t_name in tables_to_optimize:
            tbl = db.open_table(t_name)
            tbl.optimize()
            tbl.create_fts_index("text_payload", replace=True)
        logger.info(
            "🎉 INCREMENTAL SYNC COMPLETE. Routed and added %s total chunks.",
            f"{total_new_chunks:,}",
        )
    logger.info("==================================================")


def main() -> None:
    """Parses arguments and executes the incremental LanceDB vector ingestion."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="RFC Atlas Incremental Vector Forge.")
    _ = parser.parse_args()

    try:
        run_incremental_sync()
    except Exception:
        logger.exception(
            "CRITICAL FAILURE: Incremental vector forge aborted abnormally."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
