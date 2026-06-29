"""RFC Atlas — Autonomous Incremental Vector Forge.

Scans all local JSONL chunk files, performs a set-difference against the existing
LanceDB tables to isolate new records, embeds them using local CPU compute,
and appends them to the database. Idempotent and safe to run continuously.
"""

import json
import logging
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from lancedb import DBConnection
from lancedb.table import Table as LanceTable
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from vector_store.schema import (
    LANCE_CHUNK_SCHEMA,
    VECTOR_DIMENSIONS,
    build_lance_table,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

EPSILON: float = 1e-12
BATCH_SIZE: int = 8
FLUSH_BUFFER_ROWS: int = 2_000

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = _PROJECT_ROOT / "data" / "chunks"
DB_DIR = _PROJECT_ROOT / "data" / "lancedb"


def _get_existing_ids(table: LanceTable) -> set[str]:
    """Extracts all existing chunk_ids from LanceDB via ultra-fast columnar read.

    Args:
        table (LanceTable): The active LanceDB table.

    Returns:
        set[str]: A set containing all chunk_id strings currently present in the table.
    """
    logger.info("Extracting existing ID manifest from LanceDB...")

    total_rows: int = table.count_rows()
    if total_rows == 0:
        return set()

    arrow_tbl = table.to_lance().to_table(columns=["chunk_id"])  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    return set(arrow_tbl["chunk_id"].to_pylist())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]


def _flush_and_append(
    records: list[dict[str, Any]],
    lance_table: LanceTable,
    model: SentenceTransformer,
    schema: pa.Schema,
) -> None:
    """Embeds a buffer of new records and appends them to LanceDB.

    Args:
        records (list[dict[str, Any]]): The buffer of parsed JSONL chunk records.
        lance_table (LanceTable): The target LanceDB table for ingestion.
        model (SentenceTransformer): The loaded embedding model instance.
        schema (pa.Schema): The PyArrow schema to validate and enforce.
    """
    if not records:
        return

    texts = [f"search_document: {r['text_payload']}" for r in records]

    raw_embeddings: npt.NDArray[np.float32] = model.encode(  # pyright: ignore[reportUnknownMemberType]
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    sliced = raw_embeddings[:, :VECTOR_DIMENSIONS]
    norms: npt.NDArray[np.float32] = np.sqrt(
        np.sum(sliced * sliced, axis=1, keepdims=True, dtype=np.float32)
    )
    norms[norms < EPSILON] = EPSILON
    normalized_vectors = sliced / norms

    flat_vector_data = normalized_vectors.ravel()
    vector_arrow_array = pa.FixedSizeListArray.from_arrays(
        pa.array(flat_vector_data, type=pa.float32()), VECTOR_DIMENSIONS
    )

    pa_table = build_lance_table(records, vector_arrow_array, schema=schema)
    lance_table.add(pa_table)  # pyright: ignore[reportUnknownMemberType]


def _ensure_model(model: SentenceTransformer | None) -> SentenceTransformer:
    """Lazily loads the embedding model if not already initialized.

    Args:
        model (SentenceTransformer | None): The existing model instance or None.

    Returns:
        SentenceTransformer: A ready-to-use embedding model instance.
    """
    if model is None:
        logger.info("📥 Loading Nomic embedding model...")
        return SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
    return model


def _sync_single_table(
    jsonl_path: Path, db: DBConnection, model: SentenceTransformer | None
) -> tuple[int, SentenceTransformer | None]:
    """Synchronizes a single JSONL table with its LanceDB counterpart.

    Args:
        jsonl_path (Path): Path to the JSONL chunk file.
        db (DBConnection): Active LanceDB connection.
        model (SentenceTransformer | None): The current SentenceTransformer model
            instance, or None if it has not yet been loaded.

    Returns:
        tuple[int, SentenceTransformer | None]: A tuple containing the number of
            new chunks appended, and the model instance.
    """
    table_name = jsonl_path.stem

    if table_name not in db.table_names():
        logger.warning("⚠️ Table '%s' does not exist in LanceDB. Skipping.", table_name)
        return 0, model

    lance_table = db.open_table(table_name)
    table_schema = LANCE_CHUNK_SCHEMA.with_metadata(
        lance_table.schema.metadata or {}  # pyright: ignore[reportUnknownMemberType]
    )
    existing_ids = _get_existing_ids(lance_table)

    logger.info(
        "📂 Scanning %s (Found %s existing IDs)...",
        jsonl_path.name,
        f"{len(existing_ids):,}",
    )

    records_buffer: list[dict[str, Any]] = []
    table_new_chunks: int = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Checking {table_name}", unit=" lines", leave=False):
            if not line.strip():
                continue

            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", jsonl_path.name)
                continue

            if record["chunk_id"] not in existing_ids:
                records_buffer.append(record)
                table_new_chunks += 1

                if len(records_buffer) >= FLUSH_BUFFER_ROWS:
                    model = _ensure_model(model)
                    _flush_and_append(records_buffer, lance_table, model, table_schema)
                    records_buffer.clear()

    if records_buffer:
        model = _ensure_model(model)
        _flush_and_append(records_buffer, lance_table, model, table_schema)

    if table_new_chunks == 0:
        logger.info("⏭️ %s: No new chunks to add.", table_name)
    else:
        logger.info(
            "✅ %s: Appended %s new chunks.", table_name, f"{table_new_chunks:,}"
        )

    return table_new_chunks, model


def run_incremental_sync() -> None:
    """Scans all JSONL files, isolates deltas, and appends them to LanceDB."""
    logger.info("==================================================")
    logger.info("🔄 INITIATING AUTONOMOUS INCREMENTAL SYNC")
    logger.info("==================================================")

    if not CHUNKS_DIR.exists() or not DB_DIR.exists():
        logger.error(
            "❌ Required directories missing. Ensure %s and %s exist.",
            CHUNKS_DIR,
            DB_DIR,
        )
        return

    jsonl_files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    if not jsonl_files:
        logger.info("No .jsonl files found. Exiting.")
        return

    db: DBConnection = lancedb.connect(str(DB_DIR))

    model: SentenceTransformer | None = None
    total_new_chunks: int = 0

    for jsonl_path in jsonl_files:
        new_chunks, model = _sync_single_table(jsonl_path, db, model)
        total_new_chunks += new_chunks

    logger.info("==================================================")
    if total_new_chunks == 0:
        logger.info("🎉 Database is already completely up to date.")
    else:
        logger.info(
            "🎉 INCREMENTAL SYNC COMPLETE. Added %s total chunks.",
            f"{total_new_chunks:,}",
        )
    logger.info("==================================================")


if __name__ == "__main__":
    run_incremental_sync()
