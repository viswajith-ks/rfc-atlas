"""RFC Atlas — Kaggle Vector Forge.

Deterministic, fault-tolerant Matryoshka embedding across all chunk tables.
Operates on dual Kaggle T4 GPUs via SentenceTransformer multi-process pool.
Relies on Pydantic schema contracts for pristine primitive types.
"""

import argparse
import contextlib
import datetime
import gc
import json
import logging
import os
import shutil
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

from vector_store.schema import (
    LANCE_CHUNK_SCHEMA,
    VECTOR_DIMENSIONS,
    build_lance_table,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BATCH_SIZE: int = 8
FLUSH_BUFFER_ROWS: int = 65_536
SHARD_FILE_ROWS: int = 131_072
EPSILON: float = 1e-12

KAGGLE_OUT_DIR = Path("/kaggle/working/parquet_vectors")
KAGGLE_SCRATCH_DIR = Path("/kaggle/working/scratch_parquet")
_KAGGLE_BASE_INPUT = Path("/kaggle/input")

try:
    kaggle_in_dir = next(_KAGGLE_BASE_INPUT.rglob("*.jsonl")).parent
except StopIteration:
    kaggle_in_dir = _KAGGLE_BASE_INPUT

KAGGLE_IN_DIR = kaggle_in_dir

KAGGLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = KAGGLE_OUT_DIR / "embedder_telemetry.log"

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s UTC] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class ForgeContext:
    """Encapsulates static resources for the embedding pipeline."""

    out_dir: Path
    scratch_dir: Path
    schema: pa.Schema
    model: SentenceTransformer
    pool: dict[Literal["input", "output", "processes"], Any]


def verify_network_ingress() -> None:
    """Asserts DNS resolution before attempting model weights download."""
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5)
    except OSError:
        logger.exception(
            "❌ FATAL NETWORK ERROR: Container air-gapped! "
            "👉 KAGGLE: Turn 'INTERNET' toggle ON."
        )
        sys.exit(1)


def get_final_chunk_id_of_parquet(parquet_path: Path) -> str | None:
    """Extracts the trailing chunk_id from a locked Parquet shard via O(1) seek.

    Args:
        parquet_path (Path): Path to the target Parquet shard.

    Returns:
        str | None: The final chunk_id found in the file, or None if the file
            is empty or structurally corrupted.
    """
    try:
        pf = pq.ParquetFile(parquet_path)

        if pf.num_row_groups > 0:  # pyright: ignore[reportUnknownMemberType]
            last_rg: pa.Table = pf.read_row_group(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                int(pf.num_row_groups) - 1,  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
                columns=["chunk_id"],
            )

            if last_rg.num_rows > 0:  # pyright: ignore[reportUnknownMemberType]
                return str(last_rg["chunk_id"][-1].as_py())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

    except (
        OSError,
        ValueError,
        pa.ArrowInvalid,
        TypeError,
        KeyError,
        IndexError,
        pa.ArrowException,
        AttributeError,
        RuntimeError,
        EOFError,
    ):
        pass

    return None


def flush_buffer(
    records: list[dict[str, Any]],
    writer: pq.ParquetWriter,
    model: SentenceTransformer,
    schema: pa.Schema,
    multigpu_pool: dict[Literal["input", "output", "processes"], Any],
) -> None:
    """Transforms a dict buffer into PyArrow memory and writes to disk.

    Args:
        records (list[dict[str, Any]]): The buffer of parsed JSONL chunk records.
        writer (pq.ParquetWriter): The active Parquet file writer instance.
        model (SentenceTransformer): The embedding model instance.
        schema (pa.Schema): The strictly enforced PyArrow schema.
        multigpu_pool (dict): The sentence-transformers multiprocessing pool.
    """
    if not records:
        return

    texts = [f"search_document: {r['text_payload']}" for r in records]

    raw_embeddings: npt.NDArray[np.float32] = model.encode(  # pyright: ignore[reportUnknownMemberType]
        texts,
        pool=multigpu_pool,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
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
    writer.write_table(pa_table)  # pyright: ignore[reportUnknownMemberType]

    del (
        raw_embeddings,
        sliced,
        norms,
        normalized_vectors,
        flat_vector_data,
        vector_arrow_array,
        pa_table,
    )
    gc.collect()


def _commit_shard(
    writer: pq.ParquetWriter,
    local_scratch: Path,
    dest_parquet: Path,
    shard_idx: int,
    total_embedded: int,
) -> None:
    """Closes and persists a completed Parquet shard.

    Args:
        writer (pq.ParquetWriter): The active Parquet file writer.
        local_scratch (Path): Temporary scratch path of the active shard.
        dest_parquet (Path): Final destination path for the completed shard.
        shard_idx (int): The sequential index of the shard.
        total_embedded (int): Running total of vectors embedded so far.
    """
    writer.close()
    shutil.copy2(local_scratch, dest_parquet)
    local_scratch.unlink()
    logger.info(
        "🔒 Shard #%04d secured. (%s vectors done).",
        shard_idx,
        f"{total_embedded:,}",
    )


def _find_resume_point(table_name: str, out_dir: Path) -> tuple[int, str | None]:
    """Scans existing shards to find the exact resumption chunk ID.

    Args:
        table_name (str): Base name of the table being processed.
        out_dir (Path): Output directory containing completed shards.

    Returns:
        tuple[int, str | None]: The next shard index and the ID of the last
            processed chunk (if any).
    """
    shard_idx = 0
    last_known_id = None
    while True:
        dest_parquet = out_dir / f"{table_name}_shard_{shard_idx:04d}.parquet"
        if not dest_parquet.exists():
            break

        cid = get_final_chunk_id_of_parquet(dest_parquet)
        if cid is None:
            logger.warning(
                "⚠️ [FRONTIER WARN] Shard #%04d corrupted or empty! Overwriting.",
                shard_idx,
            )
            break

        last_known_id = cid
        logger.info(
            "⏩ [IDEMPOTENCY] Shard #%04d locked. (Tail ID: '%s')",
            shard_idx,
            last_known_id,
        )
        shard_idx += 1

    return shard_idx, last_known_id


def _load_records(jsonl_path: Path, start_id: str | None) -> list[dict[str, Any]]:
    """Loads the entire JSONL file into System RAM, skipping processed records.

    Args:
        jsonl_path (Path): Path to the target JSONL file.
        start_id (str | None): Chunk ID to resume from, or None to read all.

    Returns:
        list[dict[str, Any]]: A length-sorted list of unprocessed record dicts.
    """
    seeking = start_id is not None
    records: list[dict[str, Any]] = []
    search_token = f'"chunk_id": "{start_id}"' if seeking else ""

    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            if seeking and search_token not in line:
                continue

            if seeking:
                with contextlib.suppress(json.JSONDecodeError):
                    if json.loads(line).get("chunk_id") == start_id:
                        logger.info("🎯 SEEKER MATCH! Dropping lock; resuming.")
                        seeking = False
                continue

            with contextlib.suppress(json.JSONDecodeError):
                records.append(json.loads(line))

    logger.info("🧠 Applying Smart Batching (Sorting by text length)...")
    records.sort(key=lambda x: len(x.get("text_payload", "")))

    return records


def _ensure_writer(
    writer: pq.ParquetWriter | None,
    scratch_path: Path,
    schema: pa.Schema,
) -> pq.ParquetWriter:
    """Instantiates a snappy ParquetWriter on demand if one is not already open.

    Args:
        writer (pq.ParquetWriter | None): Existing writer or None.
        scratch_path (Path): Location to create the Parquet file.
        schema (pa.Schema): PyArrow schema to enforce.

    Returns:
        pq.ParquetWriter: A ready-to-use ParquetWriter instance.
    """
    if writer is None:
        return pq.ParquetWriter(scratch_path, schema, compression="snappy")
    return writer


def _process_single_table(jsonl_path: Path, ctx: ForgeContext) -> None:
    """Isolates the ingestion and vectorization loop for a single JSONL table.

    Args:
        jsonl_path (Path): The path to the source JSONL table.
        ctx (ForgeContext): The global context providing schema and models.
    """
    table_name = jsonl_path.stem
    logger.info("\n%s", "=" * 75)
    logger.info("🚀 INITIATING FORGE FOR TABLE: [%s]", table_name.upper())
    logger.info("=" * 75)

    shard_idx, last_known_id = _find_resume_point(table_name, ctx.out_dir)
    dest_parquet = ctx.out_dir / f"{table_name}_shard_{shard_idx:04d}.parquet"
    local_scratch = ctx.scratch_dir / dest_parquet.name

    shard_writer: pq.ParquetWriter | None = None

    if last_known_id:
        logger.info(
            "🔎 FORENSIC SEEKER ENGAGED: Fast-forwarding to ID '%s'...",
            last_known_id,
        )

    logger.info("📦 Loading entire JSONL table into System RAM...")
    all_records = _load_records(jsonl_path, last_known_id)
    total_to_process = len(all_records)

    if total_to_process == 0:
        logger.info("⏭️ No new records to process for [%s].", table_name.upper())
        return

    logger.info(
        "✅ Successfully buffered %s records into memory.", f"{total_to_process:,}"
    )

    rows_in_current_shard = 0
    total_embedded = 0

    try:
        for i in range(0, total_to_process, FLUSH_BUFFER_ROWS):
            records_buffer = all_records[i : i + FLUSH_BUFFER_ROWS]

            shard_writer = _ensure_writer(shard_writer, local_scratch, ctx.schema)

            logger.info("⚙️ Encoding buffer of %s rows...", f"{len(records_buffer):,}")
            flush_buffer(
                records_buffer,
                shard_writer,
                ctx.model,
                ctx.schema,
                ctx.pool,
            )

            batch_size_actual = len(records_buffer)
            rows_in_current_shard += batch_size_actual
            total_embedded += batch_size_actual

            if rows_in_current_shard >= SHARD_FILE_ROWS:
                _commit_shard(
                    shard_writer,
                    local_scratch,
                    dest_parquet,
                    shard_idx,
                    total_embedded,
                )
                shard_writer = None

                shard_idx += 1
                rows_in_current_shard = 0
                dest_parquet = (
                    ctx.out_dir / f"{table_name}_shard_{shard_idx:04d}.parquet"
                )
                local_scratch = ctx.scratch_dir / dest_parquet.name

    finally:
        if shard_writer is not None:
            with contextlib.suppress(Exception):
                shard_writer.close()

    if local_scratch.exists():
        shutil.copy2(local_scratch, dest_parquet)
        local_scratch.unlink()

    logger.info(
        "🏁 TABLE [%s] SUCCESS: %s Vectors Forged",
        table_name.upper(),
        f"{total_embedded:,}",
    )

    del all_records
    gc.collect()


def execute_pipeline(in_dir: Path, out_dir: Path, scratch_dir: Path) -> None:
    """Coordinates fault-tolerant streaming transformation across ALL target tables.

    Args:
        in_dir (Path): Source directory containing the JSONL text chunks.
        out_dir (Path): Destination directory for the finished Parquet shards.
        scratch_dir (Path): Temporary workspace for active Parquet writers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    schema = LANCE_CHUNK_SCHEMA.with_metadata({
        "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
        "vector_dimensions": str(VECTOR_DIMENSIONS),
        "matryoshka_truncated": "true",
        "task_prefix": "search_document: ",
        "execution_environment": "kaggle",
        "pipeline_version": "v8.0-strict-contract",
        "forged_at": datetime.datetime.now(datetime.UTC).isoformat(),
    })

    target_files = sorted(in_dir.glob("*.jsonl"))

    if not target_files:
        logger.error(
            "❌ CRITICAL FATAL: No .jsonl tables discovered inside [%s]!", in_dir
        )
        sys.exit(1)

    logger.info("🌍 Discovered %d tables queued for ingestion.", len(target_files))
    logger.info("📥 Loading nomic-embed-text-v1.5 weights into CPU shared memory...")

    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu"
    )

    if not hasattr(model, "start_multi_process_pool"):
        logger.error(
            "CRITICAL: API mismatch — ST release missing start_multi_process_pool!"
        )
        sys.exit(1)

    target_devices = ["cuda:0", "cuda:1"]
    logger.info("⚡ Spawning independent CUDA worker pool across %s...", target_devices)

    multigpu_pool: dict[Literal["input", "output", "processes"], Any] = (
        model.start_multi_process_pool(target_devices=target_devices)
    )

    ctx = ForgeContext(
        out_dir=out_dir,
        scratch_dir=scratch_dir,
        schema=schema,
        model=model,
        pool=multigpu_pool,
    )

    try:
        for jsonl_path in target_files:
            _process_single_table(jsonl_path, ctx)
    finally:
        logger.info("🛑 Shutting down independent CUDA multiprocessing worker pool...")
        model.stop_multi_process_pool(multigpu_pool)

        time.sleep(3)
        gc.collect()


if __name__ == "__main__":
    verify_network_ingress()
    parser = argparse.ArgumentParser(description="RFC Atlas Kaggle Vector Forge.")
    parser.add_argument("--in-dir", type=str, help="Custom input JSONL directory.")
    parser.add_argument("--out-dir", type=str, help="Custom output Parquet directory.")
    parser.add_argument(
        "--scratch-dir", type=str, help="Custom fast NVMe scratch path."
    )

    args = parser.parse_args()

    resolved_in = Path(args.in_dir) if args.in_dir else KAGGLE_IN_DIR
    resolved_out = Path(args.out_dir) if args.out_dir else KAGGLE_OUT_DIR
    resolved_scratch = (
        Path(args.scratch_dir) if args.scratch_dir else KAGGLE_SCRATCH_DIR
    )

    logger.info("\n%s", "=" * 75)
    logger.info("🔥 IGNITING KAGGLE VECTOR FORGE (STRICT CONTRACT)")
    logger.info("📂 Source: %s", resolved_in)
    logger.info("💾 Target: %s", resolved_out)
    logger.info("=" * 75)

    execute_pipeline(resolved_in, resolved_out, resolved_scratch)

    logger.info("\n%s", "=" * 75)
    logger.info("🎉 ALL ASSIGNED VECTORS LOCKED.")
    logger.info("=" * 75)
