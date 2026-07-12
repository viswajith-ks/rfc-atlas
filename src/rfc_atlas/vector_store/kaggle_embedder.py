"""RFC Atlas — Kaggle Vector Forge.

Deterministic, fault-tolerant Matryoshka embedding across the master chunk table.
Operates on dual Kaggle T4 GPUs via Isolated Zero-IPC multiprocessing workers.
Relies on Pydantic schema contracts for pristine primitive types.
"""

import argparse
import contextlib
import datetime
import gc
import json
import logging
import multiprocessing as mp
import os
import shutil
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

from rfc_atlas.vector_store.schema import (
    LANCE_CHUNK_SCHEMA,
    VECTOR_DIMENSIONS,
    build_lance_table,
    normalize_and_convert_vectors,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONNode: TypeAlias = JSONPrimitive | list["JSONNode"] | dict[str, "JSONNode"]
JSONDict: TypeAlias = dict[str, JSONNode]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BATCH_SIZE: int = 32
FLUSH_BUFFER_ROWS: int = 65_536
SHARD_FILE_ROWS: int = 131_072

KAGGLE_OUT_DIR = Path(
    os.environ.get("RFC_ATLAS_OUT_DIR", "/kaggle/working/parquet_vectors")
)
KAGGLE_SCRATCH_DIR = Path(
    os.environ.get("RFC_ATLAS_SCRATCH_DIR", "/kaggle/working/scratch_parquet")
)
_BASE_INPUT = Path(os.environ.get("RFC_ATLAS_IN_DIR", "/kaggle/input"))
KAGGLE_IN_DIR = _BASE_INPUT

LOG_FILE = (
    Path(os.environ.get("RFC_ATLAS_LOG_DIR", "/kaggle/working"))
    / "embedder_telemetry.log"
)

logger = logging.getLogger(__name__)


@dataclass
class ForgeContext:
    """Encapsulates static resources for the embedding pipeline."""

    out_dir: Path
    scratch_dir: Path
    schema: pa.Schema


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


def _get_existing_chunk_ids(out_dir: Path) -> set[str]:
    """Scans all Parquet shards to compile a global set of completed chunk IDs.

    Args:
        out_dir (Path): The output directory containing completed Parquet shards.

    Returns:
        set[str]: A set containing all chunk_id strings successfully read.
    """
    existing_ids: set[str] = set()
    for pq_file in out_dir.glob("*.parquet"):
        with contextlib.suppress(
            OSError, pa.ArrowInvalid, pa.ArrowException, KeyError, ValueError
        ):
            table = pq.read_table(pq_file, columns=["chunk_id"])  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            existing_ids.update(table["chunk_id"].to_pylist())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    return existing_ids


def _get_next_shard_idx(worker_id: int, out_dir: Path) -> int:
    """Determines the next safe Parquet shard index for a specific worker.

    Args:
        worker_id (int): The ID of the worker process.
        out_dir (Path): The output directory containing completed Parquet shards.

    Returns:
        int: The next available sequential integer for a new shard.
    """
    existing = list(out_dir.glob(f"atlas_worker_{worker_id}_shard_*.parquet"))
    if not existing:
        return 0
    idxs: list[int] = []
    for f in existing:
        with contextlib.suppress(IndexError, ValueError):
            idxs.append(int(f.stem.split("_shard_")[1]))
    return max(idxs) + 1 if idxs else 0


def _setup_worker_logging(worker_id: int) -> logging.Logger:
    """Configures isolated logging for a worker process.

    Args:
        worker_id (int): The ID of the current worker.

    Returns:
        logging.Logger: The configured worker-specific logger.
    """
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] [Worker {worker_id}] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        ],
    )
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def _load_and_shard_records(
    target_file: Path,
    existing_ids: set[str],
    worker_id: int,
    worker_logger: logging.Logger,
) -> list[JSONDict]:
    """Loads, filters, sorts, and shards the unified JSONL into worker memory.

    Args:
        target_file (Path): Path to the master JSONL chunks file.
        existing_ids (set[str]): Set of already embedded chunk IDs to skip.
        worker_id (int): The ID of the current worker.
        worker_logger (logging.Logger): Logger for output tracking.

    Returns:
        list[JSONDict]: The interleaved slice of records assigned to this worker.
    """
    worker_logger.info("📦 Loading master_chunks.jsonl directly into Worker RAM...")
    all_records: list[JSONDict] = []
    with target_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            with contextlib.suppress(json.JSONDecodeError):
                rec = json.loads(line)
                if rec.get("chunk_id") not in existing_ids:
                    all_records.append(rec)

    worker_logger.info(
        "🧠 Applying Smart Batching (Sorting %s records by length)...",
        f"{len(all_records):,}",
    )
    all_records.sort(key=lambda x: len(str(x.get("text_payload", ""))), reverse=True)

    worker_records: list[JSONDict] = []
    for i in range(0, len(all_records), BATCH_SIZE):
        if (i // BATCH_SIZE) % 2 == worker_id:
            worker_records.extend(all_records[i : i + BATCH_SIZE])

    del all_records
    gc.collect()

    return worker_records


def _worker_process(
    worker_id: int,
    device: str,
    target_file: Path,
    existing_ids: set[str],
    ctx: ForgeContext,
) -> None:
    """Isolated, Zero-IPC worker process for loading, sorting, slicing, and embedding.

    This function operates in a totally isolated memory space.
    """
    worker_logger = _setup_worker_logging(worker_id)
    worker_records = _load_and_shard_records(
        target_file, existing_ids, worker_id, worker_logger
    )
    total_to_process = len(worker_records)

    if total_to_process == 0:
        worker_logger.info("⏭️ No assigned records remaining. Shutting down gracefully.")
        return

    worker_logger.info(
        "⚡ Initializing SentenceTransformer directly on %s (FP16)...", device
    )
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device=device,
        model_kwargs={"torch_dtype": torch.float16},
    )

    shard_idx = _get_next_shard_idx(worker_id, ctx.out_dir)
    worker_logger.info(
        "Assigned %s records. Starting at shard index %04d.",
        f"{total_to_process:,}",
        shard_idx,
    )

    dest_parquet = (
        ctx.out_dir / f"atlas_worker_{worker_id}_shard_{shard_idx:04d}.parquet"
    )
    local_scratch = ctx.scratch_dir / dest_parquet.name

    shard_writer: pq.ParquetWriter | None = None
    rows_in_current_shard = 0
    total_embedded = 0

    try:
        for i in range(0, total_to_process, FLUSH_BUFFER_ROWS):
            buffer = worker_records[i : i + FLUSH_BUFFER_ROWS]

            if shard_writer is None:
                shard_writer = pq.ParquetWriter(
                    local_scratch, ctx.schema, compression="snappy"
                )

            worker_logger.info("⚙️ Encoding batch of %s rows...", f"{len(buffer):,}")

            texts = [f"search_document: {r['text_payload']}" for r in buffer]
            raw_embeddings: npt.NDArray[np.float32] = model.encode(  # pyright: ignore[reportUnknownMemberType]
                texts,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            vector_arrow_array = normalize_and_convert_vectors(raw_embeddings)
            pa_table = build_lance_table(buffer, vector_arrow_array, schema=ctx.schema)
            shard_writer.write_table(pa_table)  # pyright: ignore[reportUnknownMemberType]

            del raw_embeddings, vector_arrow_array, pa_table
            gc.collect()

            batch_size_actual = len(buffer)
            rows_in_current_shard += batch_size_actual
            total_embedded += batch_size_actual

            if rows_in_current_shard >= SHARD_FILE_ROWS:
                shard_writer.close()
                shutil.copy2(local_scratch, dest_parquet)
                local_scratch.unlink()
                worker_logger.info(
                    "🔒 Shard #%04d secured. (%s vectors done).",
                    shard_idx,
                    f"{total_embedded:,}",
                )

                shard_writer = None
                shard_idx += 1
                rows_in_current_shard = 0
                dest_parquet = (
                    ctx.out_dir
                    / f"atlas_worker_{worker_id}_shard_{shard_idx:04d}.parquet"
                )
                local_scratch = ctx.scratch_dir / dest_parquet.name

    finally:
        if shard_writer is not None:
            with contextlib.suppress(Exception):
                shard_writer.close()
            if local_scratch.exists():
                shutil.copy2(local_scratch, dest_parquet)
                local_scratch.unlink()
                worker_logger.info(
                    "🔒 Final Shard #%04d secured. (%s vectors done).",
                    shard_idx,
                    f"{total_embedded:,}",
                )

    worker_logger.info(
        "🏁 WORKER %s SUCCESS: %s Vectors Forged", worker_id, f"{total_embedded:,}"
    )


def execute_pipeline(in_dir: Path, out_dir: Path, scratch_dir: Path) -> None:
    """Coordinates the dual-GPU Zero-IPC execution framework."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    schema = LANCE_CHUNK_SCHEMA.with_metadata({
        "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
        "vector_dimensions": f"{VECTOR_DIMENSIONS}",
        "matryoshka_truncated": "true",
        "task_prefix": "search_document: ",
        "execution_environment": "kaggle",
        "pipeline_version": "v9.0-zero-ipc-forge",
        "forged_at": datetime.datetime.now(datetime.UTC).isoformat(),
    })

    target_file = in_dir / "master_chunks.jsonl"
    if not target_file.exists():
        logger.error(
            "❌ CRITICAL FATAL: master_chunks.jsonl not found in [%s]!", in_dir
        )
        sys.exit(1)

    logger.info("🌍 Discovering existing chunk IDs to resume gracefully...")
    existing_ids = _get_existing_chunk_ids(out_dir)
    logger.info(
        "✅ Found %s existing IDs in output directory.", f"{len(existing_ids):,}"
    )

    ctx = ForgeContext(out_dir=out_dir, scratch_dir=scratch_dir, schema=schema)
    logger.info("⚡ Spawning isolated CUDA worker processes (Zero IPC)...")
    ctx_mp = mp.get_context("spawn")

    p0 = ctx_mp.Process(
        target=_worker_process, args=(0, "cuda:0", target_file, existing_ids, ctx)
    )
    p1 = ctx_mp.Process(
        target=_worker_process, args=(1, "cuda:1", target_file, existing_ids, ctx)
    )

    p0.start()
    p1.start()

    p0.join()
    p1.join()

    if p0.exitcode != 0 or p1.exitcode != 0:
        logger.error(
            "❌ CRITICAL FATAL: One or more GPU workers failed. "
            "Exit codes: Worker 0 (%s), Worker 1 (%s)",
            p0.exitcode,
            p1.exitcode,
        )
        sys.exit(1)

    logger.info("🎉 Dual-GPU Forge successfully merged all worker processes.")


def main() -> None:
    """Parses arguments and executes the Kaggle vector forge pipeline."""
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [ORCHESTRATOR] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        ],
    )

    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    verify_network_ingress()
    parser = argparse.ArgumentParser(description="RFC Atlas Kaggle Vector Forge.")
    parser.add_argument("--in-dir", type=str, help="Custom input JSONL directory.")
    parser.add_argument("--out-dir", type=str, help="Custom output Parquet directory.")
    parser.add_argument(
        "--scratch-dir", type=str, help="Custom fast NVMe scratch path."
    )

    args = parser.parse_args()

    resolved_in = (
        Path(args.in_dir)
        if args.in_dir
        else Path(os.environ.get("RFC_ATLAS_IN_DIR", "/kaggle/input"))
    )
    if not args.in_dir:
        with contextlib.suppress(StopIteration):
            resolved_in = next(resolved_in.rglob("*.jsonl")).parent

    resolved_out = Path(args.out_dir) if args.out_dir else KAGGLE_OUT_DIR
    resolved_scratch = (
        Path(args.scratch_dir) if args.scratch_dir else KAGGLE_SCRATCH_DIR
    )

    logger.info("\n%s", "=" * 75)
    logger.info("🔥 IGNITING KAGGLE VECTOR FORGE (ZERO-IPC BATCHING)")
    logger.info("📂 Source: %s", resolved_in)
    logger.info("💾 Target: %s", resolved_out)
    logger.info("=" * 75)

    try:
        execute_pipeline(resolved_in, resolved_out, resolved_scratch)
        logger.info("\n%s", "=" * 75)
        logger.info("🎉 ALL ASSIGNED VECTORS LOCKED.")
        logger.info("=" * 75)
    except Exception:
        logger.exception("CRITICAL FAILURE: Kaggle Vector Forge aborted abnormally.")
        sys.exit(1)


if __name__ == "__main__":
    main()
