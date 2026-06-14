"""Hierarchy-aware chunking pipeline using Pebble for multi-core scatter-gather execution."""

import gc
import json
import logging
import os
import shutil
import sys
from concurrent.futures import Future, TimeoutError, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TextIO

from pebble import ProcessPool

from chunking.schema import TABLE_ROUTING_MAP, ChunkRecord

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
NORMALIZED_DIR: Path = DATA_DIR / "normalized"
CHUNKS_DIR: Path = DATA_DIR / "chunks"
TMP_DIR: Path = CHUNKS_DIR / "tmp_workers"
LOGS_DIR: Path = DATA_DIR / "logs"

logger = logging.getLogger(__name__)

CHUNK_SIZE_LIMIT: int = 2000
OVERLAP_SIZE: int = 250
BATCH_SIZE: int = 50
WORKER_TIMEOUT: int = 180


class BatchChunker:
    """Isolated worker class that processes a specific batch of files."""

    def __init__(self, batch_id: int) -> None:
        """Initializes state for an isolated worker handling a specific file batch.

        Args:
            batch_id (int): The unique sequential identifier for this worker's batch.
        """
        self.batch_id = batch_id
        self.blocks_processed: int = 0
        self.chunks_generated: int = 0
        self.handles: dict[str, TextIO] = {}

    def split_text_with_overlap(self, text: str) -> list[str]:
        """Splits continuous text strings using a sliding window with overlap.

        Calculates split boundaries prioritizing newlines or spaces to avoid
        slicing words in half, ensuring dense context for vector embeddings.

        Args:
            text (str): The raw text payload to be chunked.

        Returns:
            list[str]: A list of sequential, overlapping text chunks.
        """
        if not text:
            return []
        if len(text) <= CHUNK_SIZE_LIMIT:
            return [text]

        chunks: list[str] = []
        start: int = 0
        text_len: int = len(text)

        while start < text_len:
            end: int = min(start + CHUNK_SIZE_LIMIT, text_len)

            if end < text_len:
                window_start = max(start, end - OVERLAP_SIZE)
                newline_pos = text.rfind("\n", window_start, end)
                if newline_pos != -1 and newline_pos > start:
                    end = newline_pos + 1
                else:
                    space_pos = text.rfind(" ", window_start, end)
                    if space_pos != -1 and space_pos > start:
                        end = space_pos + 1

            chunks.append(text[start:end])

            if end >= text_len:
                break

            next_start: int = end - OVERLAP_SIZE

            if next_start <= start:
                next_start = start + 1

            start = next_start

        return chunks

    def _chunk_and_route(
        self,
        block: dict[str, Any],
        rfc_number: str,
        h_path: list[str],
        rfc_metadata: dict[str, Any],
    ) -> None:
        """Chunks a single document block and writes it to the appropriate isolated table log.

        Extracts the payload, determines the table routing (e.g., 'prose', 'sourcecode'),
        generates the sliding window fragments, and serializes them to disk.

        Args:
            block (dict[str, Any]): The document block payload from the normalized JSON.
            rfc_number (str): The RFC identifier (e.g., "6716").
            h_path (list[str]): The hierarchical section breadcrumb path of the block.
            rfc_metadata (dict[str, Any]): Document-level metadata injected into every chunk.
        """
        b_type: str = block.get("block_type", "paragraph")
        target_table: str = TABLE_ROUTING_MAP.get(b_type, "prose")
        text_payload: str = block.get("normalized_text", "")

        if not text_payload.strip():
            return

        text_fragments: list[str] = self.split_text_with_overlap(text_payload)
        block_id: str = block.get("block_id", f"rfc{rfc_number}-unknown")

        for i, fragment in enumerate(text_fragments):
            chunk_obj: ChunkRecord = {
                "chunk_id": f"{block_id}-chunk{i:03d}",
                "rfc_number": rfc_number,
                "block_type": b_type,
                "table_route": target_table,
                "hierarchy_path": " > ".join(h_path),
                "text_payload": fragment,
                "sourcecode_type": block.get("sourcecode_type"),
                "parsing_confidence": block.get("parsing_confidence", 1.0),
                "normative_statements": block.get("normative_statements", []),
                "rfc_title": rfc_metadata.get("title"),
                "status": rfc_metadata.get("status"),
            }
            self.handles[target_table].write(json.dumps(chunk_obj) + "\n")
            self.chunks_generated += 1

        self.blocks_processed += 1

    def run(self, file_paths: list[Path]) -> dict[str, int]:
        """Executes the chunking pipeline sequentially across the assigned batch.

        Args:
            file_paths (list[Path]): The list of canonical JSON file paths for this batch.

        Returns:
            dict[str, int]: A metrics dictionary containing:
                - 'blocks': Total normalized blocks processed.
                - 'chunks': Total overlapping chunks generated.
        """
        with ExitStack() as stack:
            for table_name in set(TABLE_ROUTING_MAP.values()):
                tmp_file = TMP_DIR / f"{table_name}_batch_{self.batch_id}.jsonl"
                self.handles[table_name] = stack.enter_context(
                    open(tmp_file, "w", encoding="utf-8")
                )

            for filepath in file_paths:
                try:
                    logger.info(f"[Batch {self.batch_id}] Starting {filepath.name}")
                    for handler in logger.handlers:
                        handler.flush()

                    with open(filepath, encoding="utf-8") as f:
                        doc = json.load(f)

                    rfc_number: str = str(
                        doc.get("metadata", {}).get("rfc_number", "unknown")
                    )

                    for block in doc.get("preface_blocks", []):
                        self._chunk_and_route(
                            block, rfc_number, ["Preface"], doc["metadata"]
                        )

                    for section in doc.get("sections", []):
                        h_path: list[str] = section.get("hierarchy_path", [])
                        for block in section.get("blocks", []):
                            self._chunk_and_route(
                                block, rfc_number, h_path, doc["metadata"]
                            )

                except Exception as e:
                    logger.error(
                        f"[Batch {self.batch_id}] Failed on {filepath.name}: {e}"
                    )
                finally:
                    gc.collect()
                    for handle in self.handles.values():
                        handle.flush()
                        os.fsync(handle.fileno())

            return {"blocks": self.blocks_processed, "chunks": self.chunks_generated}


def worker_task(batch_id: int, file_paths: list[Path]) -> dict[str, int]:
    """Isolated worker process entry point for Pebble ProcessPool.

    Configures local, process-isolated logging to prevent thread contention,
    then instantiates and runs the BatchChunker.

    Args:
        batch_id (int): The unique sequential identifier for this batch.
        file_paths (list[Path]): The subset of JSON files assigned to this worker.

    Returns:
        dict[str, int]: Batch execution metrics (blocks and chunks processed).
    """
    worker_logger = logging.getLogger(__name__)
    worker_logger.handlers.clear()

    log_path = TMP_DIR / f"batch_{batch_id}_errors.log"
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    worker_logger.addHandler(fh)

    chunker = BatchChunker(batch_id)
    return chunker.run(file_paths)


def get_optimal_workers() -> int:
    """Calculates the optimal core count for the multiprocessing pool.

    Reserves one core for the OS and the main Python orchestrator thread
    to prevent UI lockup and system thrashing.

    Returns:
        int: The number of worker processes to spawn.
    """
    cpu_count = os.cpu_count() or 4
    return max(1, cpu_count - 1)


def gather_files(total_batches: int) -> None:
    """Concatenates all isolated worker files and logs into master outputs atomically.

    Uses temporary files and OS-level atomic replacement to guarantee that
    a crash during gathering never leaves a corrupted/partial file in the chunks directory.

    Args:
        total_batches (int): The total number of batches processed.
    """
    print(
        "\n[PHASE] Gather Phase: Concatenating worker chunks and logs into master files..."
    )

    unique_tables = set(TABLE_ROUTING_MAP.values())
    COPY_BUFFER_SIZE = 16 * 1024 * 1024

    for table_name in unique_tables:
        master_path = CHUNKS_DIR / f"{table_name}.jsonl"
        tmp_master_path = CHUNKS_DIR / f"{table_name}.jsonl.tmp"

        with open(tmp_master_path, "wb") as master_file:
            for batch_id in range(total_batches):
                worker_tmp_path = TMP_DIR / f"{table_name}_batch_{batch_id}.jsonl"
                if worker_tmp_path.exists():
                    with open(worker_tmp_path, "rb") as tmp_file:
                        shutil.copyfileobj(
                            tmp_file, master_file, length=COPY_BUFFER_SIZE
                        )

        os.replace(tmp_master_path, master_path)

    master_log_path = LOGS_DIR / "chunking_pipeline.log"
    tmp_log_path = LOGS_DIR / "chunking_pipeline.log.tmp"

    with open(tmp_log_path, "wb") as master_log:
        orch_log = TMP_DIR / "orchestrator_errors.log"
        if orch_log.exists():
            with open(orch_log, "rb") as f:
                shutil.copyfileobj(f, master_log)

        for batch_id in range(total_batches):
            worker_tmp_log = TMP_DIR / f"batch_{batch_id}_errors.log"
            if worker_tmp_log.exists():
                with open(worker_tmp_log, "rb") as f:
                    shutil.copyfileobj(f, master_log)

    os.replace(tmp_log_path, master_log_path)

    shutil.rmtree(TMP_DIR)
    print(
        "[SUCCESS] Gather complete. Temporary files wiped. Master tables locked atomically."
    )


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    logger.handlers.clear()
    orch_fh = logging.FileHandler(
        TMP_DIR / "orchestrator_errors.log", mode="w", encoding="utf-8"
    )
    orch_fh.setFormatter(
        logging.Formatter("%(asctime)s [ORCHESTRATOR] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(orch_fh)

    json_files: list[Path] = list(NORMALIZED_DIR.glob("*.json"))
    total_files = len(json_files)

    if total_files == 0:
        print("[WARN] No normalized JSON files found. Exiting.")
        return

    batches = [
        json_files[i : i + BATCH_SIZE] for i in range(0, total_files, BATCH_SIZE)
    ]
    total_batches = len(batches)
    optimal_workers = get_optimal_workers()

    print("====================================================")
    print("[INIT] INITIATING PEBBLE SCATTER-GATHER PIPELINE")
    print(
        f"       Files: {total_files:,} | Worker Threads: {optimal_workers} | Batches: {total_batches}"
    )
    print("====================================================\n")

    global_blocks = 0
    global_chunks = 0
    completed_batches = 0

    with ProcessPool(max_workers=optimal_workers) as pool:
        future_map: dict[Future[dict[str, int]], int] = {}
        for batch_id, batch_files in enumerate(batches):
            future: Future[dict[str, int]] = pool.schedule(  # type: ignore
                worker_task, args=[batch_id, batch_files], timeout=WORKER_TIMEOUT
            )
            future_map[future] = batch_id

        for future in as_completed(future_map):
            batch_id = future_map[future]
            try:
                result = future.result()
                global_blocks += result["blocks"]
                global_chunks += result["chunks"]
                completed_batches += 1

                sys.stderr.write(
                    f"\r\033[K[PROCESS] Batches: {completed_batches}/{total_batches} "
                    f"| Blocks: {global_blocks:,} | Chunks: {global_chunks:,}"
                )
                sys.stderr.flush()

            except TimeoutError:
                logger.error(f"Batch {batch_id} timed out after {WORKER_TIMEOUT}s!")
                sys.stderr.write(f"\n[ERROR] Batch {batch_id} Timed Out. See Logs.\n")
                sys.stderr.flush()
            except Exception as error:
                logger.error(f"Batch {batch_id} raised a FATAL exception: {error}")
                sys.stderr.write(f"\n[ERROR] Batch {batch_id} Failed. See Logs.\n")
                sys.stderr.flush()

    print("\n\n[SUCCESS] Scatter phase complete.")
    gather_files(total_batches)

    print("\n====================================================")
    print(f"[SUCCESS] PIPELINE COMPLETE: {global_chunks:,} chunks generated safely!")
    print("====================================================")


if __name__ == "__main__":
    main()
