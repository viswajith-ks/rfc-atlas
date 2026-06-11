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
from typing import Any, TextIO, TypedDict

from pebble import ProcessPool

DATA_DIR: Path = Path("data")
NORMALIZED_DIR: Path = DATA_DIR / "normalized"
CHUNKS_DIR: Path = DATA_DIR / "chunks"
TMP_DIR: Path = CHUNKS_DIR / "tmp_workers"
LOGS_DIR: Path = DATA_DIR / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
file_handler = logging.FileHandler(
    LOGS_DIR / "chunking_pipeline.log", mode="w", encoding="utf-8"
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

CHUNK_SIZE_LIMIT: int = 2000
OVERLAP_SIZE: int = 250
BATCH_SIZE: int = 500
WORKER_TIMEOUT: int = 180

TABLE_ROUTING_MAP: dict[str, str] = {
    "paragraph": "prose",
    "list": "prose",
    "security": "security",
    "references": "references",
    "abnf": "abnf",
    "sourcecode": "sourcecode",
    "artwork": "artwork",
    "table": "table",
}


class ChunkRecord(TypedDict):
    """Schema defining a structured chunk prior to LanceDB ingestion."""

    chunk_id: str
    rfc_number: str
    block_type: str
    table_route: str
    hierarchy_path: str
    text_payload: str
    sourcecode_type: str | None
    parsing_confidence: float
    normative_statements: list[dict[str, Any]]
    rfc_title: str | None
    status: str | None


class BatchChunker:
    """Isolated worker class that processes a specific batch of files."""

    def __init__(self, batch_id: int) -> None:
        self.batch_id = batch_id
        self.blocks_processed: int = 0
        self.chunks_generated: int = 0
        self.handles: dict[str, TextIO] = {}

    def split_text_with_overlap(self, text: str) -> list[str]:
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

    def _process_sections(
        self,
        sections: list[dict[str, Any]],
        rfc_number: str,
        rfc_metadata: dict[str, Any],
    ) -> None:
        for section in sections:
            h_path: list[str] = section.get("hierarchy_path", [])
            for block in section.get("blocks", []):
                self._chunk_and_route(block, rfc_number, h_path, rfc_metadata)

            if "children" in section:
                self._process_sections(section["children"], rfc_number, rfc_metadata)

    def run(self, file_paths: list[Path]) -> dict[str, int]:
        """Executes the batch and manages isolated temporary files."""
        with ExitStack() as stack:
            for table_name in set(TABLE_ROUTING_MAP.values()):
                tmp_file = TMP_DIR / f"{table_name}_batch_{self.batch_id}.jsonl"
                self.handles[table_name] = stack.enter_context(
                    open(tmp_file, "w", encoding="utf-8")
                )

            for filepath in file_paths:
                try:
                    with open(filepath, encoding="utf-8") as f:
                        doc = json.load(f)

                    rfc_number: str = str(
                        doc.get("metadata", {}).get("rfc_number", "unknown")
                    )

                    for block in doc.get("preface_blocks", []):
                        self._chunk_and_route(
                            block, rfc_number, ["Preface"], doc["metadata"]
                        )

                    self._process_sections(
                        doc.get("sections", []), rfc_number, doc["metadata"]
                    )
                    del doc
                except Exception as e:
                    logger.error(
                        f"[Batch {self.batch_id}] Failed on {filepath.name}: {e}"
                    )

        gc.collect()
        return {"blocks": self.blocks_processed, "chunks": self.chunks_generated}


def worker_task(batch_id: int, file_paths: list[Path]) -> dict[str, int]:
    chunker = BatchChunker(batch_id)
    return chunker.run(file_paths)


def get_optimal_workers() -> int:
    """Calculates optimal process count, reserving a core for the OS."""
    cpu_count = os.cpu_count() or 4
    return max(1, cpu_count - 1)


def gather_files(total_batches: int) -> None:
    """Concatenates all isolated worker files into the final master JSONL tables."""
    print("\n[PHASE] Gather Phase: Concatenating worker chunks into master tables...")

    unique_tables = set(TABLE_ROUTING_MAP.values())

    COPY_BUFFER_SIZE = 16 * 1024 * 1024

    for table_name in unique_tables:
        master_path = CHUNKS_DIR / f"{table_name}.jsonl"

        with open(master_path, "wb") as master_file:
            for batch_id in range(total_batches):
                tmp_path = TMP_DIR / f"{table_name}_batch_{batch_id}.jsonl"
                if tmp_path.exists():
                    with open(tmp_path, "rb") as tmp_file:
                        shutil.copyfileobj(
                            tmp_file, master_file, length=COPY_BUFFER_SIZE
                        )

    shutil.rmtree(TMP_DIR)
    print("[SUCCESS] Gather complete. Temporary files wiped.")


def main() -> None:
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
