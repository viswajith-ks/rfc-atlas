"""Hierarchy-aware chunking pipeline for normalizing RFC blocks into LanceDB-ready JSONL tables."""

import gc
import json
import logging
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TextIO, TypedDict

DATA_DIR: Path = Path("data")
NORMALIZED_DIR: Path = DATA_DIR / "normalized"
CHUNKS_DIR: Path = DATA_DIR / "chunks"
LOGS_DIR: Path = DATA_DIR / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

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


class ChunkingPipeline:
    """Manages the chunking, routing, and serialization of normalized RFC blocks."""

    def __init__(self) -> None:
        """Initializes the pipeline state and file handle registries."""
        self.total_blocks_processed: int = 0
        self.total_chunks_generated: int = 0
        self.table_file_handles: dict[str, TextIO] = {}

    def split_text_with_overlap(self, text: str) -> list[str]:
        """Splits continuous text strings using a sliding window with overlap.

        Args:
            text (str): The raw text payload to be chunked.

        Returns:
            List[str]: A list of sequential text chunks.
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
        """Generates chunks for a single document block and appends them to the routing table.

        Args:
            block (Dict[str, Any]): The document block payload.
            rfc_number (str): The associated RFC identifier.
            h_path (List[str]): The hierarchical section path of the block.
            rfc_metadata (Dict[str, Any]): The document-level metadata.
        """
        b_type: str = block.get("block_type", "paragraph")
        target_table: str = TABLE_ROUTING_MAP.get(b_type, "prose")
        text_payload: str = block.get("normalized_text", "")

        if not text_payload.strip():
            return

        text_fragments: list[str] = self.split_text_with_overlap(text_payload)
        block_id: str = block.get("block_id", f"rfc{rfc_number}-unknown")

        for i, fragment in enumerate(text_fragments):
            chunk_id: str = f"{block_id}-chunk{i:03d}"

            chunk_obj: ChunkRecord = {
                "chunk_id": chunk_id,
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

            chunk_json: str = json.dumps(chunk_obj)
            self.table_file_handles[target_table].write(chunk_json + "\n")
            self.total_chunks_generated += 1

        self.total_blocks_processed += 1

    def _process_sections_recursively(
        self,
        sections: list[dict[str, Any]],
        rfc_number: str,
        rfc_metadata: dict[str, Any],
    ) -> None:
        """Traverses the document section hierarchy to extract nested blocks.

        Args:
            sections (List[Dict[str, Any]]): The list of section nodes to process.
            rfc_number (str): The associated RFC identifier.
            rfc_metadata (Dict[str, Any]): The document-level metadata.
        """
        for section in sections:
            h_path: list[str] = section.get("hierarchy_path", [])
            for block in section.get("blocks", []):
                self._chunk_and_route(block, rfc_number, h_path, rfc_metadata)

            if "children" in section:
                self._process_sections_recursively(
                    section["children"], rfc_number, rfc_metadata
                )

    def process_document(self, filepath: Path) -> None:
        """Reads a canonical JSON artifact and initiates block extraction.

        Args:
            filepath (Path): The file path to the canonical JSON document.
        """
        try:
            with open(filepath, encoding="utf-8") as f:
                doc: dict[str, Any] = json.load(f)
        except Exception as e:
            logger.error(f"[{filepath.name}] Failed to parse JSON: {e}")
            return

        rfc_number: str = str(doc.get("metadata", {}).get("rfc_number", "unknown"))

        for block in doc.get("preface_blocks", []):
            self._chunk_and_route(block, rfc_number, ["Preface"], doc["metadata"])

        self._process_sections_recursively(
            doc.get("sections", []), rfc_number, doc["metadata"]
        )

        del doc

    def _update_ticker(self, current: int, total: int) -> None:
        """Outputs a real-time progress indicator to stderr.

        Args:
            current (int): The current file index.
            total (int): The total number of files to process.
        """
        if sys.stderr.isatty():
            sys.stderr.write(
                f"\r\033[K[Processing] Files: {current}/{total} | Blocks: {self.total_blocks_processed:,} | Chunks: {self.total_chunks_generated:,}"
            )
            sys.stderr.flush()

    def run(self) -> None:
        """Executes the pipeline sequentially across all available normalized artifacts."""
        print("Initializing Phase 2 Chunking Pipeline...")

        with ExitStack() as stack:
            for table_name in set(TABLE_ROUTING_MAP.values()):
                file_path: Path = CHUNKS_DIR / f"{table_name}.jsonl"
                self.table_file_handles[table_name] = stack.enter_context(
                    open(file_path, "w", encoding="utf-8")
                )

            try:
                json_files: list[Path] = list(NORMALIZED_DIR.glob("*.json"))
                total_files: int = len(json_files)
                print(f"Discovered {total_files} files. Commencing chunking...")

                for idx, filepath in enumerate(json_files, 1):
                    self.process_document(filepath)
                    self._update_ticker(idx, total_files)

                    if idx % 100 == 0:
                        gc.collect()

            except KeyboardInterrupt:
                print("\nPipeline manually interrupted.")
            except Exception as e:
                logger.critical(f"Pipeline crashed with FATAL ERROR: {e}")
                raise
            finally:
                if sys.stderr.isatty():
                    sys.stderr.write("\n")
                    sys.stderr.flush()

        print(
            f"Pipeline complete. Output: {self.total_chunks_generated:,} chunks saved to {CHUNKS_DIR}."
        )


def main() -> None:
    pipeline = ChunkingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
