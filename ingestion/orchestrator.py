"""Pipeline orchestration engine for managing parallel RFC ingestion, parsing, and telemetry collection."""

import json
import logging
import os
import re
import sys
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, TimeoutError, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeAlias

from pebble import ProcessPool

from ingestion.manifest import DatasetManifest, TelemetryRecord
from metadata.index_parser import RFCIndexParser
from normalization.normative_extractor import NormativeExtractor
from normalization.tree_builder import CanonicalTreeBuilder
from parsers.txt_parser import LegacyTextParser
from parsers.xml_parser import ModernRFCParser

logger = logging.getLogger(__name__)

LEGACY_RFC_LIMIT = 8650

_worker_tree_builder: CanonicalTreeBuilder | None = None
WorkerFuture: TypeAlias = Future[
    tuple[str, Literal["success", "failed"], str | None, TelemetryRecord | None]
]


def _execute_rfc_parsing_worker(
    filepath: Path, rfc_num: int, source_type: Literal["txt", "xml"], output_dir: Path
) -> tuple[str, Literal["success", "failed"], str | None, TelemetryRecord | None]:
    """Parses and structures an individual RFC file inside a process-isolated worker.

    Args:
        filepath (Path): Path to the target source file.
        rfc_num (int): Numeric identifier of the RFC document.
        source_type (Literal["txt", "xml"]): Format type of the source document.
        output_dir (Path): Destination directory for the normalized JSON output.

    Returns:
        tuple[str, Literal["success", "failed"], str | None, TelemetryRecord | None]: Filename, execution status,
            error message string if failed, and telemetry metrics if successful.
    """
    filename = filepath.name
    global _worker_tree_builder

    if _worker_tree_builder is None:
        return (
            filename,
            "failed",
            "Worker process tree builder was not initialized properly.",
            None,
        )

    try:
        if source_type == "txt":
            parser = LegacyTextParser(filepath)
        else:
            parser = ModernRFCParser(filepath)

        canonical_blocks = parser.parse_document()
        extractor = NormativeExtractor()
        enriched_blocks = extractor.process_blocks(canonical_blocks)
        valid_blocks = [b for b in enriched_blocks if b["normalized_text"].strip()]

        canonical_tree = _worker_tree_builder.build_tree(
            rfc_number=rfc_num,
            flat_blocks=valid_blocks,
            source_type=source_type,
        )

        output_path = output_dir / f"rfc{rfc_num}_normalized.json"
        tmp_output_path = output_dir / f"rfc{rfc_num}_normalized.tmp"

        canonical_tree.save_to_disk(tmp_output_path)
        os.replace(tmp_output_path, output_path)

        all_instantiated_blocks = canonical_tree.preface_blocks + [
            block for section in canonical_tree.sections for block in section.blocks
        ]

        telemetry_record: TelemetryRecord

        if not all_instantiated_blocks:
            telemetry_record = {
                "file": filename,
                "status": "success",
                "total_blocks": 0,
                "normative_rules": 0,
                "total_chars": 0,
                "max_block_chars": 0,
                "min_block_chars": 0,
            }
        else:
            lengths = [len(b.normalized_text) for b in all_instantiated_blocks]
            exact_total_chars = sum(lengths)
            normative_count = sum(
                len(b.normative_statements) for b in all_instantiated_blocks
            )

            telemetry_record = {
                "file": filename,
                "status": "success",
                "total_blocks": len(all_instantiated_blocks),
                "normative_rules": normative_count,
                "total_chars": exact_total_chars,
                "max_block_chars": max(lengths),
                "min_block_chars": min(lengths),
            }

        return filename, "success", None, telemetry_record

    except Exception:
        error_msg = traceback.format_exc().replace("\n", " | ")
        return filename, "failed", error_msg, None


class PipelineOrchestrator:
    """Coordinates multi-process parsing workflows across historical and modern RFC document layers."""

    _POLL_INTERVAL: float = 1.0

    def __init__(
        self,
        raw_txt_dir: Path,
        raw_xml_dir: Path,
        output_dir: Path,
        manifest_dir: Path,
        raw_index_path: Path,
        metadata_path: Path,
        parser_version: str = "1.0.0",
        chunking_version: str = "1.0.0",
        max_workers: int | None = None,
        per_file_timeout: float = 45.0,
    ) -> None:
        """Initializes pipeline paths and processing configurations in memory.

        Note: This orchestrator relies heavily on Linux-native Copy-on-Write (CoW)
        via `fork()` to share the massive metadata index across worker processes
        with zero memory overhead. It is explicitly pinned to Linux platforms.

        Args:
            raw_txt_dir (Path): Directory containing legacy plaintext RFC documents.
            raw_xml_dir (Path): Directory containing modern XML RFC documents.
            output_dir (Path): Target directory for saving canonical JSON records.
            manifest_dir (Path): Target directory for manifests and execution logs.
            raw_index_path (Path): Path to the global rfc-index.xml source file.
            metadata_path (Path): Destination path for the compiled metadata lookup file.
            parser_version (str): Semantic version tracking the structural parser logic.
            chunking_version (str): Semantic version tracking the hierarchy chunking strategy.
            max_workers (int | None): Cap on the maximum number of concurrent child processes.
            per_file_timeout (float): Execution timeout threshold per individual file in seconds.

        Raises:
            RuntimeError: If initialized on a non-Linux operating system (Windows/macOS).
        """

        if sys.platform != "linux":
            logger.critical("Initialization aborted: Incompatible Host OS detected.")
            raise RuntimeError(
                "Unsupported Operating System. This high-throughput ingestion pipeline relies "
                "on Linux's native `fork()` process spawning to share the metadata lookup tree "
                "across workers via Copy-on-Write (CoW). Running this on Windows or macOS "
                "(which default to `spawn()`) breaks memory isolation and will crash the workers. "
                "Please run this pipeline inside a Linux environment, WSL, or a Docker container."
            )

        self.raw_txt_dir = raw_txt_dir
        self.raw_xml_dir = raw_xml_dir
        self.output_dir = output_dir
        self.manifest_dir = manifest_dir
        self.raw_index_path = raw_index_path
        self.metadata_path = metadata_path

        self.dataset_manifest_path = self.manifest_dir / "dataset_manifest.json"
        self.telemetry_log_path = self.manifest_dir / "telemetry_log.json"
        self.telemetry_manifest: list[TelemetryRecord] = []

        self.parser_version = parser_version
        self.chunking_version = chunking_version
        self.max_workers = max_workers
        self.per_file_timeout = per_file_timeout

    @classmethod
    def create_and_initialize(
        cls,
        raw_txt_dir: Path,
        raw_xml_dir: Path,
        output_dir: Path,
        manifest_dir: Path,
        raw_index_path: Path,
        metadata_path: Path,
        parser_version: str = "1.0.0",
        chunking_version: str = "1.0.0",
        max_workers: int | None = None,
        per_file_timeout: float = 45.0,
    ) -> "PipelineOrchestrator":
        """Pre-provisions directories and updates metadata indices before returning an orchestrator instance.

        Args:
            raw_txt_dir (Path): Directory containing legacy plaintext RFC documents.
            raw_xml_dir (Path): Directory containing modern XML RFC documents.
            output_dir (Path): Target directory for saving canonical JSON records.
            manifest_dir (Path): Target directory for manifests and execution logs.
            raw_index_path (Path): Path to the global rfc-index.xml source file.
            metadata_path (Path): Destination path for the compiled metadata lookup file.
            parser_version (str): Semantic version tracking the structural parser logic.
            chunking_version (str): Semantic version tracking the hierarchy chunking strategy.
            max_workers (int | None): Cap on the maximum number of concurrent child processes.
            per_file_timeout (float): Execution timeout threshold per individual file in seconds.

        Returns:
            PipelineOrchestrator: An initialized orchestrator instance ready for execution loops.

        Raises:
            FileNotFoundError: If the core rfc-index.xml file is missing from disk.
            RuntimeError: If the index parser fails to compile the metadata lookup.
        """

        if not raw_index_path.exists():
            logger.critical(
                f"Foundational raw RFC Index file not found at target directory: {raw_index_path}"
            )
            raise FileNotFoundError(
                f"Missing baseline protocol schema dependency. The raw global RFC index "
                f"file must exist at '{raw_index_path}' before running the pipeline."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if (
            metadata_path.exists()
            and metadata_path.stat().st_mtime >= raw_index_path.stat().st_mtime
        ):
            logger.info(
                f"Metadata lookup cache is up-to-date at {metadata_path}. Skipping XML re-parsing."
            )
        else:
            logger.info(
                "Metadata lookup cache is missing or stale. Compiling index ledger..."
            )
            try:
                index_parser = RFCIndexParser(raw_index_path, metadata_path)
                index_parser.parse()
            except Exception as e:
                logger.critical(f"Core metadata index failed to parse: {e}")
                raise RuntimeError(
                    f"Pipeline initialization aborted. Metadata compilation failed: {e}"
                ) from e

        return cls(
            raw_txt_dir=raw_txt_dir,
            raw_xml_dir=raw_xml_dir,
            output_dir=output_dir,
            manifest_dir=manifest_dir,
            raw_index_path=raw_index_path,
            metadata_path=metadata_path,
            parser_version=parser_version,
            chunking_version=chunking_version,
            max_workers=max_workers,
            per_file_timeout=per_file_timeout,
        )

    @staticmethod
    def _extract_rfc_num(source: Path) -> int:
        """Extracts the numeric identifier integer from an RFC filename string or path.

        Args:
            source (Path): Input filename string or file path.

        Returns:
            int: Isolated numeric component of the RFC filename, or 0 if unmatchable.
        """
        filename = source.name

        # Matches a string strictly beginning with "rfc" (case-insensitive) followed by digits and a literal dot.
        # Isolates the ID from filenames like rfc1234.txt.
        match = re.match(r"^rfc(\d+)\.", filename, re.IGNORECASE)

        return int(match.group(1)) if match else 0

    def _execute_era_ingestion(self, source_type: Literal["txt", "xml"]) -> None:
        """Discovers, filters, and processes all RFC source files matching an execution era.

        Args:
            source_type (Literal["txt", "xml"]): Targeted source format for directory scans.
        """
        if source_type == "xml":
            target_dir = self.raw_xml_dir
            glob_pattern = "rfc*.xml"
            log_prefix = "Modern"
            era_label = f"Modern Era (RFC {LEGACY_RFC_LIMIT}+) XML"
        else:
            target_dir = self.raw_txt_dir
            glob_pattern = "rfc*.txt"
            log_prefix = "Legacy"
            era_label = f"Legacy Era (RFC 1 - {LEGACY_RFC_LIMIT - 1}) Text"

        logger.info(f"Starting {era_label} Ingestion...")
        logger.info("-" * 50)

        raw_files = list(target_dir.glob(glob_pattern))

        file_num_pairs = ((f, self._extract_rfc_num(f)) for f in raw_files)

        if source_type == "xml":
            valid_pairs = ((f, n) for f, n in file_num_pairs if n > 0)
        else:
            valid_pairs = (
                (f, n) for f, n in file_num_pairs if 0 < n < LEGACY_RFC_LIMIT
            )

        sorted_files = [f for f, _ in sorted(valid_pairs, key=lambda x: x[1])]

        success, failed = self._run_parallel_ingestion_pool(
            sorted_files, source_type, log_prefix
        )

        logger.info("-" * 50)
        logger.info(
            f"--- {log_prefix.upper()} ERA COMPLETE | Success: {success} | Failed: {failed} ---"
        )

    def _print_progress_ticker(
        self, log_prefix: str, filename: str, success: int, failed: int, total: int
    ) -> None:
        """Outputs a unified pipeline processing ticker directly to stderr.

        Args:
            log_prefix (str): Prefix specifying the operational ingestion category.
            filename (str): Name of the file undergoing processing.
            success (int): Cumulative count of successfully extracted files.
            failed (int): Cumulative count of processing execution failures.
            total (int): Total file count scheduled for processing inside the current batch.
        """
        if sys.stderr.isatty():
            sys.stderr.write(
                f"\r\033[KProcessing {log_prefix}: {filename}... Success: {success} | Failed: {failed} | Total: {total}"
            )
            sys.stderr.flush()

    def _run_parallel_ingestion_pool(
        self,
        target_files: list[Path],
        source_type: Literal["txt", "xml"],
        log_prefix: str,
    ) -> tuple[int, int]:
        """Manages process pool worker allocation using a sliding task submission strategy.

        Args:
            target_files (list[Path]): Target collection of file paths targeted for extraction.
            source_type (Literal["txt", "xml"]): Era format selector string.
            log_prefix (str): Label prefix for progress tracking outputs.

        Returns:
            tuple[int, int]: Accumulated success count and task failure count across the run.
        """
        success_count = 0
        failure_count = 0
        total_files = len(target_files)

        if total_files == 0:
            logger.warning(
                f"No targeting elements identified for {log_prefix} Ingestion."
            )
            return 0, 0

        if self.max_workers is not None:
            allocated_cores = max(1, self.max_workers)
            logger.info(
                f"Spawning concurrent environment with user-configured cap: {allocated_cores} CPU cores."
            )
        else:
            detected_cores = os.cpu_count() or 2
            allocated_cores = min(max(1, detected_cores - 1), 8)
            logger.info(
                f"Spawning concurrent environment utilizing safety fallback: {allocated_cores} CPU cores."
            )

        global _worker_tree_builder
        if _worker_tree_builder is None:
            logger.info(
                "Pre-loading canonical tree builder metadata index into parent memory..."
            )
            _worker_tree_builder = CanonicalTreeBuilder(self.metadata_path)

        # Target workers to recycle after processing roughly 25% of their total expected workload share.
        calculated_max_tasks = round(0.25 * total_files / allocated_cores)

        try:
            with ProcessPool(
                max_workers=allocated_cores,
                max_tasks=calculated_max_tasks,
            ) as pool:
                file_iterator = iter(target_files)
                future_to_rfc: dict[WorkerFuture, Path] = {}
                initial_buffer_size = allocated_cores * 2

                for _ in range(initial_buffer_size):
                    try:
                        filepath = next(file_iterator)
                        rfc_num = self._extract_rfc_num(filepath)
                        future = pool.schedule(  # type: ignore
                            _execute_rfc_parsing_worker,
                            args=[filepath, rfc_num, source_type, self.output_dir],
                            timeout=self.per_file_timeout,
                        )
                        future_to_rfc[future] = filepath
                    except StopIteration:
                        break

                pending_futures: set[WorkerFuture] = set(future_to_rfc.keys())

                while pending_futures:
                    done_batch, pending_futures = wait(
                        pending_futures,
                        timeout=self._POLL_INTERVAL,
                        return_when=FIRST_COMPLETED,
                    )

                    for future in done_batch:
                        filepath = future_to_rfc[future]
                        filename = filepath.name

                        try:
                            filename, status, error_msg, telemetry = future.result()

                            if status == "success" and telemetry is not None:
                                if telemetry.get("total_blocks") == 0:
                                    logger.warning(
                                        f"File {filename} processed successfully but produced 0 semantic blocks."
                                    )
                                self._record_telemetry(
                                    filename, "success", None, telemetry
                                )
                                success_count += 1
                            else:
                                self._record_telemetry(
                                    filename, "failed", error_msg, None
                                )
                                failure_count += 1

                        except TimeoutError:
                            timeout_err = f"Execution exceeded per-file timeout limit of {self.per_file_timeout}s."
                            logger.warning(
                                f"Worker process for {filename} exceeded runtime limits and was terminated."
                            )
                            self._record_telemetry(
                                filename, "failed", timeout_err, None
                            )
                            failure_count += 1

                        except Exception as exc:
                            error_msg = f"Task execution error or abrupt process cancellation: {exc}"
                            self._record_telemetry(filename, "failed", error_msg, None)
                            failure_count += 1

                        self._print_progress_ticker(
                            log_prefix,
                            filename,
                            success_count,
                            failure_count,
                            total_files,
                        )

                        try:
                            filepath = next(file_iterator)
                            rfc_num = self._extract_rfc_num(filepath)
                            new_future = pool.schedule(  # type: ignore
                                _execute_rfc_parsing_worker,
                                args=[filepath, rfc_num, source_type, self.output_dir],
                                timeout=self.per_file_timeout,
                            )
                            future_to_rfc[new_future] = filepath
                            pending_futures.add(new_future)
                        except StopIteration:
                            pass

                        del future_to_rfc[future]
        finally:
            if sys.stderr.isatty():
                sys.stderr.write("\n")
                sys.stderr.flush()

        return success_count, failure_count

    def run_legacy_text_ingestion(self) -> None:
        """Processes historical documents (RFC 1 - 8649) via the heuristic plaintext parsing workflow."""
        self._execute_era_ingestion("txt")

    def run_modern_xml_ingestion(self) -> None:
        """Processes modern documents (RFC 8650+) via the structural XML parsing workflow."""
        self._execute_era_ingestion("xml")

    def _record_telemetry(
        self,
        filename: str,
        status: Literal["success", "failed"] = "success",
        error_msg: str | None = None,
        precomputed_telemetry: TelemetryRecord | None = None,
    ) -> None:
        """Appends processing records or errors directly onto the runtime orchestrator tracking ledger.

        Args:
            filename (str): Name of the targeted document file.
            status (Literal["success", "failed"]): Operational execution state indicator.
            error_msg (str | None): Trace string containing failure data if execution failed.
            precomputed_telemetry (TelemetryRecord | None): Worker compiled statistical profile payload.
        """
        if status == "success" and precomputed_telemetry is not None:
            self.telemetry_manifest.append(precomputed_telemetry)
        else:
            clean_error = (
                error_msg
                if error_msg is not None
                else "Unknown internal processing failure."
            )
            self.telemetry_manifest.append(
                {"file": filename, "status": status, "error": clean_error}
            )

    def save_manifest(self) -> None:
        """Aggregates telemetry records to generate the final deployment contract and audit logs on disk."""
        self.telemetry_manifest.sort(
            key=lambda x: self._extract_rfc_num(Path(x["file"]))
        )
        logger.info("Compiling Final Dataset Manifest...")

        successful = [r for r in self.telemetry_manifest if r["status"] == "success"]
        total_blocks = sum(r.get("total_blocks", 0) for r in successful)
        total_normative = sum(r.get("normative_rules", 0) for r in successful)

        txt_count = sum(1 for r in successful if r["file"].endswith(".txt"))
        xml_count = sum(1 for r in successful if r["file"].endswith(".xml"))

        now = datetime.now(UTC)
        version_stamp = now.strftime("%Y-%m-%d-%H%M")

        manifest = DatasetManifest(
            dataset_version=version_stamp,
            pipeline_run_at=now,
            parser_version=self.parser_version,
            chunking_version=self.chunking_version,
            total_rfcs_indexed=len(successful),
            total_blocks_generated=total_blocks,
            total_normative_statements=total_normative,
            xml_rfcs_processed=xml_count,
            txt_rfcs_processed=txt_count,
        )

        manifest.save_to_disk(self.dataset_manifest_path)
        logger.info(f"Receipt saved to {self.dataset_manifest_path}.")

        self.telemetry_log_path.write_text(
            json.dumps(self.telemetry_manifest, indent=2), encoding="utf-8"
        )
        logger.info(f"Telemetry log saved to {self.telemetry_log_path}.")
