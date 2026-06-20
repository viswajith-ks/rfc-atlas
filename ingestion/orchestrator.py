"""Pipeline orchestration engine for managing parallel RFC ingestion, parsing, and telemetry collection."""

import json
import logging
import os
import re
import sys
import traceback
from collections.abc import Generator, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, TimeoutError, wait
from contextlib import contextmanager
from dataclasses import dataclass
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

_worker_tree_builder: CanonicalTreeBuilder | None = None
WorkerFuture: TypeAlias = Future[
    tuple[Literal["success", "failed"], str | None, TelemetryRecord | None]
]


def _execute_rfc_parsing_worker(
    filepath: Path, rfc_num: int, source_type: Literal["txt", "xml"], output_dir: Path
) -> tuple[Literal["success", "failed"], str | None, TelemetryRecord | None]:
    """Parses and structures an individual RFC file inside a process-isolated worker.

    Args:
        filepath (Path): Exact file path to the target RFC document.
        rfc_num (int): Numeric identifier of the RFC document.
        source_type (Literal["txt", "xml"]): Format type of the source document.
        output_dir (Path): Destination directory for the normalized JSON output.

    Returns:
        tuple[Literal["success", "failed"], str | None, TelemetryRecord | None]: Execution status,
            error message string if failed, and telemetry metrics if successful.
    """
    filename = filepath.name

    assert _worker_tree_builder is not None, (
        "FATAL: CoW global memory lost. Worker initialized improperly! "
        "Ensure your OS supports 'fork' multiprocessing contexts."
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

        lengths = [len(b.normalized_text) for b in all_instantiated_blocks]

        telemetry_record: TelemetryRecord = {
            "file": filename,
            "status": "success",
            "total_blocks": len(all_instantiated_blocks),
            "normative_rules": sum(
                len(b.normative_statements) for b in all_instantiated_blocks
            ),
            "total_chars": sum(lengths),
            "max_block_chars": max(lengths, default=0),
            "min_block_chars": min(lengths, default=0),
        }

        return "success", None, telemetry_record

    except Exception:
        error_msg = traceback.format_exc().replace("\n", " | ")
        return "failed", error_msg, None


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration container for the PipelineOrchestrator execution boundaries."""

    raw_txt_dir: Path
    raw_xml_dir: Path
    output_dir: Path
    manifest_dir: Path
    raw_index_path: Path
    metadata_path: Path
    parser_version: str = "1.0.0"
    chunking_version: str = "1.0.0"
    max_workers: int | None = None
    per_file_timeout: float = 45.0


class PipelineOrchestrator:
    """Coordinates multi-process parsing workflows across historical and modern RFC document layers."""

    _POLL_INTERVAL: float = 1.0
    _is_instantiated: bool = False

    def __init__(self, config: PipelineConfig) -> None:
        """Initializes pipeline paths and processing configurations in memory.

        Args:
            config (PipelineConfig): The unified configuration mapping.
        """
        if PipelineOrchestrator._is_instantiated:
            logger.critical("Initialization aborted: Orchestrator Singleton violation.")
            raise RuntimeError("PipelineOrchestrator is a strict Singleton...")

        if sys.platform != "linux":
            logger.critical("Initialization aborted: Incompatible Host OS detected.")
            raise RuntimeError("Unsupported Operating System...")

        PipelineOrchestrator._is_instantiated = True

        self.raw_txt_dir = config.raw_txt_dir
        self.raw_xml_dir = config.raw_xml_dir
        self.output_dir = config.output_dir
        self.manifest_dir = config.manifest_dir
        self.raw_index_path = config.raw_index_path
        self.metadata_path = config.metadata_path
        self.parser_version = config.parser_version
        self.chunking_version = config.chunking_version
        self.max_workers = config.max_workers
        self.per_file_timeout = config.per_file_timeout

        self.dataset_manifest_path = self.manifest_dir / "dataset_manifest.json"
        self.telemetry_log_path = self.manifest_dir / "telemetry_log.json"
        self.telemetry_manifest: list[TelemetryRecord] = []

    @classmethod
    @contextmanager
    def managed_instance(
        cls, config: PipelineConfig
    ) -> Generator["PipelineOrchestrator", None, None]:
        instance = cls(config)
        try:
            yield instance
        finally:
            cls.reset_state()

    @classmethod
    def create_and_initialize(cls, config: PipelineConfig) -> "PipelineOrchestrator":
        """Pre-provisions directories and updates metadata indices before returning an instance.

        Args:
            config (PipelineConfig): The unified configuration mapping.
        """
        if not config.raw_index_path.exists():
            logger.critical(
                f"Foundational raw RFC Index file not found: {config.raw_index_path}"
            )
            raise FileNotFoundError("Missing baseline protocol schema dependency.")

        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.manifest_dir.mkdir(parents=True, exist_ok=True)
        config.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if (
            config.metadata_path.exists()
            and config.metadata_path.stat().st_mtime
            >= config.raw_index_path.stat().st_mtime
        ):
            logger.info("Metadata lookup cache is up-to-date. Skipping XML re-parsing.")
        else:
            logger.info(
                "Metadata lookup cache is missing or stale. Compiling index ledger..."
            )
            try:
                index_parser = RFCIndexParser(
                    config.raw_index_path, config.metadata_path
                )
                index_parser.parse()
            except Exception as e:
                logger.critical(f"Core metadata index failed to parse: {e}")
                raise RuntimeError(f"Metadata compilation failed: {e}") from e

        return cls(config)

    @classmethod
    def reset_state(cls) -> None:
        """Resets the singleton instantiation lock and clears the global worker cache.

        WARNING: This is intended strictly for CI/CD test suite teardowns
        and should never be called during standard pipeline execution.
        """
        cls._is_instantiated = False

        global _worker_tree_builder
        _worker_tree_builder = None

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

    def _get_xml_covered_rfcs(self) -> frozenset[int]:
        """Returns the set of RFC numbers for which an XML source file exists in raw_xml_dir.

        Used during the txt ingestion pass to skip any RFC that already has a
        higher-fidelity XML counterpart. This replaces the old hard-coded
        LEGACY_RFC_LIMIT numeric cutoff, which would go stale as new RFCs are
        published and their XML backports become available.

        Returns:
            frozenset[int]: RFC numbers with a valid XML source file present on disk.
        """
        return frozenset(
            rfc_num
            for f in self.raw_xml_dir.glob("rfc*.xml")
            if (rfc_num := self._extract_rfc_num(f)) > 0
        )

    def _execute_era_ingestion(self, source_type: Literal["txt", "xml"]) -> None:
        """Discovers, filters, and processes all RFC source files matching an execution era.

        For the txt pass, any RFC whose number appears in the XML source directory is
        skipped — the XML pass will handle it with higher fidelity. This means the txt
        pass is self-adjusting: as XML backports are added to raw_xml_dir over time,
        they are automatically preferred without any configuration change.

        Args:
            source_type (Literal["txt", "xml"]): Targeted source format for directory scans.
        """
        if source_type == "xml":
            target_dir = self.raw_xml_dir
            glob_pattern = "rfc*.xml"
            log_prefix = "Modern"
            era_label = "Modern Era XML (Including Legacy Backports)"

            raw_files = list(target_dir.glob(glob_pattern))
            file_num_pairs = ((f, self._extract_rfc_num(f)) for f in raw_files)
            valid_pairs = ((f, n) for f, n in file_num_pairs if n > 0)

        else:
            target_dir = self.raw_txt_dir
            glob_pattern = "rfc*.txt"
            log_prefix = "Legacy"
            era_label = "Legacy Text (txt-only, skipping RFCs covered by XML)"

            # Build the exclusion set once before scanning txt files. Any RFC number
            # present here has an XML counterpart and will be handled at higher fidelity
            # by the subsequent xml pass, so we skip it entirely in the txt pass.
            xml_covered = self._get_xml_covered_rfcs()
            logger.info(
                f"Found {len(xml_covered):,} XML-covered RFCs. "
                f"Their txt counterparts will be skipped."
            )

            raw_files = list(target_dir.glob(glob_pattern))
            file_num_pairs = ((f, self._extract_rfc_num(f)) for f in raw_files)
            valid_pairs = (
                (f, n) for f, n in file_num_pairs if n > 0 and n not in xml_covered
            )

        logger.info(f"Starting {era_label} Ingestion...")
        logger.info("-" * 50)

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
        processed = success + failed

        if sys.stderr.isatty():
            sys.stderr.write(
                f"\r\033[KProcessing {log_prefix}: {filename}... Success: {success} | Failed: {failed} | Total: {total}"
            )
            sys.stderr.flush()
        elif processed % 100 == 0 or processed == total:
            logger.info(
                f"[{log_prefix}] {processed:,}/{total:,} files processed (Success: {success}, Failed: {failed})"
            )

    def _allocate_workers(self) -> int:
        """Determines the optimal number of worker cores to allocate for parallel execution.

        Returns:
            int: The calculated number of CPU cores to assign to the process pool.
        """
        if self.max_workers is not None:
            allocated = max(1, self.max_workers)
            logger.info(
                f"Spawning concurrent environment with user-configured "
                f"cap: {allocated} CPU cores."
            )
            return allocated

        detected_cores = os.cpu_count() or 2
        allocated = min(max(1, detected_cores - 1), 8)
        logger.info(
            f"Spawning concurrent environment utilizing safety "
            f"fallback: {allocated} CPU cores."
        )
        return allocated

    def _process_worker_result(self, future: WorkerFuture, filename: str) -> bool:
        """Evaluates a completed worker process, records telemetry, and returns success state.

        Args:
            future (WorkerFuture): The resolved Pebble process future containing results.
            filename (str): Name of the processed RFC document.

        Returns:
            bool: True if the document was successfully extracted, False otherwise.
        """
        try:
            status, error_msg, telemetry = future.result()

            if (
                status == "success"
                and telemetry
                and telemetry.get("status") == "success"
            ):
                if telemetry.get("total_blocks") == 0:
                    logger.warning(
                        f"File {filename} processed successfully but produced 0 blocks."
                    )
                self._record_telemetry(filename, "success", None, telemetry)
                return True

            self._record_telemetry(filename, "failed", error_msg, None)
            return False

        except TimeoutError:
            timeout_err = f"Execution exceeded per-file timeout limit of {self.per_file_timeout}s."
            logger.warning(
                f"Worker process for {filename} exceeded runtime limits and was terminated."
            )
            self._record_telemetry(filename, "failed", timeout_err, None)
            return False

        except Exception as exc:
            error_msg = f"Task execution error or abrupt process cancellation: {exc}"
            self._record_telemetry(filename, "failed", error_msg, None)
            return False

    def _schedule_next_task(
        self,
        pool: ProcessPool,
        file_iterator: Iterator[Path],
        source_type: Literal["txt", "xml"],
    ) -> tuple[WorkerFuture, Path] | None:
        """Attempts to pull the next file from the iterator and schedule it on the pool.

        Args:
            pool (ProcessPool): The active Pebble process pool.
            file_iterator (Iterator[Path]): The remaining queue of document file paths.
            source_type (Literal["txt", "xml"]): Format type of the source documents.

        Returns:
            tuple[WorkerFuture, Path] | None: A tuple containing the scheduled future
                and its target path, or None if the iterator is exhausted.
        """
        try:
            filepath = next(file_iterator)
            rfc_num = self._extract_rfc_num(filepath)
            future = pool.schedule(  # type: ignore
                _execute_rfc_parsing_worker,
                args=[filepath, rfc_num, source_type, self.output_dir],
                timeout=self.per_file_timeout,
            )
            return future, filepath
        except StopIteration:
            return None

    def _drain_file_iterator(
        self,
        pool: ProcessPool,
        target_files: list[Path],
        source_type: Literal["txt", "xml"],
        log_prefix: str,
        allocated_cores: int,
    ) -> tuple[int, int]:
        """Iterates through target files, managing concurrent task state and updates.

        Args:
            pool (ProcessPool): The active Pebble process pool.
            target_files (list[Path]): The collection of files to process.
            source_type (Literal["txt", "xml"]): Format type of the source documents.
            log_prefix (str): Label prefix for progress tracking outputs.
            allocated_cores (int): Number of cores to calculate pre-fill buffering.

        Returns:
            tuple[int, int]: The final success and failure counts.
        """
        success_count = 0
        failure_count = 0
        total_files = len(target_files)
        file_iterator = iter(target_files)
        future_to_rfc: dict[WorkerFuture, Path] = {}

        for _ in range(allocated_cores * 2):
            task = self._schedule_next_task(pool, file_iterator, source_type)
            if task:
                future_to_rfc[task[0]] = task[1]
            else:
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

                if self._process_worker_result(future, filepath.name):
                    success_count += 1
                else:
                    failure_count += 1

                self._print_progress_ticker(
                    log_prefix, filepath.name, success_count, failure_count, total_files
                )

                next_task = self._schedule_next_task(pool, file_iterator, source_type)
                if next_task:
                    future_to_rfc[next_task[0]] = next_task[1]
                    pending_futures.add(next_task[0])

                del future_to_rfc[future]

        return success_count, failure_count

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
        total_files = len(target_files)
        if total_files == 0:
            logger.warning(
                f"No targeting elements identified for {log_prefix} Ingestion."
            )
            return 0, 0

        allocated_cores = self._allocate_workers()

        global _worker_tree_builder
        if _worker_tree_builder is None:
            logger.info(
                "Pre-loading canonical tree builder metadata index into memory..."
            )
            _worker_tree_builder = CanonicalTreeBuilder(self.metadata_path)

        calculated_max_tasks = max(1, round(0.25 * total_files / allocated_cores))
        success_count, failure_count = 0, 0

        try:
            with ProcessPool(
                max_workers=allocated_cores, max_tasks=calculated_max_tasks
            ) as pool:
                success_count, failure_count = self._drain_file_iterator(
                    pool, target_files, source_type, log_prefix, allocated_cores
                )
        finally:
            if sys.stderr.isatty():
                sys.stderr.write(
                    f"\r\033[K[{log_prefix}] Parallel pool complete. (Total: {total_files:,})\n"
                )
                sys.stderr.flush()

        return success_count, failure_count

    def run_legacy_text_ingestion(self) -> None:
        """Processes plaintext RFC documents that have no XML counterpart in raw_xml_dir.

        Rather than filtering by a hard-coded RFC number ceiling, this pass dynamically
        computes which RFCs are already covered by an XML source file and skips them.
        As XML backports of older RFCs are added to the xml directory over time, they
        are automatically preferred here without any configuration change.
        """
        self._execute_era_ingestion("txt")

    def run_modern_xml_ingestion(self) -> None:
        """Processes modern documents via the structural XML parsing workflow.

        Note: This does not enforce a lower-bound RFC limit. If high-fidelity XML
        backports of legacy RFCs exist in the target directory, they will be parsed
        here and will safely overwrite any heuristic TXT extractions generated
        in previous pipeline steps.
        """
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
                {"file": filename, "status": "failed", "error": clean_error}
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
            # TODO: Populate embedding_model during Phase 3 pipeline integration
            embedding_model=None,
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
