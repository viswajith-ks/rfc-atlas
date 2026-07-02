"""Sequential incremental ingestion script for processing new or upgraded RFCs.

Scans the existing normalized JSON artifacts to determine the current dataset state,
identifies missing RFCs or XML format upgrades, and processes them sequentially
without the overhead of the multiprocessing orchestrator.
"""

import argparse
import json
import logging
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from rfc_atlas.ingestion.manifest import DatasetManifest, TelemetryRecord
from rfc_atlas.ingestion.orchestrator import process_document
from rfc_atlas.metadata.index_parser import RFCIndexParser
from rfc_atlas.normalization.tree_builder import CanonicalTreeBuilder
from rfc_atlas.utils import atomic_write

if TYPE_CHECKING:
    from rfc_atlas.normalization.schema import SourceType

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONNode: TypeAlias = JSONPrimitive | list["JSONNode"] | dict[str, "JSONNode"]

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class IncrementalConfig:
    """Configuration container for the incremental document ingestor."""

    raw_txt_dir: Path
    raw_xml_dir: Path
    output_dir: Path
    manifest_dir: Path
    raw_index_path: Path
    metadata_path: Path
    embedding_model: str


def _extract_rfc_num(filename: str) -> int:
    """Extracts the numeric identifier from a filename.

    Args:
        filename (str): The filename string (e.g., 'rfc1234.txt').

    Returns:
        int: The isolated numeric component, or 0 if invalid.
    """
    match = re.match(r"^rfc(\d+)\.", filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _build_existing_state(normalized_dir: Path) -> dict[int, str]:
    """Scans existing JSON artifacts to determine what is already parsed.

    Args:
        normalized_dir (Path): The directory containing the parsed JSON trees.

    Returns:
        dict[int, str]: A mapping of RFC Number -> Source Type ('txt' or 'xml').
    """
    state: dict[int, str] = {}
    if not normalized_dir.exists():
        return state

    for json_file in normalized_dir.glob("*.json"):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data: dict[str, JSONNode] = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning(
                "Failed to read existing artifact state from %s", json_file.name
            )
            continue

        metadata = data.get("metadata")

        if not isinstance(metadata, dict):
            continue

        rfc_num = metadata.get("rfc_number")
        src_type = metadata.get("source_type")
        status = metadata.get("status")
        title = metadata.get("title")

        if isinstance(rfc_num, int) and isinstance(src_type, str):
            if status == "UNKNOWN" and title == f"RFC {rfc_num}":
                logger.info(
                    "Detected placeholder metadata in %s. Forcing reprocessing.",
                    json_file.name,
                )
                continue
            state[rfc_num] = src_type

    return state


def _identify_deltas(
    config: IncrementalConfig, existing_state: dict[int, str]
) -> tuple[list[Path], list[Path], set[int]]:
    """Identifies missing documents and format upgrades based on the existing state.

    Args:
        config (IncrementalConfig): Configuration containing directory paths.
        existing_state (dict[int, str]): Map of existing RFCs to their source types.

    Returns:
        tuple[list[Path], list[Path], set[int]]: A tuple containing the XML queue,
            the TXT queue, and a set of all queued RFC numbers.
    """
    xml_queue: list[Path] = []
    txt_queue: list[Path] = []
    queued_rfc_nums: set[int] = set()

    for xml_file in config.raw_xml_dir.glob("rfc*.xml"):
        rfc_num = _extract_rfc_num(xml_file.name)
        if rfc_num == 0:
            continue

        if rfc_num not in existing_state or existing_state[rfc_num] == "txt":
            xml_queue.append(xml_file)
            queued_rfc_nums.add(rfc_num)

    for txt_file in config.raw_txt_dir.glob("rfc*.txt"):
        rfc_num = _extract_rfc_num(txt_file.name)
        if rfc_num == 0:
            continue

        if rfc_num not in existing_state and rfc_num not in queued_rfc_nums:
            txt_queue.append(txt_file)
            queued_rfc_nums.add(rfc_num)

    return xml_queue, txt_queue, queued_rfc_nums


def _patch_telemetry_log(
    manifest_dir: Path, new_telemetry: list[TelemetryRecord], queued_rfc_nums: set[int]
) -> list[TelemetryRecord]:
    """Appends new telemetry and strips out old records for upgraded RFCs.

    Args:
        manifest_dir (Path): The directory containing the manifest files.
        new_telemetry (list[TelemetryRecord]): The newly generated telemetry records.
        queued_rfc_nums (set[int]): Set of RFC numbers processed in this run.

    Returns:
        list[TelemetryRecord]: The fully patched and sorted telemetry list.
    """
    telemetry_path = manifest_dir / "telemetry_log.json"
    existing_telemetry: list[TelemetryRecord] = []

    if telemetry_path.exists():
        try:
            with telemetry_path.open("r", encoding="utf-8") as f:
                existing_telemetry = json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                "Existing telemetry log is corrupted. It will be overwritten."
            )

    patched_telemetry = [
        t
        for t in existing_telemetry
        if _extract_rfc_num(t["file"]) not in queued_rfc_nums
    ]

    patched_telemetry.extend(new_telemetry)
    patched_telemetry.sort(key=lambda x: _extract_rfc_num(x["file"]))

    with atomic_write(telemetry_path) as f:
        json.dump(patched_telemetry, f, indent=2)
    logger.info("Updated telemetry log saved to %s.", telemetry_path)
    return patched_telemetry


def _rebuild_dataset_manifest(
    patched_telemetry: list[TelemetryRecord],
    config: IncrementalConfig,
) -> None:
    """Aggregates the patched telemetry state into the master deployment contract."""
    successful = [r for r in patched_telemetry if r["status"] == "success"]
    total_blocks = sum(r.get("total_blocks", 0) for r in successful)
    total_normative = sum(r.get("normative_rules", 0) for r in successful)

    txt_count = sum(1 for r in successful if r["file"].endswith(".txt"))
    xml_count = sum(1 for r in successful if r["file"].endswith(".xml"))

    now = datetime.now(UTC)
    manifest = DatasetManifest(
        dataset_version=now.strftime("%Y-%m-%d-%H%M"),
        pipeline_run_at=now,
        parser_version="1.0.0",
        chunking_version="1.0.0",
        embedding_model=config.embedding_model,
        total_rfcs_indexed=len(successful),
        total_blocks_generated=total_blocks,
        total_normative_statements=total_normative,
        xml_rfcs_processed=xml_count,
        txt_rfcs_processed=txt_count,
    )

    manifest_path = config.manifest_dir / "dataset_manifest.json"
    manifest.save_to_disk(manifest_path)
    logger.info("Updated Dataset Manifest saved to %s.", manifest_path)


def run_incremental_ingest(config: IncrementalConfig) -> None:
    """Executes the sequential delta-ingestion pipeline."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.manifest_dir.mkdir(parents=True, exist_ok=True)

    if not config.raw_index_path.exists():
        logger.error("Raw RFC Index file not found at %s", config.raw_index_path)
        sys.exit(1)

    if (
        not config.metadata_path.exists()
        or config.metadata_path.stat().st_mtime < config.raw_index_path.stat().st_mtime
    ):
        logger.info(
            "Metadata lookup cache is missing or stale. Compiling index ledger..."
        )
        index_parser = RFCIndexParser(config.raw_index_path, config.metadata_path)
        index_parser.parse()
    else:
        logger.info("Metadata lookup cache is up-to-date. Skipping XML re-parsing.")

    logger.info("Scanning existing normalized artifacts...")
    existing_state = _build_existing_state(config.output_dir)
    logger.info("Found %d existing normalized documents.", len(existing_state))

    xml_queue, txt_queue, queued_rfc_nums = _identify_deltas(config, existing_state)

    if not xml_queue and not txt_queue:
        logger.info(
            "No missing documents or format upgrades detected. "
            "Pipeline is completely up-to-date!"
        )
        return

    logger.info(
        "Queued %d XML files (New/Upgrades) and %d TXT files (New).",
        len(xml_queue),
        len(txt_queue),
    )

    builder = CanonicalTreeBuilder(config.metadata_path)
    new_telemetry: list[TelemetryRecord] = []

    processing_list: list[tuple[Path, SourceType]] = []
    processing_list.extend((f, "xml") for f in xml_queue)
    processing_list.extend((f, "txt") for f in txt_queue)

    for filepath, src_type in processing_list:
        rfc_num = _extract_rfc_num(filepath.name)
        logger.info("Processing [%s]: %s", src_type, filepath.name)
        try:
            record = process_document(
                filepath, rfc_num, src_type, config.output_dir, builder
            )
            new_telemetry.append(record)
        except Exception:
            logger.exception("Failed to parse %s", filepath.name)
            error_msg = traceback.format_exc().replace("\n", " | ")
            new_telemetry.append({
                "file": filepath.name,
                "status": "failed",
                "error": error_msg,
            })

    patched_telemetry = _patch_telemetry_log(
        config.manifest_dir, new_telemetry, queued_rfc_nums
    )
    _rebuild_dataset_manifest(patched_telemetry, config)


def main() -> None:
    """Parses arguments and executes the incremental document ingestion pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="RFC Atlas Incremental Document Ingestion"
    )
    parser.add_argument(
        "--raw-txt-dir", type=Path, default=_PROJECT_ROOT / "data" / "raw" / "rfcs_txt"
    )
    parser.add_argument(
        "--raw-xml-dir", type=Path, default=_PROJECT_ROOT / "data" / "raw" / "rfcs_xml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_PROJECT_ROOT / "data" / "normalized"
    )
    parser.add_argument(
        "--manifest-dir", type=Path, default=_PROJECT_ROOT / "data" / "manifests"
    )
    parser.add_argument(
        "--raw-index-path",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "rfc_index" / "rfc-index.xml",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=_PROJECT_ROOT / "data" / "metadata" / "rfc_metadata_lookup.json",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="nomic-ai/nomic-embed-text-v1.5"
    )
    args = parser.parse_args()

    try:
        cfg = IncrementalConfig(
            raw_txt_dir=args.raw_txt_dir,
            raw_xml_dir=args.raw_xml_dir,
            output_dir=args.output_dir,
            manifest_dir=args.manifest_dir,
            raw_index_path=args.raw_index_path,
            metadata_path=args.metadata_path,
            embedding_model=args.embedding_model,
        )
        run_incremental_ingest(cfg)
    except Exception:
        logger.exception("CRITICAL FAILURE: Incremental ingestion aborted abnormally.")
        sys.exit(1)


if __name__ == "__main__":
    main()
