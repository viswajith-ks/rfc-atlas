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
from typing import TypeAlias

from rfc_atlas.ingestion.manifest import DatasetManifest, TelemetryRecord
from rfc_atlas.normalization.normative_extractor import NormativeExtractor
from rfc_atlas.normalization.schema import SourceType
from rfc_atlas.normalization.tree_builder import CanonicalTreeBuilder
from rfc_atlas.parsers.txt_parser import LegacyTextParser
from rfc_atlas.parsers.xml_parser import ModernRFCParser

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONNode: TypeAlias = JSONPrimitive | list["JSONNode"] | dict[str, "JSONNode"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class IncrementalConfig:
    """Configuration container for the incremental document ingestor."""

    raw_txt_dir: Path
    raw_xml_dir: Path
    output_dir: Path
    manifest_dir: Path
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
        if isinstance(rfc_num, int) and isinstance(src_type, str):
            state[rfc_num] = src_type

    return state


def _process_single_document(
    filepath: Path,
    rfc_num: int,
    source_type: SourceType,
    output_dir: Path,
    builder: CanonicalTreeBuilder,
) -> TelemetryRecord:
    """Parses and normalizes a single RFC document.

    Args:
        filepath (Path): Exact file path to the target RFC document.
        rfc_num (int): Numeric identifier of the RFC document.
        source_type (SourceType): Format type of the source document ('txt' or 'xml').
        output_dir (Path): Destination directory for the normalized JSON output.
        builder (CanonicalTreeBuilder): The tree builder instance.

    Returns:
        TelemetryRecord: The final dictionary containing processing metrics.
    """
    if source_type == "txt":
        parser = LegacyTextParser(filepath)
    else:
        parser = ModernRFCParser(filepath)

    canonical_blocks = parser.parse_document()
    extractor = NormativeExtractor()
    enriched_blocks = extractor.process_blocks(canonical_blocks)
    valid_blocks = [b for b in enriched_blocks if b["normalized_text"].strip()]

    canonical_tree = builder.build_tree(
        rfc_number=rfc_num,
        flat_blocks=valid_blocks,
        source_type=source_type,
    )

    output_path = output_dir / f"rfc{rfc_num}_normalized.json"
    canonical_tree.save_to_disk(output_path)

    all_blocks = canonical_tree.preface_blocks + [
        block for section in canonical_tree.sections for block in section.blocks
    ]

    lengths = [len(b.normalized_text) for b in all_blocks]

    return {
        "file": filepath.name,
        "status": "success",
        "total_blocks": len(all_blocks),
        "normative_rules": sum(len(b.normative_statements) for b in all_blocks),
        "total_chars": sum(lengths),
        "max_block_chars": max(lengths, default=0),
        "min_block_chars": min(lengths, default=0),
    }


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

    telemetry_path.write_text(json.dumps(patched_telemetry, indent=2), encoding="utf-8")
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
            record = _process_single_document(
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


if __name__ == "__main__":
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
            metadata_path=args.metadata_path,
            embedding_model=args.embedding_model,
        )
        run_incremental_ingest(cfg)
    except Exception:
        logger.exception("CRITICAL FAILURE: Incremental ingestion aborted abnormally.")
        sys.exit(1)
