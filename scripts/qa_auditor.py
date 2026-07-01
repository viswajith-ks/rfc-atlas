"""Automated Quality Assurance Auditor for Chunking phase.

Validates chunk boundary limits and verifies the conservation of mass
between normalized JSON artifacts and generated LanceDB JSONL chunks.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONNode: TypeAlias = JSONPrimitive | list["JSONNode"] | dict[str, "JSONNode"]

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
NORMALIZED_DIR: Path = _PROJECT_ROOT / "data" / "normalized"
CHUNKS_DIR: Path = _PROJECT_ROOT / "data" / "chunks"
LOGS_DIR: Path = _PROJECT_ROOT / "data" / "logs"

MAX_CHUNK_SIZE: int = 2000
MIN_CHUNK_SIZE: int = 12
MAX_HEALTHY_EXPANSION_RATIO: float = 1.20

DISTRIBUTION_BUCKETS: list[tuple[int, str]] = [
    (10, "<10"),
    (50, "10-50"),
    (100, "50-100"),
    (250, "100-250"),
    (500, "250-500"),
    (1000, "500-1000"),
    (2001, "1000-2000"),
]
OVERFLOW_BUCKET: str = ">2000"


def empty_string_list() -> list[str]:
    """Creates and returns an empty list of strings.

    Returns:
        list[str]: An empty list intended to store string elements.
    """
    return []


@dataclass(frozen=True)
class AuditMetrics:
    """Immutable data transfer object for tracking QA pipeline validation results."""

    total_normalized: int
    total_chunks: int
    missing_rfcs: int
    bloated_rfcs: int
    anomalies: list[str] = field(default_factory=empty_string_list)
    distribution: dict[str, int] = field(default_factory=dict[str, int])


def _scan_single_jsonl(
    jsonl_path: Path,
    chunk_mass: defaultdict[str, int],
    anomalies: list[str],
    distribution: dict[str, int],
) -> int:
    """Scans a single LanceDB JSONL chunk table file.

    Args:
        jsonl_path (Path): Path to the `.jsonl` file.
        chunk_mass (defaultdict[str, int]): Accumulator for character mass.
        anomalies (list[str]): Accumulator for corrupted lines or boundary warnings.
        distribution (dict[str, int]): Accumulator for length distribution buckets.

    Returns:
        int: Number of valid chunks processed in this specific file.
    """
    file_chunks: int = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                raw_chunk: JSONNode = json.loads(line)
            except (ValueError, KeyError) as e:
                anomalies.append(
                    f"[JSON FATAL] Corrupted line in {jsonl_path.name} "
                    f"at line {line_num}: {e}"
                )
            else:
                if not isinstance(raw_chunk, dict):
                    anomalies.append(
                        f"[JSON FATAL] Corrupted line in {jsonl_path.name} "
                        f"at line {line_num}: "
                        f"Expected dict, got {type(raw_chunk).__name__}"
                    )
                    continue

                chunk = raw_chunk
                rfc: str = str(chunk.get("rfc_number", "unknown"))

                payload_node = chunk.get("text_payload", "")
                text: str = str(payload_node) if payload_node else ""
                size: int = len(text)
                file_chunks += 1

                chunk_mass[rfc] += size

                for limit, label in DISTRIBUTION_BUCKETS:
                    if size < limit:
                        distribution[label] += 1
                        break
                else:
                    distribution[OVERFLOW_BUCKET] += 1

                if size > MAX_CHUNK_SIZE:
                    anomalies.append(
                        f"[BOUNDARY FATAL] {rfc} in {jsonl_path.name} "
                        f"line {line_num} has {size} chars!"
                    )
                elif size < MIN_CHUNK_SIZE:
                    anomalies.append(
                        f"[BOUNDARY WARN] {rfc} in {jsonl_path.name} "
                        f"line {line_num} has only {size} chars: {text!r}"
                    )

    return file_chunks


def scan_chunk_tables(
    chunks_dir: Path,
) -> tuple[
    defaultdict[str, int],
    int,
    int,
    list[str],
    dict[str, int],
]:
    """Scans LanceDB JSONL chunk tables to calculate mass and identify boundaries.

    Args:
        chunks_dir (Path): The directory containing the output `.jsonl` files.

    Returns:
        tuple[defaultdict[str, int], int, int, list[str], dict[str, int]]:
            A tuple containing chunk mass dictionary, total chunk count,
            tables scanned, anomalies, and length distribution dictionary.
    """
    chunk_mass: defaultdict[str, int] = defaultdict(int)
    anomalies: list[str] = []
    total_chunks: int = 0
    tables_scanned: int = 0

    distribution: dict[str, int] = {label: 0 for _, label in DISTRIBUTION_BUCKETS}
    distribution[OVERFLOW_BUCKET] = 0

    for jsonl_path in chunks_dir.glob("*.jsonl"):
        tables_scanned += 1
        total_chunks += _scan_single_jsonl(
            jsonl_path, chunk_mass, anomalies, distribution
        )

    return chunk_mass, total_chunks, tables_scanned, anomalies, distribution


def _calculate_blocks_mass(blocks_node: JSONNode) -> int:
    """Safely calculates the total character mass from a JSON list of blocks.

    Args:
        blocks_node (JSONNode): The raw JSON node expected to be a list of blocks.

    Returns:
        int: The accumulated character mass of all valid text payloads.
    """
    mass = 0
    if not isinstance(blocks_node, list):
        return 0

    for block in blocks_node:
        if not isinstance(block, dict):
            continue
        payload = block.get("normalized_text", "")
        if isinstance(payload, str) and payload.strip():
            mass += len(payload)

    return mass


def _scan_single_normalized(json_path: Path, norm_mass: defaultdict[str, int]) -> None:
    """Scans a single normalized JSON artifact to accumulate baseline character mass.

    Args:
        json_path (Path): Path to the normalized `.json` file.
        norm_mass (defaultdict[str, int]): Accumulator for baseline character mass.
    """
    try:
        with json_path.open(encoding="utf-8") as f:
            raw_doc: JSONNode = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    if not isinstance(raw_doc, dict):
        return

    metadata = raw_doc.get("metadata")
    rfc_str: str = "unknown"
    if isinstance(metadata, dict):
        rfc_str = str(metadata.get("rfc_number", "unknown"))

    size_acc: int = _calculate_blocks_mass(raw_doc.get("preface_blocks"))

    sections = raw_doc.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if isinstance(section, dict):
                size_acc += _calculate_blocks_mass(section.get("blocks"))

    norm_mass[rfc_str] = size_acc


def scan_normalized_files(normalized_dir: Path) -> tuple[defaultdict[str, int], int]:
    """Scans normalized JSON artifacts to establish baseline mass.

    Args:
        normalized_dir (Path): Path to the normalized JSON document directory.

    Returns:
        tuple[defaultdict[str, int], int]:
            A tuple containing baseline character mass dict and total document count.
    """
    norm_mass: defaultdict[str, int] = defaultdict(int)
    normalized_files: list[Path] = list(normalized_dir.glob("*.json"))

    for json_path in normalized_files:
        _scan_single_normalized(json_path, norm_mass)

    return norm_mass, len(normalized_files)


def calculate_conservation(
    norm_mass: defaultdict[str, int],
    chunk_mass: defaultdict[str, int],
    anomalies: list[str],
) -> tuple[int, int]:
    """Compares baseline mass to chunk mass to detect data loss or bloat.

    Args:
        norm_mass (defaultdict[str, int]): Source baseline character counts per RFC.
        chunk_mass (defaultdict[str, int]): Downstream chunked character counts.
        anomalies (list[str]): The active anomaly tracking list.

    Returns:
        tuple[int, int]: Missing RFC count and bloated RFC count.
    """
    missing_rfcs: int = 0
    bloated_rfcs: int = 0

    for rfc_id, initial_mass in norm_mass.items():
        if initial_mass == 0:
            continue

        final_mass: int = chunk_mass.get(rfc_id, 0)

        if final_mass == 0:
            anomalies.append(
                f"[MASS FATAL] RFC {rfc_id} vanished completely. 0 chunks found."
            )
            missing_rfcs += 1
            continue

        ratio: float = final_mass / initial_mass

        if ratio < 1.0:
            anomalies.append(
                f"[MASS ERROR] RFC {rfc_id} lost data! "
                f"Normalized: {initial_mass:,} -> Chunks: {final_mass:,} "
                f"(Ratio: {ratio:.2f})"
            )
            missing_rfcs += 1
        elif ratio > MAX_HEALTHY_EXPANSION_RATIO:
            anomalies.append(
                f"[MASS WARN] RFC {rfc_id} bloated. "
                f"Normalized: {initial_mass:,} -> Chunks: {final_mass:,} "
                f"(Ratio: {ratio:.2f})"
            )
            bloated_rfcs += 1

    return missing_rfcs, bloated_rfcs


def write_audit_report(report_file_path: Path, metrics: AuditMetrics) -> None:
    """Serializes the QA audit findings to a text report on disk.

    Args:
        report_file_path (Path): Destination path for the log output.
        metrics (AuditMetrics): Bundled statistics and anomaly tracking data.
    """
    print(f"Writing final report to {report_file_path.name}...")
    with report_file_path.open("w", encoding="utf-8") as report:
        report.write("====================================================\n")
        report.write(" QA AUDIT REPORT\n")
        report.write("====================================================\n\n")

        report.write("----------------------------------------------------\n")
        report.write(" CHUNK LENGTH DISTRIBUTION\n")
        report.write("----------------------------------------------------\n")
        for bucket, count in metrics.distribution.items():
            pct = (count / metrics.total_chunks * 100) if metrics.total_chunks else 0
            report.write(f"   {bucket:<10}: {count:>10,} ({pct:>5.2f}%)\n")
        report.write("----------------------------------------------------\n\n")

        report.write(f"Total Normalized Files Scanned: {metrics.total_normalized:,}\n")
        report.write(f"Total Chunks Scanned: {metrics.total_chunks:,}\n\n")

        if not metrics.anomalies:
            report.write(
                "✅ PERFECT PASS. Zero boundary violations. Zero missing data.\n"
            )
        else:
            report.write(f"❌ FOUND {len(metrics.anomalies)} ANOMALIES.\n")
            report.write(f"   - Data Loss / Timeouts: {metrics.missing_rfcs} files\n")
            report.write(f"   - Overlap Bloat (Loops): {metrics.bloated_rfcs} files\n")
            report.write("\nDetailed Anomaly Log:\n")
            for anomaly in metrics.anomalies:
                report.write(f"  > {anomaly}\n")


def run_audit() -> None:
    """Executes the QA audit pipeline and outputs metrics to a log file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_file_path: Path = LOGS_DIR / "qa_audit_report.txt"

    print("====================================================")
    print(" INITIATING PIPELINE QA AUDIT")
    print(f" Report will be saved to: {report_file_path}")
    print("====================================================\n")

    print("[1/3] Scanning all LanceDB JSONL Tables for Boundaries...")
    (
        chunk_mass,
        total_chunks,
        tables_scanned,
        anomalies,
        distribution,
    ) = scan_chunk_tables(CHUNKS_DIR)
    print(f"      Scanned {total_chunks:,} chunks across {tables_scanned} tables.")

    print("\n[2/3] Scanning Normalized JSON files for Initial Mass...")
    norm_mass, total_normalized = scan_normalized_files(NORMALIZED_DIR)
    print(f"      Scanned {total_normalized:,} normalized files.")

    print("\n[3/3] Executing Conservation of Mass Equations...\n")
    missing_rfcs, bloated_rfcs = calculate_conservation(
        norm_mass, chunk_mass, anomalies
    )

    metrics = AuditMetrics(
        total_normalized=total_normalized,
        total_chunks=total_chunks,
        missing_rfcs=missing_rfcs,
        bloated_rfcs=bloated_rfcs,
        anomalies=anomalies,
        distribution=distribution,
    )

    write_audit_report(report_file_path, metrics)

    print("[SUCCESS] Audit complete. Check the log file for details!")


if __name__ == "__main__":
    run_audit()
