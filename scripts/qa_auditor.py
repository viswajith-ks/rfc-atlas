"""Automated Quality Assurance Auditor for Phase 2 Chunking.

Validates chunk boundary limits and verifies the conservation of mass
between normalized JSON artifacts and generated LanceDB JSONL chunks.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
NORMALIZED_DIR: Path = PROJECT_ROOT / "data" / "normalized"
CHUNKS_DIR: Path = PROJECT_ROOT / "data" / "chunks"
LOGS_DIR: Path = PROJECT_ROOT / "data" / "logs"

MAX_CHUNK_SIZE: int = 2000
MIN_CHUNK_SIZE: int = 2
MAX_HEALTHY_EXPANSION_RATIO: float = 1.20


@dataclass(frozen=True)
class AuditMetrics:
    """Immutable data transfer object for tracking QA pipeline validation results."""

    total_normalized: int
    total_chunks: int
    missing_rfcs: int
    bloated_rfcs: int
    anomalies: list[str] = field(default=[])


def scan_chunk_tables(
    chunks_dir: Path,
) -> tuple[defaultdict[str, int], int, int, list[str]]:
    """Scans LanceDB JSONL chunk tables to calculate total chunk mass and identify boundary violations."""
    chunk_mass: defaultdict[str, int] = defaultdict(int)
    anomalies: list[str] = []
    total_chunks: int = 0
    tables_scanned: int = 0

    for jsonl_path in chunks_dir.glob("*.jsonl"):
        tables_scanned += 1
        with jsonl_path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    chunk: dict[str, Any] = json.loads(line)
                    rfc: str = str(chunk.get("rfc_number", "unknown"))
                    text: str = chunk.get("text_payload", "")
                    size: int = len(text)
                    total_chunks += 1

                    chunk_mass[rfc] += size

                    if size > MAX_CHUNK_SIZE:
                        anomalies.append(
                            f"[BOUNDARY FATAL] {rfc} in {jsonl_path.name} line {line_num} has {size} chars!"
                        )
                    elif size < MIN_CHUNK_SIZE:
                        anomalies.append(
                            f"[BOUNDARY WARN] {rfc} in {jsonl_path.name} line {line_num} has only {size} chars: {text!r}"
                        )

                except Exception as e:
                    anomalies.append(
                        f"[JSON FATAL] Corrupted line in {jsonl_path.name} at line {line_num}: {e}"
                    )

    return chunk_mass, total_chunks, tables_scanned, anomalies


def scan_normalized_files(normalized_dir: Path) -> tuple[defaultdict[str, int], int]:
    """Scans normalized JSON artifacts to establish the baseline conservation of mass."""
    norm_mass: defaultdict[str, int] = defaultdict(int)
    normalized_files: list[Path] = list(normalized_dir.glob("*.json"))

    for json_path in normalized_files:
        with json_path.open(encoding="utf-8") as f:
            doc: dict[str, Any] = json.load(f)
            rfc_str: str = str(doc.get("metadata", {}).get("rfc_number", "unknown"))
            size_acc: int = 0

            for block in doc.get("preface_blocks", []):
                payload: str = block.get("normalized_text", "")
                if payload.strip():
                    size_acc += len(payload)

            for section in doc.get("sections", []):
                for block in section.get("blocks", []):
                    payload = block.get("normalized_text", "")
                    if payload.strip():
                        size_acc += len(payload)

            norm_mass[rfc_str] = size_acc

    return norm_mass, len(normalized_files)


def calculate_conservation(
    norm_mass: defaultdict[str, int],
    chunk_mass: defaultdict[str, int],
    anomalies: list[str],
) -> tuple[int, int]:
    """Compares baseline mass to chunk mass to detect data loss or overlap bloat."""
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
                f"[MASS ERROR] RFC {rfc_id} lost data! Normalized: {initial_mass:,} -> Chunks: {final_mass:,} (Ratio: {ratio:.2f})"
            )
            missing_rfcs += 1
        elif ratio > MAX_HEALTHY_EXPANSION_RATIO:
            anomalies.append(
                f"[MASS WARN] RFC {rfc_id} bloated. Normalized: {initial_mass:,} -> Chunks: {final_mass:,} (Ratio: {ratio:.2f})"
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

        report.write(f"Total Normalized Files Scanned: {metrics.total_normalized:,}\n")
        report.write(f"Total Chunks Scanned: {metrics.total_chunks:,}\n\n")

        if not metrics.anomalies:
            report.write(
                "✅ PERFECT PASS. Zero boundary violations. Zero missing data.\n"
            )
        else:
            report.write(f"❌ FOUND {len(metrics.anomalies)} ANOMALIES.\n")
            report.write(f"   - Data Loss / Timeouts: {metrics.missing_rfcs} files\n")
            report.write(
                f"   - Overlap Bloat (Infinite Loops): {metrics.bloated_rfcs} files\n"
            )
            report.write("\nDetailed Anomaly Log:\n")
            for anomaly in metrics.anomalies:
                report.write(f"  > {anomaly}\n")


def run_audit() -> None:
    """Executes the QA audit pipeline and outputs metrics to a log file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_file_path: Path = LOGS_DIR / "qa_audit_report.txt"

    print("====================================================")
    print(" INITIATING PHASE 2 PIPELINE QA AUDIT")
    print(f" Report will be saved to: {report_file_path}")
    print("====================================================\n")

    print("[1/3] Scanning all LanceDB JSONL Tables for Boundaries...")
    chunk_mass, total_chunks, tables_scanned, anomalies = scan_chunk_tables(CHUNKS_DIR)
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
    )

    write_audit_report(report_file_path, metrics)

    print("[SUCCESS] Audit complete. Check the log file for details!")


if __name__ == "__main__":
    run_audit()
