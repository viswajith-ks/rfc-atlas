"""Automated Quality Assurance Auditor for Phase 2 Chunking.

Validates chunk boundary limits and verifies the conservation of mass
between normalized JSON artifacts and generated LanceDB JSONL chunks.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
NORMALIZED_DIR: Path = PROJECT_ROOT / "data" / "normalized"
CHUNKS_DIR: Path = PROJECT_ROOT / "data" / "chunks"
LOGS_DIR: Path = PROJECT_ROOT / "data" / "logs"

MAX_CHUNK_SIZE: int = 2000
MIN_CHUNK_SIZE: int = 2
MAX_HEALTHY_EXPANSION_RATIO: float = 1.50


def run_audit() -> None:
    """Executes the QA audit pipeline, compiling mass conservation metrics and
    identifying anomalous data expansion or data loss, outputting to a log file.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_file_path: Path = LOGS_DIR / "qa_audit_report.txt"

    print("====================================================")
    print(" INITIATING PHASE 2 PIPELINE QA AUDIT")
    print(f" Report will be saved to: {report_file_path}")
    print("====================================================\n")

    chunk_mass: defaultdict[str, int] = defaultdict(int)
    norm_mass: defaultdict[str, int] = defaultdict(int)
    anomalies: list[str] = []

    total_chunks: int = 0
    tables_scanned: int = 0

    print("[1/3] Scanning all LanceDB JSONL Tables for Boundaries...")
    for jsonl_path in CHUNKS_DIR.glob("*.jsonl"):
        tables_scanned += 1
        with open(jsonl_path, encoding="utf-8") as f:
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

    print(f"      Scanned {total_chunks:,} chunks across {tables_scanned} tables.")
    print("\n[2/3] Scanning Normalized JSON files for Initial Mass...")

    normalized_files: list[Path] = list(NORMALIZED_DIR.glob("*.json"))
    for json_path in normalized_files:
        with open(json_path, encoding="utf-8") as f:
            doc: dict[str, Any] = json.load(f)
            rfc_str: str = str(doc.get("metadata", {}).get("rfc_number", "unknown"))
            size_acc: int = 0

            for section in doc.get("sections", []):
                for block in section.get("blocks", []):
                    payload: str = block.get("normalized_text", "")
                    if payload.strip():
                        size_acc += len(payload)

            norm_mass[rfc_str] = size_acc

    print(f"      Scanned {len(normalized_files):,} normalized files.")
    print("\n[3/3] Executing Conservation of Mass Equations...\n")

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

    print(f"Writing final report to {report_file_path.name}...")
    with open(report_file_path, "w", encoding="utf-8") as report:
        report.write("====================================================\n")
        report.write(" QA AUDIT REPORT\n")
        report.write("====================================================\n\n")

        report.write(f"Total Normalized Files Scanned: {len(normalized_files):,}\n")
        report.write(f"Total Chunks Scanned: {total_chunks:,}\n\n")

        if not anomalies:
            report.write(
                "✅ PERFECT PASS. Zero boundary violations. Zero missing data.\n"
            )
        else:
            report.write(f"❌ FOUND {len(anomalies)} ANOMALIES.\n")
            report.write(f"   - Data Loss / Timeouts: {missing_rfcs} files\n")
            report.write(f"   - Overlap Bloat (Infinite Loops): {bloated_rfcs} files\n")
            report.write("\nDetailed Anomaly Log:\n")
            for anomaly in anomalies:
                report.write(f"  > {anomaly}\n")

    print("[SUCCESS] Audit complete. Check the log file for details!")


if __name__ == "__main__":
    run_audit()
