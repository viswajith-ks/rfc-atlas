"""Telemetry script for analyzing block length distributions in normalized RFC artifacts."""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import TypeAlias, TypedDict

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONNode: TypeAlias = JSONPrimitive | list["JSONNode"] | dict[str, "JSONNode"]


class BlockLengthRecord(TypedDict):
    """Strict schema for telemetry records to avoid dictionary covariance errors."""

    rfc_id: str
    block_type: str
    length: int


DATA_DIR: Path = Path("data")
NORMALIZED_DIR: Path = DATA_DIR / "normalized"
LOGS_DIR: Path = DATA_DIR / "logs"

OUTLIERS_CSV: Path = LOGS_DIR / "outliers_blocks.csv"
STANDARD_CSV: Path = LOGS_DIR / "standard_blocks.csv"

CHUNK_THRESHOLD: int = 2000


def extract_blocks_recursive(node: JSONNode, rfc_id: str) -> list[BlockLengthRecord]:
    """Recursively traverses the canonical JSON tree to find and measure text blocks.

    Args:
        node: The current JSON node (dict, list, or primitive) being inspected.
        rfc_id: The identifier of the RFC being processed.

    Returns:
        A strictly typed list of telemetry records containing block metadata and text lengths.
    """
    blocks: list[BlockLengthRecord] = []

    if isinstance(node, list):
        for item in node:
            blocks.extend(extract_blocks_recursive(item, rfc_id))

    elif isinstance(node, dict):
        if "block_type" in node and "normalized_text" in node:
            text_payload: str = str(node.get("normalized_text") or "")
            blocks.append(
                {
                    "rfc_id": rfc_id,
                    "block_type": str(node["block_type"]),
                    "length": len(text_payload),
                }
            )

        for value in node.values():
            if isinstance(value, (dict, list)):
                blocks.extend(extract_blocks_recursive(value, rfc_id))

    return blocks


def main() -> None:
    """Executes the telemetry analysis and generates distribution reports."""
    if not NORMALIZED_DIR.exists():
        print(f"Error: Directory {NORMALIZED_DIR} not found.")
        sys.exit(1)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Scanning canonical artifacts in {NORMALIZED_DIR}...")

    all_blocks: list[BlockLengthRecord] = []

    json_files: list[Path] = list(NORMALIZED_DIR.glob("*.json"))
    total_files: int = len(json_files)

    for idx, filepath in enumerate(json_files, 1):
        try:
            with Path.open(filepath, encoding="utf-8") as f:
                raw_data: JSONNode = json.load(f)
                data = raw_data

                if isinstance(data, dict):
                    metadata_node = data.get("metadata")

                    if isinstance(metadata_node, dict):
                        rfc_num_node = metadata_node.get("rfc_number", filepath.stem)
                        rfc_number = str(rfc_num_node)
                    else:
                        rfc_number = str(filepath.stem)

                    all_blocks.extend(extract_blocks_recursive(data, rfc_number))

        except Exception as e:
            print(f"Failed to parse {filepath.name}: {e}")

        if sys.stderr.isatty() and idx % 100 == 0:
            sys.stderr.write(f"\r\033[K[Scanning] Files processed: {idx}/{total_files}")
            sys.stderr.flush()

    if sys.stderr.isatty():
        sys.stderr.write(
            f"\r\033[K[Scanning] Complete. Total files scanned: {total_files:,}\n"
        )
        sys.stderr.flush()

    if not all_blocks:
        print("No blocks found. Check your directory path and JSON structure.")
        sys.exit(1)

    lengths_by_type: dict[str, list[int]] = defaultdict(list)
    outliers: list[BlockLengthRecord] = []
    standards: list[BlockLengthRecord] = []

    for block in all_blocks:
        lengths_by_type[block["block_type"]].append(block["length"])
        if block["length"] > CHUNK_THRESHOLD:
            outliers.append(block)
        else:
            standards.append(block)

    outliers.sort(key=lambda x: x["length"], reverse=True)
    standards.sort(key=lambda x: x["length"], reverse=True)

    with Path.open(OUTLIERS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rfc_id", "block_type", "length"])
        for b in outliers:
            writer.writerow([b["rfc_id"], b["block_type"], b["length"]])

    with Path.open(STANDARD_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rfc_id", "block_type", "length"])
        for b in standards:
            writer.writerow([b["rfc_id"], b["block_type"], b["length"]])

    total_blocks: int = len(all_blocks)
    over_threshold: int = len(outliers)

    print("\n" + "=" * 50)
    print(" 📊 PHASE 2: BLOCK LENGTH TELEMETRY REPORT")
    print("=" * 50)
    print(f"Total Blocks Scanned : {total_blocks:,}")
    print(f"Target Chunk Limit   : {CHUNK_THRESHOLD:,} characters")
    print(
        f"Blocks > Limit       : {over_threshold:,} ({(over_threshold / total_blocks) * 100:.2f}% of total)\n"
    )

    print(
        f"{'Block Type':<15} | {'Count':<8} | {'p50 (Med)':<9} | {'p95':<7} | {'Max Length':<10}"
    )
    print("-" * 60)

    for b_type, lengths in sorted(lengths_by_type.items()):
        lengths.sort()
        count: int = len(lengths)
        p50: int = int(statistics.median(lengths)) if count > 0 else 0
        p95: int = int(statistics.quantiles(lengths, n=100)[94]) if count > 0 else 0
        max_len: int = max(lengths) if count > 0 else 0

        print(f"{b_type:<15} | {count:<8,} | {p50:<9,} | {p95:<7,} | {max_len:<10,}")

    print("\n" + "=" * 50)
    print(f"✅ Outliers saved to: {OUTLIERS_CSV}")
    print(f"✅ Standard blocks saved to: {STANDARD_CSV}")


if __name__ == "__main__":
    main()
