"""Command-line script for parsing and summarizing pipeline execution telemetry logs."""

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.manifest import TelemetryRecord


def analyze_telemetry(telemetry_log_path: Path) -> None:
    """Parses the telemetry log file and prints an executive summary report to the console.

    Args:
        telemetry_log_path (Path): Path to the JSON telemetry log file.

    Raises:
        FileNotFoundError: If the target log file cannot be found on disk.
    """
    if not telemetry_log_path.exists():
        raise FileNotFoundError(
            f"Error: Target log file '{telemetry_log_path}' could not be discovered. "
            f"Verify that you successfully ran the ingestion pipeline framework first."
        )

    with telemetry_log_path.open(encoding="utf-8") as f:
        data: list[TelemetryRecord] = json.load(f)

    successful = [d for d in data if d.get("status") == "success"]
    failed_count = len(data) - len(successful)

    if not successful:
        print("No successful records to analyze.", file=sys.stderr)
        return

    total_files = len(successful)
    total_blocks = sum(r.get("total_blocks", 0) for r in successful)
    total_normative = sum(r.get("normative_rules", 0) for r in successful)
    total_chars = sum(r.get("total_chars", 0) for r in successful)

    global_avg_chars = total_chars / total_blocks if total_blocks > 0 else 0
    largest_block_file = max(successful, key=lambda x: x.get("max_block_chars", 0))
    valid_mins = [r for r in successful if r.get("min_block_chars", 0) > 0]
    smallest_block_file = None

    if valid_mins:
        smallest_block_file = min(
            valid_mins,
            key=lambda x: x.get("min_block_chars", float("inf")),
        )

    top_normative = sorted(
        successful, key=lambda x: x.get("normative_rules", 0), reverse=True
    )[:5]

    print("=" * 50)
    print(" RFC INGESTION: GLOBAL TELEMETRY REPORT")
    print("=" * 50)
    print(f"Total Documents Processed: {len(data):,}")
    print(f"  -> Successful: {total_files:,}")
    print(f"  -> Failed:     {failed_count:,}")
    print("-" * 50)
    print(" VOLUME METRICS:")
    print(f"Total Vector Chunks (Blocks) Generated: {total_blocks:,}")
    print(f"Total Normative Rules (MUST/SHOULD/MAY): {total_normative:,}")
    print(
        f"Average Rules per RFC:                   {total_normative // total_files if total_files else 0:,}"
    )
    print("-" * 50)
    print(" CHUNKING BOUNDARIES (Character Counts):")
    print(f"Global Average Block Size: {global_avg_chars:,.0f} chars")
    print(
        f"Largest Single Block:      {largest_block_file.get('max_block_chars', 0):,} chars (Found in {largest_block_file['file']})"
    )

    if valid_mins and smallest_block_file is not None:
        print(
            f"Smallest Single Block:     {smallest_block_file.get('min_block_chars', 0):,} chars (Found in {smallest_block_file['file']})"
        )
    else:
        print(
            "Smallest Single Block:     N/A (No non-empty semantic blocks identified)"
        )

    print("-" * 50)
    print(" TOP 5 MOST STRICT PROTOCOLS (By Normative Rules):")

    for i, r in enumerate(top_normative, 1):
        print(f"  {i}. {r['file'].ljust(15)} : {r.get('normative_rules', 0):,} rules")

    print("=" * 50)


def main() -> None:
    """Main entry point for parsing execution arguments and launching telemetry summary logic."""
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Aggregates and formats operational runtime telemetry summaries for compiled RFC corpora."
    )

    parser.add_argument(
        "--telemetry-log-path",
        type=Path,
        default=root_dir / "data" / "manifests" / "telemetry_log.json",
        help="Path to the JSON file where pipeline tracking metrics are recorded.",
    )

    args = parser.parse_args()

    try:
        analyze_telemetry(args.telemetry_log_path)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
