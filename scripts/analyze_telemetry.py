"""Telemetry Analysis Script

Reads the generated telemetry log and outputs an executive summary of the dataset.
Provides volumetric data on block counts, normative rules, and character constraints
necessary for configuring downstream embedding vector configurations.
"""

import json
import os


def analyze_telemetry():
    """Parses the telemetry ledger and prints a formatted terminal report of dataset statistics."""
    telemetry_path = os.path.join("data", "manifests", "telemetry_log.json")

    if not os.path.exists(telemetry_path):
        print(
            f"Error: {telemetry_path} not found. Make sure you ran the ingestion pipeline first."
        )
        return

    with open(telemetry_path, encoding="utf-8") as f:
        data = json.load(f)

    successful = [d for d in data if d.get("status") == "success"]
    failed = [d for d in data if d.get("status") == "failed"]

    if not successful:
        print("No successful records to analyze.")
        return

    # Calculate global aggregates
    total_files = len(successful)
    total_blocks = sum(r.get("total_blocks", 0) for r in successful)
    total_normative = sum(r.get("normative_rules", 0) for r in successful)

    # Calculate global character statistics
    total_chars = sum(
        r.get("avg_block_chars", 0) * r.get("total_blocks", 0) for r in successful
    )
    global_avg_chars = total_chars // total_blocks if total_blocks > 0 else 0

    # Find extreme length variations to inform chunking limits
    largest_block_file = max(successful, key=lambda x: x.get("max_block_chars", 0))

    valid_mins = [r for r in successful if r.get("min_block_chars", 0) > 0]
    smallest_block_file = min(
        valid_mins, key=lambda x: x.get("min_block_chars", float("inf"))
    )

    # Top 5 most rule-heavy RFCs
    top_normative = sorted(
        successful, key=lambda x: x.get("normative_rules", 0), reverse=True
    )[:5]

    # Print the Executive Summary
    print("=" * 50)
    print(" RFC INGESTION: GLOBAL TELEMETRY REPORT")
    print("=" * 50)
    print(f"Total Documents Processed: {total_files + len(failed):,}")
    print(f"  -> Successful: {total_files:,}")
    print(f"  -> Failed:     {len(failed):,}")
    print("-" * 50)
    print(" VOLUME METRICS:")
    print(f"Total Vector Chunks (Blocks) Generated: {total_blocks:,}")
    print(f"Total Normative Rules (MUST/SHOULD/MAY): {total_normative:,}")
    print(
        f"Average Rules per RFC:                   {total_normative // total_files if total_files else 0:,}"
    )
    print("-" * 50)
    print(" CHUNKING BOUNDARIES (Character Counts):")
    print(f"Global Average Block Size: {global_avg_chars:,} chars")
    print(
        f"Largest Single Block:      {largest_block_file['max_block_chars']:,} chars (Found in {largest_block_file['file']})"
    )
    print(
        f"Smallest Single Block:     {smallest_block_file['min_block_chars']:,} chars (Found in {smallest_block_file['file']})"
    )
    print("-" * 50)
    print(" TOP 5 MOST STRICT PROTOCOLS (By Normative Rules):")
    for i, r in enumerate(top_normative, 1):
        print(f"  {i}. {r['file'].ljust(15)} : {r['normative_rules']:,} rules")
    print("=" * 50)


if __name__ == "__main__":
    analyze_telemetry()
