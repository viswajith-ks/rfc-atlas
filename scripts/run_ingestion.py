"""Pipeline Orchestration Engine

Coordinates the full ingestion, extraction, and validation pipeline. Routes raw documents
to format-specific extractors, enriches them with normative metadata, builds validated
canonical tree structures, and logs generation telemetry.
"""

import json
import os
import sys
from datetime import UTC, datetime

from ingestion.manifest import DatasetManifest
from metadata.index_parser import RFCIndexParser
from normalization.normative_extractor import NormativeExtractor
from normalization.tree_builder import CanonicalTreeBuilder
from parsers.txt_parser import LegacyTextParser
from parsers.xml_parser import ModernRFCParser


class PipelineOrchestrator:
    """Unified Orchestrator for the RFC Retrieval-Augmented Generation Pipeline.
    Includes a comprehensive Telemetry Engine to log block statistics.
    """

    def __init__(self, raw_txt_dir, raw_xml_dir, output_dir, manifest_dir):
        """Initializes the orchestrator, provisions local infrastructure, and instantiates
        the document extractors and tree builders.

        Args:
            raw_txt_dir (str): Path containing raw .txt legacy RFC files.
            raw_xml_dir (str): Path containing raw .xml modern RFC files.
            output_dir (str): Destination path for canonical JSON structures.
            manifest_dir (str): Destination path for operational receipts and logs.
        """
        self.raw_txt_dir = raw_txt_dir
        self.raw_xml_dir = raw_xml_dir
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(manifest_dir, exist_ok=True)

        self.dataset_manifest_path = os.path.join(manifest_dir, "dataset_manifest.json")
        self.telemetry_log_path = os.path.join(manifest_dir, "telemetry_log.json")
        self.telemetry_manifest = []

        self.extractor = NormativeExtractor()

        raw_index_path = os.path.join("data", "raw", "rfc_index", "rfc-index.xml")
        metadata_path = os.path.join("metadata", "rfc_metadata_lookup.json")

        os.makedirs("metadata", exist_ok=True)

        if os.path.exists(raw_index_path):
            index_parser = RFCIndexParser(raw_index_path, metadata_path)
            index_parser.parse()
        else:
            print(
                f"Warning: Raw index not found at {raw_index_path}. TreeBuilder may fail."
            )

        self.tree_builder = CanonicalTreeBuilder(metadata_path)

    def _record_telemetry(self, filename, blocks, status="success", error_msg=None):
        """Calculates and logs chunking statistics for the embedding stage."""
        if not blocks and status == "success":
            return

        if status == "success":
            lengths = [
                len(b["normalized_text"]) for b in blocks if b.get("normalized_text")
            ]
            normative_count = sum(1 for b in blocks if b["block_type"] == "normative")

            self.telemetry_manifest.append(
                {
                    "file": filename,
                    "status": status,
                    "total_blocks": len(blocks),
                    "normative_rules": normative_count,
                    "max_block_chars": max(lengths) if lengths else 0,
                    "min_block_chars": min(lengths) if lengths else 0,
                    "avg_block_chars": sum(lengths) // len(lengths) if lengths else 0,
                }
            )
        else:
            self.telemetry_manifest.append(
                {"file": filename, "status": "failed", "error": error_msg}
            )

    def run_legacy_text_ingestion(self):
        """Processes RFC 1 through 8649 via the heuristic plaintext parsing pipeline."""
        print("Starting Legacy Era (RFC 1 - 8649) Text Ingestion...")
        print("-" * 50)

        success_count = 0
        failure_count = 0

        for rfc_num in range(1, 8650):
            filename = f"rfc{rfc_num}.txt"
            filepath = os.path.join(self.raw_txt_dir, filename)

            if not os.path.exists(filepath):
                continue

            try:
                parser = LegacyTextParser(filepath)
                canonical_blocks = parser.parse_document()

                enriched_blocks = self.extractor.process_blocks(canonical_blocks)

                canonical_tree = self.tree_builder.build_tree(
                    rfc_id=str(rfc_num), flat_blocks=enriched_blocks, source_type="txt"
                )

                output_path = os.path.join(
                    self.output_dir, f"rfc{rfc_num}_normalized.json"
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(canonical_tree.model_dump_json(indent=2))

                self._record_telemetry(filename, enriched_blocks, "success")
                success_count += 1

            except Exception as e:
                error_msg = str(e).replace("\n", " | ")
                self._record_telemetry(filename, [], "failed", error_msg)
                failure_count += 1

            sys.stdout.write(
                f"\rProcessing Legacy: {filename}... Success: {success_count} | Failed: {failure_count} "
            )
            sys.stdout.flush()

        print("\n" + "-" * 50)
        print(
            f"--- LEGACY ERA COMPLETE | Success: {success_count} | Failed: {failure_count} ---"
        )

    def run_modern_xml_ingestion(self):
        """Processes RFC 8650+ via the native XML structural parsing pipeline."""
        print("\nStarting Modern Era (RFC 8650+) XML Ingestion...")
        print("-" * 50)

        success_count = 0
        failure_count = 0

        for rfc_num in range(8650, 10000):
            filename = f"rfc{rfc_num}.xml"
            filepath = os.path.join(self.raw_xml_dir, filename)

            if not os.path.exists(filepath):
                continue

            try:
                parser = ModernRFCParser(filepath)
                canonical_blocks = parser.parse_document()

                enriched_blocks = self.extractor.process_blocks(canonical_blocks)

                canonical_tree = self.tree_builder.build_tree(
                    rfc_id=str(rfc_num),
                    flat_blocks=enriched_blocks,
                    source_type="xml",
                )

                output_path = os.path.join(
                    self.output_dir, f"rfc{rfc_num}_normalized.json"
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(canonical_tree.model_dump_json(indent=2))

                self._record_telemetry(filename, enriched_blocks, "success")
                success_count += 1

            except Exception as e:
                error_msg = str(e).replace("\n", " | ")
                self._record_telemetry(filename, [], "failed", error_msg)
                failure_count += 1

            sys.stdout.write(
                f"\rProcessing Modern: {filename}... Success: {success_count} | Failed: {failure_count} "
            )
            sys.stdout.flush()

        print("\n" + "-" * 50)
        print(
            f"--- MODERN ERA COMPLETE | Success: {success_count} | Failed: {failure_count} ---"
        )

    def save_manifest(self):
        """Compiles the telemetry ledger into a strict Pydantic DatasetManifest."""
        print("\nCompiling Final Dataset Manifest...")

        successful = [r for r in self.telemetry_manifest if r["status"] == "success"]
        total_blocks = sum(r.get("total_blocks", 0) for r in successful)
        total_normative = sum(r.get("normative_rules", 0) for r in successful)

        txt_count = sum(1 for r in successful if r["file"].endswith(".txt"))
        xml_count = sum(1 for r in successful if r["file"].endswith(".xml"))

        # Generate a timestamp for the dataset version
        version_stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M")

        # Hydrate the Pydantic Model
        manifest = DatasetManifest(
            dataset_version=version_stamp,
            pipeline_run_at=datetime.now(UTC),
            parser_version="1.0.0",
            chunking_version="1.0.0",
            total_rfcs_indexed=len(successful),
            total_blocks_generated=total_blocks,
            total_normative_statements=total_normative,
            xml_rfcs_processed=xml_count,
            txt_rfcs_processed=txt_count,
        )

        manifest.save_to_disk(self.dataset_manifest_path)
        print(f"Receipt saved securely to {self.dataset_manifest_path}.")

        with open(self.telemetry_log_path, "w", encoding="utf-8") as f:
            json.dump(self.telemetry_manifest, f, indent=2)


if __name__ == "__main__":
    RAW_TXT_DIR = "data/raw/rfcs_txt"
    RAW_XML_DIR = "data/raw/rfcs_xml"
    NORMALIZED_DIR = "data/normalized"
    LOG_DIR = "data/manifests"

    orchestrator = PipelineOrchestrator(
        RAW_TXT_DIR, RAW_XML_DIR, NORMALIZED_DIR, LOG_DIR
    )

    orchestrator.run_legacy_text_ingestion()
    orchestrator.run_modern_xml_ingestion()
    orchestrator.save_manifest()
