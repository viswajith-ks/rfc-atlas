"""Dataset Manifest Specification

This module defines the operational receipt for the dataset pipeline.
The generated manifest acts as a strict versioning contract between the
data generation backend (GitHub Actions) and the inference runtime (Hugging Face).

If the inference engine detects a mismatch in parser or chunking versions,
it should refuse to boot rather than serve misaligned vector citations.
"""

import os
from datetime import datetime

from pydantic import BaseModel


class DatasetManifest(BaseModel):
    """The deterministic receipt of a completed ingestion pipeline run."""

    # ---------------------------------------------------------
    # Dataset Identity
    # ---------------------------------------------------------
    dataset_version: str  # Timestamp string, e.g., "2026-06-02-1027"
    pipeline_run_at: datetime  # Exact UTC time the ingestion job finished

    # ---------------------------------------------------------
    # Reproducibility Contracts (Component Versions)
    # ---------------------------------------------------------
    parser_version: str  # Semantic version or Git commit hash of the parser
    chunking_version: str  # Semantic version of the chunking strategy
    embedding_model: str | None = None  # Left None until later stages

    # ---------------------------------------------------------
    # Corpus Statistics (Sanity Checks)
    # ---------------------------------------------------------
    total_rfcs_indexed: int
    total_blocks_generated: int
    total_normative_statements: int

    # ---------------------------------------------------------
    # Hybrid Ingestion Breakdown
    # ---------------------------------------------------------
    xml_rfcs_processed: int  # Count of modern RFCs (8650+) processed via XML
    txt_rfcs_processed: int  # Count of legacy RFCs processed via Regex/Heuristics

    def save_to_disk(self, filepath: str):
        """Serializes the dataset manifest to a JSON receipt.

        Args:
            filepath (str): The destination path where the JSON manifest will be saved.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
