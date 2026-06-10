"""Data schemas for tracking dataset pipeline validation manifests and system telemetry logs."""

from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import BaseModel


class DatasetManifest(BaseModel):
    """Authoritative receipt tracking validation parameters across an ingestion pipeline run."""

    dataset_version: str
    pipeline_run_at: datetime
    parser_version: str
    chunking_version: str
    embedding_model: str | None = None
    total_rfcs_indexed: int
    total_blocks_generated: int
    total_normative_statements: int
    xml_rfcs_processed: int
    txt_rfcs_processed: int

    def save_to_disk(self, filepath: Path) -> None:
        """Serializes the dataset manifest payload directly to a JSON file on disk.

        Args:
            filepath (Path): Target path where the JSON manifest file will be written.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.model_dump_json(indent=2), encoding="utf-8")


class SuccessRecord(TypedDict):
    """Schema tracking a completely parsed and extracted RFC document pipeline run."""

    file: str
    status: Literal["success"]
    total_blocks: int
    normative_rules: int
    total_chars: int
    max_block_chars: int
    min_block_chars: int


class FailureRecord(TypedDict):
    """Schema tracing a catastrophic document parsing or I/O bottleneck failure."""

    file: str
    status: Literal["failed"]
    error: str


TelemetryRecord = SuccessRecord | FailureRecord
