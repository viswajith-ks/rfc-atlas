"""Data schemas for tracking dataset pipeline validation manifests and system telemetry logs."""

from datetime import datetime
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

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


class TelemetryRecord(TypedDict):
    """Dictionary schema specifying execution profile logs for individual source documents."""

    file: str
    status: Literal["success", "failed"]
    total_blocks: NotRequired[int]
    normative_rules: NotRequired[int]
    total_chars: NotRequired[int]
    max_block_chars: NotRequired[int]
    min_block_chars: NotRequired[int]
    error: NotRequired[str]
