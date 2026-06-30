"""CLI execution script for running the RFC dataset ingestion pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from rfc_atlas.ingestion.orchestrator import PipelineConfig, PipelineOrchestrator

logger = logging.getLogger(__name__)


def main() -> None:
    """Coordinates parallel execution of the RFC ingestion pipeline via CLI args."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    base_dir = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="RFC Intelligent System Ingestion Pipeline Tooling Interface."
    )

    parser.add_argument(
        "--raw-txt-dir",
        type=Path,
        default=base_dir / "data" / "raw" / "rfcs_txt",
        help="Path to directory containing raw legacy text RFC files.",
    )
    parser.add_argument(
        "--raw-xml-dir",
        type=Path,
        default=base_dir / "data" / "raw" / "rfcs_xml",
        help="Path to directory containing raw modern XML RFC files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "data" / "normalized",
        help="Destination directory for serialized canonical JSON trees.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=base_dir / "data" / "manifests",
        help="Destination directory for operational manifest tracking logs.",
    )
    parser.add_argument(
        "--raw-index-path",
        type=Path,
        default=base_dir / "data" / "raw" / "rfc_index" / "rfc-index.xml",
        help="Path to the foundational global raw rfc-index.xml file.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=base_dir / "data" / "metadata" / "rfc_metadata_lookup.json",
        help="Path to the compiled metadata lookup cache file.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Explicit ceiling cap for process pool worker process constraints.",
    )
    parser.add_argument(
        "--per-file-timeout",
        type=float,
        default=45.0,
        help="Ceiling execution limit duration granted to process an individual file.",
    )

    args = parser.parse_args()

    try:
        config = PipelineConfig(
            raw_txt_dir=args.raw_txt_dir,
            raw_xml_dir=args.raw_xml_dir,
            output_dir=args.output_dir,
            manifest_dir=args.manifest_dir,
            raw_index_path=args.raw_index_path,
            metadata_path=args.metadata_path,
            max_workers=args.max_workers,
            per_file_timeout=args.per_file_timeout,
        )

        orchestrator = PipelineOrchestrator.create_and_initialize(config)
        orchestrator.run_legacy_text_ingestion()
        orchestrator.run_modern_xml_ingestion()
        orchestrator.save_manifest()

    except Exception:
        logger.exception(
            "CRITICAL FAILURE: Ingestion pipeline processing aborted abnormally."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
