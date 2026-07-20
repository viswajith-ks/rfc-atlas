"""Master Orchestrator for the RFC Atlas End-to-End Pipeline.

Executes data synchronization, parsing, chunking, vectorization, and auditing
in a single, automated, crash-safe execution sequence.
"""

import argparse
import logging
import shutil
import subprocess  # noqa: S404
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _PROJECT_ROOT / "data"


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    """Executes a shell command and aborts the pipeline on failure."""
    cmd_str = " ".join(cmd)
    logger.info("Running: %s", cmd_str)
    try:
        subprocess.run(cmd, cwd=cwd or _PROJECT_ROOT, check=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        logger.critical(
            "❌ FATAL: Command failed with exit code %d: %s", e.returncode, cmd_str
        )
        sys.exit(1)


def step_0_nuke_and_scaffold(*, from_scratch: bool) -> None:
    """Wipes old artifacts if starting from scratch, and ensures directory structure."""
    if from_scratch:
        logger.info("\n🧹 STEP 0: Wiping all derived and raw data (--from-scratch)...")
        folders_to_nuke = [
            "raw/rfcs_xml",
            "raw/rfcs_txt",
            "raw/rfc_index",
            "normalized",
            "chunks",
            "logs",
            "embeddings",
            "graph",
            "lancedb",
            "metadata",
            "manifests",
        ]
        for folder in folders_to_nuke:
            path = _DATA_DIR / folder
            if path.exists():
                shutil.rmtree(path)
                logger.info("   Deleted: %s", path.relative_to(_PROJECT_ROOT))
    else:
        logger.info(
            "\n⏩ STEP 0: Incremental Mode Active. "
            "Keeping existing database and vectors."
        )

    directories = [
        "raw/rfcs_xml",
        "raw/rfcs_txt",
        "raw/rfc_index",
        "normalized",
        "chunks",
        "logs",
        "metadata",
        "manifests",
        "embeddings",
        "graph",
        "lancedb",
    ]
    for d in directories:
        (_DATA_DIR / d).mkdir(parents=True, exist_ok=True)


def step_1_sync_raw_corpus() -> None:
    """Synchronizes the authoritative IETF files via rsync and curl."""
    logger.info("\n⬇️ STEP 1: Synchronizing authoritative IETF files...")

    xml_dir = _DATA_DIR / "raw" / "rfcs_xml"
    txt_dir = _DATA_DIR / "raw" / "rfcs_txt"
    idx_dir = _DATA_DIR / "raw" / "rfc_index"
    errata_path = _DATA_DIR / "metadata" / "errata.json"

    run_command([
        "rsync",
        "-azm",
        "--info=progress2",
        "--exclude=prerelease/",
        "--exclude=rfc-index.xml",
        "--include=*/",
        "--include=*.xml",
        "--exclude=*",
        "rsync.rfc-editor.org::rfcs/",
        str(xml_dir) + "/",
    ])

    run_command([
        "rsync",
        "-az",
        "--info=progress2",
        "--include=rfc[0-9]*.txt",
        "--exclude=*",
        "rsync.rfc-editor.org::rfcs-text-only/",
        str(txt_dir) + "/",
    ])

    run_command([
        "rsync",
        "-az",
        "--info=progress1",
        "--include=rfc-index.xml",
        "--exclude=*",
        "rsync.rfc-editor.org::rfcs/",
        str(idx_dir) + "/",
    ])

    run_command([
        "curl",
        "-L",
        "-o",
        str(errata_path),
        "https://www.rfc-editor.org/errata.json",
    ])
    logger.info("✅ Raw data synchronization complete.")


def step_2_normalization(*, from_scratch: bool) -> None:
    """Runs the Pipeline Orchestrator to parse and build Canonical Trees."""
    if from_scratch:
        logger.info(
            "\n⚙️ STEP 2: Running the Pipeline Orchestrator (Full XML/TXT parsing)..."
        )
        run_command([sys.executable, "scripts/run_ingestion.py"])
    else:
        logger.info(
            "\n⚙️ STEP 2: Running Incremental Document Ingestion "
            "(Only parsing new/upgraded RFCs)..."
        )
        run_command([sys.executable, "scripts/ingest_incremental_docs.py"])
    logger.info("✅ Corpus normalization complete.")


def step_3_telemetry() -> None:
    """Generates analytical reports on the chunk distributions and system health."""
    logger.info("\n📊 STEP 3: Running Telemetry & Chunk Length Analysis...")
    run_command([sys.executable, "scripts/analyze_chunk_length.py"])
    run_command([sys.executable, "scripts/analyze_telemetry.py"])
    logger.info("✅ Telemetry reports generated.")


def step_4_chunking() -> None:
    """Executes recursive hierarchy-aware splitting and QA audits."""
    logger.info("\n🔪 STEP 4: Executing Recursive JSONL Splitter & QA Checks...")
    run_command([sys.executable, "-m", "rfc_atlas.chunking.recursive_splitter"])
    run_command([sys.executable, "scripts/qa_auditor.py"])
    logger.info("✅ Vector database chunks generated and audited.")


def step_5_vectorization(*, from_scratch: bool) -> None:
    """Generates embeddings and constructs the LanceDB database."""
    logger.info("\n🧠 STEP 5: Resolving Vector Embeddings & Storage...")
    if from_scratch:
        logger.info("[*] Booting Kaggle Cloud-Bridge for Massive GPU Embeddings...")
        run_command([
            sys.executable,
            "scripts/sync_kaggle_vectors.py",
            "--phase",
            "full",
        ])

        logger.info("[*] Compiling LanceDB from Parquet Shards (Tantivy & IVF-PQ)...")
        run_command([sys.executable, "-m", "rfc_atlas.vector_store.lancedb_builder"])
    else:
        logger.info(
            "[*] Autonomous Local Sync: Identifying and embedding deltas on CPU..."
        )
        run_command([sys.executable, "scripts/ingest_incremental.py"])


def step_6_verification() -> None:
    """Audits the final LanceDB structure and Tantivy FTS indices."""
    logger.info("\n🔍 STEP 6: Auditing LanceDB Integrity...")
    run_command([sys.executable, "scripts/audit_lancedb.py"])


def main() -> None:
    """Parses arguments and coordinates the end-to-end pipeline execution."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="RFC Atlas End-to-End Pipeline Orchestrator"
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Wipes all existing data and rebuilds from zero.",
    )
    args = parser.parse_args()

    mode_str = "FROM-SCRATCH" if args.from_scratch else "INCREMENTAL"

    logger.info("====================================================")
    logger.info(" 🚀 INITIATING RFC ATLAS PIPELINE")
    logger.info(" 🛠️  MODE: %s", mode_str)
    logger.info(" Started at: %s", time.ctime())
    logger.info("====================================================")

    step_0_nuke_and_scaffold(from_scratch=args.from_scratch)
    step_1_sync_raw_corpus()
    step_2_normalization(from_scratch=args.from_scratch)
    step_3_telemetry()
    step_4_chunking()
    step_5_vectorization(from_scratch=args.from_scratch)
    step_6_verification()

    logger.info("\n====================================================")
    logger.info(" 🎉 PIPELINE COMPLETION SUCCESSFUL")
    logger.info(" Finished at: %s", time.ctime())
    logger.info("====================================================")


if __name__ == "__main__":
    main()
