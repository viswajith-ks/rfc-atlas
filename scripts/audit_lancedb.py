"""LanceDB Integrity & Retrieval Audit.

Asserts total row counts across all local tables and executes
a test Tantivy BM25 Full-Text Search (FTS) to verify index health.
"""

import logging
from pathlib import Path
from typing import Any

import lancedb

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "lancedb"


def run_audit() -> None:
    """Executes the integrity audit and FTS test on the local LanceDB instance.

    Scans all available `.lance` tables within the database directory, logs their
    total row counts, and runs a test Tantivy BM25 Full-Text Search against the
    'prose' table to guarantee search functionality.
    """
    if not DB_PATH.exists():
        logger.error("❌ ERROR: LanceDB not found at %s", DB_PATH)
        return

    db = lancedb.connect(str(DB_PATH))

    clean_table_names = [p.stem for p in DB_PATH.glob("*.lance")]

    logger.info("==================================================")
    logger.info("📊 LANCEDB INTEGRITY AUDIT")
    logger.info("==================================================")

    total_rows: int = 0
    for table_name in sorted(clean_table_names):
        try:
            tbl = db.open_table(table_name)
            count: int = tbl.count_rows()
            total_rows += count
            logger.info(
                "[%s] %10s rows",
                table_name.upper().ljust(12),
                f"{count:,}",
            )
        except (ValueError, OSError, RuntimeError):
            logger.exception(
                "[%s] ❌ ERROR",
                table_name.upper().ljust(12),
            )

    logger.info("-" * 50)
    logger.info("TOTAL VECTORS : %10s", f"{total_rows:,}")
    logger.info("==================================================\n")

    logger.info("==================================================")
    logger.info("🔍 TANTIVY BM25 ENGINE TEST")
    logger.info("==================================================")

    query = "Transmission Control Protocol"
    logger.info("Executing exact BM25 query: '%s'...\n", query)

    try:
        prose_tbl = db.open_table("prose")

        results: list[dict[str, Any]] = (  # pyright: ignore[reportUnknownVariableType]
            prose_tbl
            .search(query, query_type="fts")  # pyright: ignore[reportUnknownMemberType]
            .select(["chunk_id", "rfc_number", "text_payload", "_score"])
            .limit(3)
            .to_list()
        )

    except (ValueError, OSError, RuntimeError):
        logger.exception("❌ FTS Query Failed")
        return

    for idx, row in enumerate(results, 1):
        score: float = float(row.get("_score", 0.0))
        logger.info(
            "Match #%d (BM25 Score: %.2f) | RFC %s | Chunk: %s",
            idx,
            score,
            row["rfc_number"],
            row["chunk_id"],
        )
        text_preview = str(row["text_payload"]).replace("\n", " ")[:120]
        logger.info("Text: %s...\n", text_preview)

    logger.info("✅ Tantivy FTS Engine is operational.")


if __name__ == "__main__":
    run_audit()
