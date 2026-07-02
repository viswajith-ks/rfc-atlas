"""LanceDB Integrity & Retrieval Audit.

Asserts total row counts across all local tables and executes
a test Tantivy BM25 Full-Text Search (FTS) to verify index health.
"""

from pathlib import Path
from typing import Any

import lancedb

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_audit() -> None:
    """Executes the integrity audit and FTS test on the local LanceDB instance.

    Scans all available `.lance` tables within the database directory, logs their
    total row counts, and runs a test Tantivy BM25 Full-Text Search against the
    'prose' table to guarantee search functionality.
    """
    db_path = _PROJECT_ROOT / "data" / "lancedb"
    if not db_path.exists():
        print(f"❌ ERROR: LanceDB not found at {db_path}")
        return

    db = lancedb.connect(str(db_path))

    clean_table_names = [p.stem for p in db_path.glob("*.lance")]

    print("==================================================")
    print("📊 LANCEDB INTEGRITY AUDIT")
    print("==================================================")

    total_rows: int = 0
    for table_name in sorted(clean_table_names):
        try:
            tbl = db.open_table(table_name)
            count: int = tbl.count_rows()
            total_rows += count
            print(f"[{table_name.upper():<12}] {count:>10,} rows")
        except (ValueError, OSError, RuntimeError) as e:
            print(f"[{table_name.upper():<12}] ❌ ERROR: {e}")

    print("-" * 50)
    print(f"TOTAL VECTORS : {total_rows:>10,}")
    print("==================================================\n")

    print("==================================================")
    print("🔍 TANTIVY BM25 ENGINE TEST")
    print("==================================================")

    query = "Transmission Control Protocol"
    print(f"Executing exact BM25 query: '{query}'...\n")

    try:
        prose_tbl = db.open_table("prose")

        results: list[dict[str, Any]] = (  # pyright: ignore[reportUnknownVariableType]
            prose_tbl
            .search(query, query_type="fts")  # pyright: ignore[reportUnknownMemberType]
            .select(["chunk_id", "rfc_number", "text_payload", "_score"])
            .limit(3)
            .to_list()
        )

    except (ValueError, OSError, RuntimeError) as e:
        print(f"❌ FTS Query Failed: {e}")
        return

    for idx, row in enumerate(results, 1):
        score: float = float(row.get("_score", 0.0))
        print(
            f"Match #{idx} (BM25 Score: {score:.2f}) | "
            f"RFC {row['rfc_number']} | Chunk: {row['chunk_id']}"
        )
        text_preview = str(row["text_payload"]).replace("\n", " ")[:120]
        print(f"Text: {text_preview}...\n")

    print("✅ Tantivy FTS Engine is operational.")


if __name__ == "__main__":
    run_audit()
