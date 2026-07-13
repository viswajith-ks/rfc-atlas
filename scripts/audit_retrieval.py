"""Automated Retrieval & Routing Auditor.

Fires a battery of semantic queries designed to trigger specific intent routers,
lineage expansions, and errata injections. Validates that the Retrieval Engine
functions end-to-end and degrades gracefully if the Hugging Face GPU is missing.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import TextIO

from rfc_atlas.retrieval.orchestrator import RetrievalOrchestrator
from rfc_atlas.retrieval.query_router import QueryRouter

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = _PROJECT_ROOT / "data" / "logs"

_TRUNCATE_LIMIT = 1000

TEST_BATTERY = [
    (
        "In RFC 793, what are the exact states in the TCP connection progression?",
        "Explicit Metadata Routing (Forces the engine to filter by RFC 793)",
    ),
    (
        "What is the ABNF syntax for the absolute-URI in HTTP/1.1?",
        "Cross-Table Fragmentation Test (Forces stitching prose and ABNF together)",
    ),
    (
        "Explain the SPF (Shortest Path First) calculation for OSPF routing.",
        "Acronym Collision & Dense Semantic Ambiguity",
    ),
    (
        "How does the QUIC crypto handshake differ from TLS 1.3?",
        "Multi-Document Synthesis (Requires retrieving from multiple distinct RFCs)",
    ),
    (
        "What does RFC 8446 say about 0-RTT data security?",
        "Deep Section Retrieval with Metadata Filter",
    ),
]


def _audit_single_query(
    query: str,
    description: str,
    orchestrator: RetrievalOrchestrator,
    file_handle: TextIO,
) -> bool:
    """Executes and logs the retrieval process for a single audit query.

    Args:
        query (str): The natural language query to test.
        description (str): Human-readable intent of the test.
        orchestrator (RetrievalOrchestrator): The active retrieval pipeline.
        file_handle (TextIO): The open log file handle.

    Returns:
        bool: True if the audit query succeeds and returns context, False otherwise.
    """
    intents = QueryRouter.classify_intents(query)
    tables = QueryRouter.route_query(query)
    print(f"  Intents Triggered : {', '.join(intents)}")
    print(f"  Routed Tables     : {', '.join(tables)}")

    file_handle.write(f"TEST: {description}\n")
    file_handle.write(f"QUERY: {query}\n")
    file_handle.write(f"INTENTS: {', '.join(intents)}\n")
    file_handle.write(f"TABLES: {', '.join(tables)}\n")

    start_query = time.time()
    context_payload = orchestrator.retrieve_context(query, top_k=2)
    elapsed = time.time() - start_query

    if "No relevant context found" in context_payload:
        err_msg = "Search returned empty results."
        print(f"  [!] Query Failed: {err_msg}")
        print("-" * 50)
        file_handle.write(f"ERROR: {err_msg}\n")
        file_handle.write("\n" + "=" * 50 + "\n\n")
        return False

    print(f"  [+] Retrieval completed in {elapsed:.2f} seconds.")
    print("-" * 50)

    file_handle.write(f"LATENCY: {elapsed:.2f} seconds\n")
    file_handle.write("PAYLOAD SAMPLE:\n")

    if len(context_payload) > _TRUNCATE_LIMIT:
        sample = context_payload[:_TRUNCATE_LIMIT] + "\n...[TRUNCATED FOR AUDIT]..."
    else:
        sample = context_payload

    file_handle.write(f"{sample}\n")
    file_handle.write("\n" + "=" * 50 + "\n\n")
    return True


def run_audit() -> int:
    """Executes the retrieval audit and writes a comprehensive report to disk.

    Returns:
        int: The number of queries that failed or returned empty results.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = LOGS_DIR / "retrieval_audit_report.txt"

    print("==================================================")
    print(" 🧠 INITIATING RETRIEVAL ENGINE AUDIT")
    print("==================================================")
    print(f"Report will be saved to: {report_file.name}\n")

    print("[*] Booting Retrieval Orchestrator...")
    start_boot = time.time()
    orchestrator = RetrievalOrchestrator()
    print(f"[+] Orchestrator booted in {time.time() - start_boot:.2f} seconds.\n")

    failed_audits = 0

    with report_file.open("w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write(" RETRIEVAL ENGINE AUDIT REPORT\n")
        f.write("====================================================\n\n")

        for query, description in TEST_BATTERY:
            print(f"▶ Testing: {description}")
            print(f"  Query: '{query}'")

            try:
                success = _audit_single_query(query, description, orchestrator, f)
                if not success:
                    failed_audits += 1
            except (
                ValueError,
                TypeError,
                RuntimeError,
                OSError,
                KeyError,
                ImportError,
            ) as e:
                failed_audits += 1
                err_msg = f"  [!] Query Failed: {e}"
                print(err_msg)
                print("-" * 50)
                f.write(f"ERROR: {e}\n")
                f.write("\n" + "=" * 50 + "\n\n")

    print(f"✅ Audit complete. Review {report_file} for formatting and lineage checks.")
    return failed_audits


def main() -> None:
    """Parses arguments and executes the retrieval engine audit."""
    parser = argparse.ArgumentParser(description="RFC Atlas Retrieval Auditor.")
    parser.parse_args()

    try:
        failures = run_audit()
        if failures > 0:
            print(
                f"\n❌ AUDIT FAILED: {failures} queries failed to execute.",
                file=sys.stderr,
            )
            sys.exit(1)
    except (OSError, ValueError, RuntimeError, ImportError) as e:
        print(
            f"CRITICAL FAILURE: Retrieval Audit aborted abnormally: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
