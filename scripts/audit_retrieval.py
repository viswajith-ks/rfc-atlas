"""Automated Retrieval & Routing Auditor.

Fires a battery of semantic queries designed to trigger specific intent routers,
lineage expansions, and errata injections. Validates that the Retrieval Engine
functions end-to-end and degrades gracefully if the Hugging Face GPU is missing.
"""

import argparse
import sys
import time
from pathlib import Path

from rfc_atlas.retrieval.orchestrator import RetrievalOrchestrator
from rfc_atlas.retrieval.query_router import QueryRouter

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = _PROJECT_ROOT / "data" / "logs"

TEST_BATTERY = [
    (
        "How does the TLS 1.3 handshake work?",
        "Standard Prose Retrieval",
    ),
    (
        "What is the ABNF grammar for the HTTP Host header?",
        "Syntax & Grammar Routing",
    ),
    (
        "Show me the state machine diagram for the TCP protocol.",
        "Visual & Artwork Routing",
    ),
    (
        "What are the security considerations for OAuth 2.0?",
        "Security Routing",
    ),
    (
        "How does RFC 2616 define HTTP caching?",
        "Temporal Lineage Expansion (RFC 2616 is Obsolete)",
    ),
]


def run_audit() -> None:
    """Executes the retrieval audit and writes a comprehensive report to disk."""
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

    with report_file.open("w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write(" RETRIEVAL ENGINE AUDIT REPORT\n")
        f.write("====================================================\n\n")

        for query, description in TEST_BATTERY:
            print(f"▶ Testing: {description}")
            print(f"  Query: '{query}'")

            intents = QueryRouter.classify_intents(query)
            tables = QueryRouter.route_query(query)
            print(f"  Intents Triggered : {', '.join(intents)}")
            print(f"  Routed Tables     : {', '.join(tables)}")

            f.write(f"TEST: {description}\n")
            f.write(f"QUERY: {query}\n")
            f.write(f"INTENTS: {', '.join(intents)}\n")
            f.write(f"TABLES: {', '.join(tables)}\n")

            start_query = time.time()
            context_payload = orchestrator.retrieve_context(query, top_k=2)
            elapsed = time.time() - start_query

            print(f"  [+] Retrieval completed in {elapsed:.2f} seconds.")
            print("-" * 50)

            f.write(f"LATENCY: {elapsed:.2f} seconds\n")
            f.write("PAYLOAD SAMPLE:\n")

            sample = context_payload[:1000] + "\n...[TRUNCATED FOR AUDIT]..."
            f.write(f"{sample}\n")
            f.write("\n" + "=" * 50 + "\n\n")

    print(f"✅ Audit complete. Review {report_file} for formatting and lineage checks.")


def main() -> None:
    """Parses arguments and executes the retrieval engine audit."""
    parser = argparse.ArgumentParser(description="RFC Atlas Retrieval Auditor.")
    _ = parser.parse_args()

    try:
        run_audit()
    except (OSError, ValueError, RuntimeError, ImportError) as e:
        print(
            f"CRITICAL FAILURE: Retrieval Audit aborted abnormally: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
