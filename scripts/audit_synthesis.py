"""Automated Synthesis Auditor.

Fires real API calls to Gemini and verifies that the generated responses
strictly obey the System Prompt formatting constraints (e.g., Citations).
"""

import argparse
import sys
import time
from pathlib import Path

from rfc_atlas.synthesis.orchestrator import SynthesisOrchestrator
from rfc_atlas.utils.exceptions import SynthesisError

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = _PROJECT_ROOT / "data" / "logs"

TEST_BATTERY = [
    (
        "In to RFC 793, what are the exact states in the TCP connection progression?",
        "Standard Prose Synthesis & Citations",
    ),
    (
        "What is the ABNF syntax for the absolute-URI in HTTP/1.1?",
        "Code/Grammar Block Formatting",
    ),
    (
        "Explain the gravitational pull of the moon.",
        "Grounding/Hallucination Guard (Should short-circuit safely)",
    ),
]


def run_audit() -> int:
    """Executes the synthesis audit and writes a comprehensive report to disk.

    Returns:
        int: The number of queries that failed compliance checks.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = LOGS_DIR / "synthesis_audit_report.txt"

    print("==================================================")
    print(" 🧠 INITIATING SYNTHESIS ENGINE AUDIT")
    print("==================================================")
    print(f"Report will be saved to: {report_file.name}\n")

    orchestrator = SynthesisOrchestrator()
    failed_compliance = 0

    with report_file.open("w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write(" SYNTHESIS ENGINE AUDIT REPORT\n")
        f.write("====================================================\n\n")

        for query, description in TEST_BATTERY:
            print(f"▶ Testing: {description}")
            print(f"  Query: '{query}'")

            start_query = time.time()
            response = orchestrator.query(query, top_k=3)
            elapsed = time.time() - start_query

            # Check compliance with System Prompts
            is_cited = "[Source " in response or "does not contain enough" in response
            if not is_cited:
                print("  [!] COMPLIANCE FAILURE: Response missing required citations.")
                failed_compliance += 1
            else:
                print(f"  [+] Response completed in {elapsed:.2f} seconds.")

            print("-" * 50)

            f.write(f"TEST: {description}\n")
            f.write(f"QUERY: {query}\n")
            f.write(f"LATENCY: {elapsed:.2f} seconds\n")
            f.write(f"COMPLIANT: {is_cited}\n")
            f.write("LLM RESPONSE:\n")
            f.write(f"{response}\n")
            f.write("\n" + "=" * 50 + "\n\n")

    print(f"✅ Audit complete. Review {report_file} for full LLM responses.")
    return failed_compliance


def main() -> None:
    """Parses arguments and executes the synthesis engine audit."""
    parser = argparse.ArgumentParser(description="RFC Atlas Synthesis Auditor.")
    parser.parse_args()

    try:
        failures = run_audit()
        if failures > 0:
            print(
                f"\n❌ AUDIT FAILED: {failures} responses failed compliance checks.",
                file=sys.stderr,
            )
            sys.exit(1)
    except (OSError, ValueError, RuntimeError, ImportError, SynthesisError) as e:
        print(f"CRITICAL FAILURE: Synthesis Audit aborted: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
