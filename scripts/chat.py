"""Interactive CLI Chat Interface for RFC Atlas.

Provides a real-time, streaming terminal UI to interact with the LLM Synthesis Layer
and the underlying LanceDB vector database.
"""

import argparse
import logging
import sys

from dotenv import load_dotenv
from google.genai.errors import APIError

from rfc_atlas.synthesis.orchestrator import SynthesisOrchestrator


def _process_query(query: str, orchestrator: SynthesisOrchestrator, top_k: int) -> bool:
    """Processes a single user query and streams the output to the terminal.

    Args:
        query (str): The raw text input from the user.
        orchestrator (SynthesisOrchestrator): The active backend synthesis pipeline.
        top_k (int): The number of chunks to fetch per query.

    Returns:
        bool: True to continue the chat loop, False to trigger a shutdown.
    """
    if not query:
        return True

    if query.lower() in {"quit", "exit"}:
        print("\nShutting down RFC Atlas. Goodbye! 👋\n")
        return False

    print("\n🤖 ATLAS:")

    for text_chunk in orchestrator.stream_query(query, top_k=top_k):
        print(text_chunk, end="", flush=True)

    print("\n" + "-" * 57)
    return True


def main() -> None:
    """Parses arguments, boots the orchestrator, and runs the interactive chat loop."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="RFC Atlas Interactive CLI Chat.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose backend logging during the chat.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of context chunks to retrieve and inject per query.",
    )
    args = parser.parse_args()

    log_level = logging.INFO if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=========================================================")
    print(" 🌐 BOOTING RFC ATLAS ENGINE...")
    print("=========================================================")

    try:
        orchestrator = SynthesisOrchestrator()
    except (OSError, RuntimeError, ValueError) as e:
        print(f"\n❌ INITIALIZATION FAILED: {e}")
        sys.exit(1)

    print(
        f"\n✅ System Ready (Top-K: {args.top_k}). Type 'quit' or 'exit' to shut down."
    )
    print("-" * 57)

    while True:
        try:
            query = input("\n👤 YOU: \n> ").strip()

            if not _process_query(query, orchestrator, args.top_k):
                break

        except (KeyboardInterrupt, EOFError):
            print("\n\nShutting down RFC Atlas. Goodbye! 👋\n")
            break
        except (
            APIError,
            ConnectionError,
            TimeoutError,
            OSError,
            RuntimeError,
            ValueError,
        ) as e:
            print(f"\n\n❌ RUNTIME ERROR: {e}")
            print("-" * 57)


if __name__ == "__main__":
    main()
