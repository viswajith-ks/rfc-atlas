"""Interactive CLI Chat Interface for RFC Atlas.

Provides a real-time, streaming terminal UI to interact with the LLM Synthesis Layer
and the underlying LanceDB vector database.
"""

import argparse
import logging
import sys

from dotenv import load_dotenv

from rfc_atlas.synthesis.orchestrator import SynthesisOrchestrator
from rfc_atlas.utils.exceptions import SynthesisError

load_dotenv()


def _process_query(query: str, orchestrator: SynthesisOrchestrator) -> bool:
    """Processes a single user query and streams the output to the terminal.

    Args:
        query (str): The raw text input from the user.
        orchestrator (SynthesisOrchestrator): The active backend synthesis pipeline.

    Returns:
        bool: True to continue the chat loop, False to trigger a shutdown.
    """
    if not query:
        return True

    if query.lower() in {"quit", "exit"}:
        print("\nShutting down RFC Atlas. Goodbye! 👋\n")
        return False

    print("\n🤖 ATLAS:")

    for text_chunk in orchestrator.stream_query(query, top_k=10):
        print(text_chunk, end="", flush=True)

    print("\n" + "-" * 57)
    return True


def main() -> None:
    """Parses arguments, boots the orchestrator, and runs the interactive chat loop."""
    parser = argparse.ArgumentParser(description="RFC Atlas Interactive CLI Chat.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose backend logging during the chat.",
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
    except ValueError as e:
        print(f"\n❌ BOOT FAILED: {e}")
        print("Please ensure your .env file exists and contains GEMINI_API_KEY.")
        sys.exit(1)
    except (OSError, RuntimeError, ImportError) as e:
        print(f"\n❌ HARDWARE/OS BOOT FAILED: {e}")
        sys.exit(1)

    print("\n✅ System Ready. Type 'quit' or 'exit' to shut down.")
    print("-" * 57)

    while True:
        try:
            query = input("\n👤 YOU: \n> ").strip()

            if not _process_query(query, orchestrator):
                break

        except (KeyboardInterrupt, EOFError):
            print("\n\nShutting down RFC Atlas. Goodbye! 👋\n")
            break
        except SynthesisError as e:
            print(f"\n\n❌ SYNTHESIS ERROR: {e}")
            print("-" * 57)
        except (
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
