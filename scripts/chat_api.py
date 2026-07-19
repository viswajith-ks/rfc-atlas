"""Interactive CLI Chat client for testing the RFC Atlas API boundary.

Mimics the Next.js Vercel AI SDK behavior by sending conversation history
and consuming Server-Sent Events (SSE) streams in real-time over HTTP.
"""

import argparse
import sys
from typing import Any

import httpx


def _stream_response(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    messages: list[dict[str, str]],
) -> None:
    """Handles the HTTP streaming connection and prints the response."""
    print("\n🤖 ATLAS: \n", end="")

    with client.stream("POST", url, json=payload) as response:
        if response.status_code != 200:  # noqa: PLR2004
            err_text = response.read().decode("utf-8")
            print(f"[HTTP {response.status_code}] Error: {err_text}")
            messages.pop()
            print("-" * 57)
            return

        assistant_response = ""
        for text_chunk in response.iter_text():
            print(text_chunk, end="", flush=True)
            assistant_response += text_chunk

    print("\n" + "-" * 57)
    messages.append({"role": "assistant", "content": assistant_response})


def main() -> None:
    """Parses arguments and runs the interactive streaming HTTP client."""
    parser = argparse.ArgumentParser(description="RFC Atlas API Client")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/api/chat",
        help="API endpoint URL.",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of context chunks to retrieve."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="LLM sampling temperature."
    )
    args = parser.parse_args()

    print("=========================================================")
    print(f" 🌐 CONNECTING TO RFC ATLAS API ({args.url})")
    print("=========================================================")
    print("Type 'quit' or 'exit' to shut down.\n")

    messages: list[dict[str, str]] = []

    with httpx.Client(timeout=120.0) as client:
        while True:
            try:
                query = input("👤 YOU: \n> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nDisconnecting. Goodbye! 👋\n")
                break

            if not query:
                continue
            if query.lower() in {"quit", "exit"}:
                print("\nDisconnecting. Goodbye! 👋\n")
                break

            messages.append({"role": "user", "content": query})

            payload: dict[str, Any] = {
                "messages": messages,
                "top_k": args.top_k,
                "temperature": args.temperature,
            }

            try:
                _stream_response(client, args.url, payload, messages)
            except httpx.ConnectError:
                print(f"\n❌ CONNECTION ERROR: Could not reach the API at {args.url}.")
                print("Make sure you are running: python scripts/serve.py")
                sys.exit(1)
            except httpx.HTTPError as e:
                print(f"\n❌ HTTP ERROR: {e}")
                messages.pop()
                print("-" * 57)


if __name__ == "__main__":
    main()
