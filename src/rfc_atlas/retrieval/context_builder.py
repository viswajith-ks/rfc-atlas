"""Context Assembly Engine.

Formats the final, reranked, and enriched chunk payloads into a strict
Markdown-based context window designed to force LLM grounding and citations.
"""

from rfc_atlas.retrieval.search_client import RetrievalResult


class ContextBuilder:
    """Assembles retrieved chunks into LLM-ready context strings."""

    @staticmethod
    def format_context(results: list[RetrievalResult]) -> str:
        """Compiles a list of retrieval results into a strict citation block.

        Args:
            results (list[RetrievalResult]): The final list of enriched chunks.

        Returns:
            str: A formatted Markdown string ready for prompt injection.
        """
        if not results:
            return "No relevant context found in the RFC Atlas database."

        context_blocks: list[str] = []

        for idx, chunk in enumerate(results, 1):
            header = (
                f"### [SOURCE {idx}: RFC {chunk.rfc_number} | "
                f"Format: {chunk.table_route.upper()} | "
                f"Path: {chunk.hierarchy_path}]"
            )

            safe_payload = chunk.text_payload.replace("### [SOURCE", "--- [SOURCE")
            block = f"{header}\n{safe_payload.strip()}"
            context_blocks.append(block)

        separator = "\n\n" + ("-" * 65) + "\n\n"

        final_output = (
            "The following authoritative context blocks have been retrieved from the "
            "IETF RFC Atlas. Use them to formulate your answer. Always cite your "
            "sources using the provided RFC and Path metadata.\n\n"
        )
        final_output += separator.join(context_blocks)

        return final_output
