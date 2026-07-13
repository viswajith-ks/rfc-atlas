"""Context Interceptor Middleware.

Hooks into the retrieval pipeline immediately after vector search to dynamically
inject historical lineage warnings and official IETF errata corrections into the
chunk payloads before they reach the LLM.
"""

import logging

from rfc_atlas.graph.lineage import TemporalLineageGraph
from rfc_atlas.retrieval.search_client import RetrievalResult
from rfc_atlas.vector_store.errata_ledger import ErrataLedger

logger = logging.getLogger(__name__)


class ContextInterceptor:
    """Middleware for enriching retrieved chunks with temporal and errata context."""

    @staticmethod
    def _apply_temporal_lineage(chunk: RetrievalResult) -> None:
        """Queries the in-memory graph and prepends lineage warnings if needed.

        Args:
            chunk (RetrievalResult): The mutable retrieved chunk object.
        """
        if "[TEMPORAL" in chunk.text_payload:
            return

        node = TemporalLineageGraph.get_node(chunk.rfc_number)
        if node and node.obsoleted_by:
            chunk.is_obsolete = True

        warning_msg = TemporalLineageGraph.format_lineage_warning(chunk.rfc_number)
        if warning_msg:
            chunk.text_payload = f"{warning_msg}\n\n{chunk.text_payload}"

    @staticmethod
    def _apply_errata_corrections(chunk: RetrievalResult) -> None:
        """Scans the chunk for known IETF typos and appends verified corrections.

        Performs a strict string-match against the errata's "Original Text". If a
        match is found within the chunk's boundaries, the "Corrected Text" is injected.

        Args:
            chunk (RetrievalResult): The mutable retrieved chunk object.
        """
        if chunk.has_errata or "[IETF ERRATA" in chunk.text_payload:
            return

        active_errata = ErrataLedger.get_errata(chunk.rfc_number)
        if not active_errata:
            return

        applied_corrections: list[str] = []
        payload_clean = " ".join(chunk.text_payload.split())

        for erratum in active_errata:
            orig_text = str(erratum.get("orig_text", "")).strip()
            if not orig_text:
                continue

            orig_clean = " ".join(orig_text.split())

            if orig_clean in payload_clean:
                chunk.has_errata = True
                status = str(erratum.get("errata_status_code", "Unknown"))
                correction = str(erratum.get("correct_text", "")).strip()

                injection = (
                    f"⚠️ [IETF ERRATA ({status.upper()})] The text above contains a "
                    f"known error. Replace:\n'{orig_text}'\nWITH:\n'{correction}'"
                )
                applied_corrections.append(injection)

        if applied_corrections:
            correction_block = "\n\n".join(applied_corrections)
            chunk.text_payload = f"{chunk.text_payload}\n\n{correction_block}"

        if applied_corrections:
            correction_block = "\n\n".join(applied_corrections)
            chunk.text_payload = f"{chunk.text_payload}\n\n{correction_block}"

    @classmethod
    def enrich_results(cls, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Intercepts a batch of retrieved chunks and injects relational context.

        Args:
            results (list[RetrievalResult]): The raw results from the LanceDB search.

        Returns:
            list[RetrievalResult]: The enriched results ready for the LLM.
        """
        TemporalLineageGraph.load()
        ErrataLedger.load()

        enriched_count = 0
        obsolete_count = 0

        for chunk in results:
            cls._apply_temporal_lineage(chunk)
            cls._apply_errata_corrections(chunk)

            if chunk.has_errata:
                enriched_count += 1
            if chunk.is_obsolete:
                obsolete_count += 1

        if enriched_count > 0 or obsolete_count > 0:
            logger.info(
                "Context Intercepted: "
                "Flagged %d obsolete chunks and injected %d errata.",
                obsolete_count,
                enriched_count,
            )

        return results
