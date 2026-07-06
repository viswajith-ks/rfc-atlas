"""Master Orchestrator for the Retrieval Engine.

Acts as the unified API boundary for the retrieval pipeline. Coordinates intent
routing, hybrid search, context interception (lineage/errata), semantic reranking,
and final context string assembly.
"""

import logging
from pathlib import Path

from rfc_atlas.retrieval.context_builder import ContextBuilder
from rfc_atlas.retrieval.interceptor import ContextInterceptor
from rfc_atlas.retrieval.query_router import QueryRouter
from rfc_atlas.retrieval.reranker import SemanticReranker
from rfc_atlas.retrieval.search_client import HybridSearchClient

logger = logging.getLogger(__name__)


class RetrievalOrchestrator:
    """Coordinates the end-to-end execution of a context retrieval request."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initializes the underlying search client and reranking engines.

        Args:
            db_path (Path | str | None): Optional path to a specific LanceDB instance.
                If None, the SearchClient uses its default internal routing.
        """
        if db_path:
            self.search_client = HybridSearchClient(db_path=db_path)
        else:
            self.search_client = HybridSearchClient()

        self.reranker = SemanticReranker()

    def retrieve_context(self, query: str, top_k: int = 10) -> str:
        """Executes the full Phase 5 retrieval pipeline for a natural language query.

        1. Routes the query to specific LanceDB tables based on heuristic intents.
        2. Executes a hybrid search to retrieve an over-fetched candidate pool.
        3. Intercepts the pool to inject temporal lineage and errata corrections.
        4. Re-sorts the candidates using an absolute-precision semantic reranker.
        5. Formats the final Top-K chunks into a strict LLM citation block.

        Args:
            query (str): The raw natural language query from the user.
            top_k (int): The final number of context chunks to format for the LLM.

        Returns:
            str: A formatted Markdown string containing the fully enriched context.
        """
        logger.info("🔍 Received Query: '%s'", query)

        target_tables = QueryRouter.route_query(query)
        logger.info("🛤️ Routed query to tables: %s", target_tables)

        fetch_limit = top_k * 5
        raw_candidates = self.search_client.search_multiple(
            query, tables=target_tables, limit=fetch_limit
        )

        if not raw_candidates:
            logger.warning("No candidates found in LanceDB for query.")
            return ContextBuilder.format_context([])
        logger.info("🎯 Retrieved %d raw candidates from LanceDB.", len(raw_candidates))

        enriched_candidates = ContextInterceptor.enrich_results(raw_candidates)
        final_candidates = self.reranker.rerank(query, enriched_candidates, top_k=top_k)

        logger.info("✅ Context assembly complete. Returning Top %d chunks.", top_k)

        return ContextBuilder.format_context(final_candidates)
