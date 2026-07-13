"""LanceDB Hybrid Search Client.

Connects to the local LanceDB instance, dynamically encodes queries using the
Nomic embedding model, and executes high-performance hybrid (Dense + Sparse)
retrieval across specified table routes.
"""

import logging
import re
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from rfc_atlas.chunking.schema import LanceTableRoute
from rfc_atlas.vector_store.schema import EPSILON, VECTOR_DIMENSIONS

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_DIR = _PROJECT_ROOT / "data" / "lancedb"


class RetrievalResult(BaseModel):
    """Strict schema for an individual chunk retrieved from LanceDB."""

    chunk_id: str
    rfc_number: int
    table_route: str
    hierarchy_path: str
    text_payload: str
    score: float = Field(
        description="The hybrid relevance score (Dense + Sparse combination)."
    )
    hybrid_score: float | None = Field(
        default=None,
        description="Preserved LanceDB score if overwritten by cross-encoder rerank.",
    )

    has_errata: bool = False
    is_obsolete: bool = False


class HybridSearchClient:
    """Manages local LanceDB connections and executes hybrid semantic queries."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_DIR) -> None:
        """Initializes the LanceDB connection and prepares the embedding lazy-loader.

        Args:
            db_path (Path | str): Path to the local LanceDB directory.
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            logger.warning(
                "LanceDB directory not found at %s. Searches will fail.", self.db_path
            )

        self._db = lancedb.connect(str(self.db_path))
        self._model: SentenceTransformer | None = None

    def _ensure_model(self) -> SentenceTransformer:
        """Lazily loads the Nomic embedding model into memory.

        Returns:
            SentenceTransformer: The active model instance.
        """
        if self._model is None:
            logger.info("📥 Loading Nomic embedding model for query encoding...")
            self._model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device="cpu"
            )
        return self._model

    def _encode_query(self, query_text: str) -> list[float]:
        """Encodes and mathematically L2-normalizes the search query.

        Args:
            query_text (str): The raw string query from the user/router.

        Returns:
            list[float]: A 256-dimensional L2-normalized float array.
        """
        model = self._ensure_model()

        nomic_query = f"search_query: {query_text}"

        raw_vector: np.ndarray = model.encode(  # pyright: ignore[reportUnknownMemberType]
            [nomic_query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        sliced = raw_vector[:VECTOR_DIMENSIONS]
        norm = np.linalg.norm(sliced)  # pyright: ignore[reportUnknownMemberType]
        norm = max(norm, EPSILON)

        return (sliced / norm).tolist()

    def search_table(
        self,
        query: str,
        query_vector: list[float],
        table_name: LanceTableRoute,
        limit: int = 15,
    ) -> list[RetrievalResult]:
        """Executes a hybrid search against a single LanceDB table.

        Args:
            query (str): The raw user query.
            query_vector (list[float]): The encoded L2-normalized float array.
            table_name (LanceTableRoute): The target LanceDB table (e.g., 'prose').
            limit (int): Maximum number of chunks to retrieve.

        Returns:
            list[RetrievalResult]: A list of strictly validated retrieved chunks.
        """
        if table_name not in self._db.list_tables().tables:
            logger.warning("Table '%s' does not exist in LanceDB.", table_name)
            return []

        table = self._db.open_table(table_name)
        if table.count_rows() == 0:
            return []

        rfc_match = re.search(r"(?i)\brfc\s*(\d+)\b", query)

        try:
            search_op = table.search(query_type="hybrid")  # pyright: ignore[reportUnknownMemberType]

            if rfc_match:
                search_op = search_op.where(f"rfc_number = {int(rfc_match.group(1))}")

            raw_results: list[dict[str, Any]] = (  # pyright: ignore[reportUnknownVariableType]
                search_op.vector(query_vector).text(query).limit(limit).to_list()  # pyright: ignore[reportUnknownMemberType]
            )
        except (ValueError, TypeError, RuntimeError, OSError):
            logger.exception("Hybrid search failed on table '%s'", table_name)
            return []

        parsed_results: list[RetrievalResult] = []
        for row in raw_results:
            try:
                score = float(row.get("_score", 0.0))
                parsed_results.append(
                    RetrievalResult(
                        chunk_id=str(row["chunk_id"]),
                        rfc_number=int(row["rfc_number"]),
                        table_route=str(row.get("table_route", table_name)),
                        hierarchy_path=str(row.get("hierarchy_path", "Unknown")),
                        text_payload=str(row["text_payload"]),
                        score=score,
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning("Dropping malformed retrieval row: %s", e)

        return parsed_results

    def search_multiple(
        self, query: str, tables: list[LanceTableRoute], limit: int = 20
    ) -> list[RetrievalResult]:
        """Executes hybrid searches across multiple tables and merges the results.

        Note: LanceDB hybrid scores (BM25 + Dense) are mathematically incommensurable
        across different tables. Therefore, this method returns the full combined pool
        (limit * N tables) WITHOUT TRUNCATION, passing the responsibility of global
        absolute-precision sorting to the SemanticReranker.

        Args:
            query (str): The raw user query.
            tables (list[LanceTableRoute]): A list of tables to target.
            limit (int): The maximum number of chunks to fetch PER TABLE.

        Returns:
            list[RetrievalResult]: The combined un-truncated candidate pool.
        """
        combined_results: list[RetrievalResult] = []
        target_tables = sorted(set(tables))

        try:
            query_vector = self._encode_query(query)
        except (ValueError, RuntimeError, OSError, TypeError) as e:
            logger.warning("Query encoding failed: %s. Aborting search.", e)
            return []

        for table_name in target_tables:
            results = self.search_table(query, query_vector, table_name, limit=limit)
            combined_results.extend(results)

        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results

    def stitch_neighbors(self, chunks: list[RetrievalResult]) -> list[RetrievalResult]:
        """Expands context by fetching adjacent blocks (blk-1, blk+1) across tables.

        Args:
            chunks (list[RetrievalResult]): The reranked candidate chunks.

        Returns:
            list[RetrievalResult]: The candidates with newly stitched text payloads.
        """
        if not chunks:
            return []

        all_tables = self._db.list_tables().tables

        for chunk in chunks:
            match = re.match(
                r"^(rfc\d+-sec[a-zA-Z0-9_.-]+)-blk(\d+)-chunk", chunk.chunk_id
            )
            if not match:
                continue

            prefix = match.group(1)
            blk_idx = int(match.group(2))

            prev_prefix = f"{prefix}-blk{blk_idx - 1}-"
            next_prefix = f"{prefix}-blk{blk_idx + 1}-"

            reconstructed: dict[str, str] = {chunk.chunk_id: chunk.text_payload}

            for t_name in all_tables:
                try:
                    tbl = self._db.open_table(t_name)
                    query_str = (
                        f"chunk_id LIKE '{prev_prefix}%' "
                        f"OR chunk_id LIKE '{next_prefix}%'"
                    )
                    raw_res: list[dict[str, Any]] = (  # pyright: ignore[reportUnknownVariableType]
                        tbl
                        .search()  # pyright: ignore[reportUnknownMemberType]
                        .where(query_str)
                        .limit(20)
                        .to_list()
                    )
                    for r in raw_res:
                        reconstructed[str(r["chunk_id"])] = str(r["text_payload"])
                except (ValueError, OSError, RuntimeError) as e:
                    logger.debug("Failed to stitch adjacent block in %s: %s", t_name, e)

            sorted_texts = [reconstructed[k] for k in sorted(reconstructed.keys())]
            chunk.text_payload = "\n\n".join(sorted_texts)

        return chunks
