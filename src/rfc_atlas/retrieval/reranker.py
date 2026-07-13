"""Semantic Reranking Engine.

Applies absolute cross-encoder semantic precision to the candidate pool retrieved
by LanceDB. Designed to execute on Hugging Face ZeroGPU infrastructure with
strict graceful degradation (fallback to hybrid scores) if compute is unavailable.
"""

import logging
from typing import Any

from rfc_atlas.retrieval.search_client import RetrievalResult

logger = logging.getLogger(__name__)

_ml_libs_loaded = False
try:
    import torch
    import transformers

    _ml_libs_loaded = True
except ImportError:
    pass

HAS_ML_LIBS: bool = _ml_libs_loaded


class SemanticReranker:
    """Re-scores and re-sorts document chunks using a high-precision cross-encoder."""

    _MODEL_ID = "BAAI/bge-reranker-v2-m3"
    _BATCH_SIZE = 16
    _MAX_LENGTH = 1024

    def __init__(self) -> None:
        """Initializes the reranker state without immediately loading weights."""
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"
        self._is_loaded: bool = False

    def _ensure_loaded(self) -> None:
        """Lazily loads the model into the optimal available hardware backend.

        Raises:
            ImportError: If torch or transformers are missing from the environment.
        """
        if self._is_loaded:
            return

        if not HAS_ML_LIBS:
            e = "transformers and torch are required for the SemanticReranker."
            raise ImportError(e)

        logger.info("📥 Loading Reranker model (%s)...", self._MODEL_ID)

        self._device = (
            "cuda"
            if torch.cuda.is_available()  # pyright: ignore[reportPossiblyUnboundVariable]
            else "cpu"
        )
        dtype = (
            torch.float16  # pyright: ignore[reportPossiblyUnboundVariable]
            if self._device == "cuda"
            else torch.float32  # pyright: ignore[reportPossiblyUnboundVariable]
        )

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(  # pyright: ignore[reportPossiblyUnboundVariable, reportUnknownMemberType]
            self._MODEL_ID
        )
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(  # pyright: ignore[reportPossiblyUnboundVariable, reportUnknownMemberType]
            self._MODEL_ID, torch_dtype=dtype
        ).to(self._device)

        self._model.eval()  # pyright: ignore[reportUnknownMemberType]
        self._is_loaded = True
        logger.info("⚡ Reranker activated on %s.", self._device.upper())

    def _compute_scores(self, pairs: list[list[str]]) -> list[float]:
        """Executes the PyTorch forward pass to score the pairs.

        Args:
            pairs (list[list[str]]): List of [query, chunk_text] string pairs.

        Returns:
            list[float]: A list of relevance scores corresponding to the pairs.
        """
        all_scores: list[float] = []

        with torch.no_grad():  # pyright: ignore[reportPossiblyUnboundVariable]
            for i in range(0, len(pairs), self._BATCH_SIZE):
                batch_pairs = pairs[i : i + self._BATCH_SIZE]

                inputs = self._tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self._MAX_LENGTH,
                    return_tensors="pt",
                ).to(self._device)

                logits = self._model(**inputs, return_dict=True).logits.view(-1)
                scores = logits.float().cpu().tolist()
                all_scores.extend(scores)

        return all_scores

    def rerank(
        self, query: str, candidates: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]:
        """Scores candidate chunks against the query using cross-encoder attention.

        If the hardware is unavailable or an Out-Of-Memory error occurs, the system
        will catch the exception, log a warning, and gracefully degrade by returning
        the original top_k candidates based on their pre-existing LanceDB hybrid scores.

        Args:
            query (str): The raw user query.
            candidates (list[RetrievalResult]): The unrefined chunks from LanceDB.
            top_k (int): The final number of context chunks to return to the LLM.

        Returns:
            list[RetrievalResult]: The semantically re-sorted candidate list.
        """
        if not candidates:
            return []

        try:
            self._ensure_loaded()
        except (ImportError, RuntimeError, ValueError, OSError) as e:
            logger.warning(
                "⚠️ DEGRADED MODE: Reranker failed to initialize (%s). "
                "Falling back to base hybrid retrieval scores.",
                e,
            )
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates[:top_k]

        pairs = [[query, chunk.text_payload] for chunk in candidates]

        try:
            all_scores = self._compute_scores(pairs)
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning(
                "⚠️ DEGRADED MODE: Reranker forward-pass failed (%s). "
                "Falling back to base hybrid retrieval scores.",
                e,
            )
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates[:top_k]

        if len(all_scores) != len(candidates):
            logger.warning(
                "⚠️ DEGRADED MODE: Reranker output length mismatch. "
                "Falling back to base hybrid retrieval scores."
            )
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates[:top_k]

        for chunk, new_score in zip(candidates, all_scores, strict=True):
            chunk.hybrid_score = chunk.score
            chunk.score = new_score

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]
