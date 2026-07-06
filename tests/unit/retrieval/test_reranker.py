from unittest.mock import MagicMock, patch

import pytest

from rfc_atlas.retrieval.reranker import SemanticReranker
from rfc_atlas.retrieval.search_client import RetrievalResult


@pytest.fixture
def mock_candidates() -> list[RetrievalResult]:
    # Creating candidates with predefined LanceDB hybrid scores.
    # Chunk 1 is rated highest by LanceDB, Chunk 3 is lowest.
    return [
        RetrievalResult(
            chunk_id="chunk-1",
            rfc_number=1000,
            table_route="prose",
            hierarchy_path="Root",
            text_payload="First candidate.",
            score=0.9,
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            rfc_number=1000,
            table_route="prose",
            hierarchy_path="Root",
            text_payload="Second candidate.",
            score=0.8,
        ),
        RetrievalResult(
            chunk_id="chunk-3",
            rfc_number=1000,
            table_route="prose",
            hierarchy_path="Root",
            text_payload="Third candidate.",
            score=0.7,
        ),
    ]


@patch("rfc_atlas.retrieval.reranker.HAS_ML_LIBS", new=False)
def test_degrade_gracefully_missing_libraries(
    mock_candidates: list[RetrievalResult],
) -> None:
    reranker = SemanticReranker()

    # Execute rerank without torch/transformers installed
    results = reranker.rerank("test query", mock_candidates, top_k=2)

    # 1. It should not crash.
    # 2. It should return the top 2 items based on the ORIGINAL LanceDB scores.
    assert len(results) == 2
    assert results[0].chunk_id == "chunk-1"
    assert results[1].chunk_id == "chunk-2"


@patch.object(SemanticReranker, "_ensure_loaded")
def test_degrade_gracefully_initialization_crash(
    mock_ensure_loaded: MagicMock,
    mock_candidates: list[RetrievalResult],
) -> None:
    # Simulate a CUDA OOM or weight-download timeout during initialization
    mock_ensure_loaded.side_effect = RuntimeError("CUDA Out of Memory")

    reranker = SemanticReranker()
    results = reranker.rerank("test query", mock_candidates, top_k=2)

    # It should catch the RuntimeError and fall back to the original top 2
    assert len(results) == 2
    assert results[0].chunk_id == "chunk-1"


@patch.object(SemanticReranker, "_ensure_loaded")
@patch.object(SemanticReranker, "_compute_scores")
def test_degrade_gracefully_forward_pass_crash(
    mock_compute: MagicMock,
    mock_ensure_loaded: MagicMock,
    mock_candidates: list[RetrievalResult],
) -> None:
    mock_ensure_loaded.return_value = None

    # Simulate the model loading fine, but crashing during the actual matrix math
    mock_compute.side_effect = ValueError("Tensor size mismatch")

    reranker = SemanticReranker()

    # Shuffle the input candidates to prove the fallback explicitly re-sorts them correctly
    shuffled_candidates = [mock_candidates[2], mock_candidates[0], mock_candidates[1]]

    results = reranker.rerank("test query", shuffled_candidates, top_k=3)

    # Even though we passed them in shuffled, the fallback MUST sort them by original score!
    assert len(results) == 3
    assert results[0].chunk_id == "chunk-1"  # Score 0.9
    assert results[1].chunk_id == "chunk-2"  # Score 0.8
    assert results[2].chunk_id == "chunk-3"  # Score 0.7


@patch.object(SemanticReranker, "_ensure_loaded")
@patch.object(SemanticReranker, "_compute_scores")
def test_successful_semantic_reordering(
    mock_compute: MagicMock,
    mock_ensure_loaded: MagicMock,
    mock_candidates: list[RetrievalResult],
) -> None:
    mock_ensure_loaded.return_value = None

    # The Cross-Encoder reads the text and decides Chunk 3 is actually the best match!
    # It assigns new scores: Chunk 1 -> -1.5, Chunk 2 -> 0.0, Chunk 3 -> 5.5
    mock_compute.return_value = [-1.5, 0.0, 5.5]

    reranker = SemanticReranker()
    results = reranker.rerank("test query", mock_candidates, top_k=2)

    # 1. It should successfully truncate to top_k = 2
    assert len(results) == 2

    # 2. It should REORDER the chunks based entirely on the new semantic scores
    assert results[0].chunk_id == "chunk-3"
    assert results[0].score == pytest.approx(5.5)

    assert results[1].chunk_id == "chunk-2"
    assert results[1].score == pytest.approx(0.0)
