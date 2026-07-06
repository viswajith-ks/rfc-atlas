from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rfc_atlas.retrieval.search_client import HybridSearchClient, RetrievalResult


@patch("rfc_atlas.retrieval.search_client.SentenceTransformer")
def test_encode_query_normalization(mock_st_class: MagicMock, tmp_path: Path) -> None:
    # 1. Setup the mocked SentenceTransformer
    mock_model_instance = MagicMock()
    mock_st_class.return_value = mock_model_instance

    # Simulate Nomic's native 768D output
    fake_768_vector = np.random.rand(1, 768).astype(np.float32)
    mock_model_instance.encode.return_value = fake_768_vector

    client = HybridSearchClient(db_path=tmp_path)
    encoded = client._encode_query("How does TCP work?")  # pyright: ignore[reportPrivateUsage]

    # 2. Verify Prefix Injection
    mock_model_instance.encode.assert_called_once()
    passed_texts = mock_model_instance.encode.call_args[0][0]
    assert passed_texts[0] == "search_query: How does TCP work?"

    # 3. Verify Truncation to 256 Dimensions
    assert len(encoded) == 256

    # 4. Verify strict L2 Normalization (The sum of squares must equal 1.0)
    sum_of_squares = sum(x * x for x in encoded)
    assert pytest.approx(sum_of_squares, rel=1e-5) == 1.0


def test_search_table_missing(tmp_path: Path) -> None:
    client = HybridSearchClient(db_path=tmp_path)

    # Mock the database connection to simulate an empty DB
    client._db = MagicMock()  # pyright: ignore[reportPrivateUsage]
    client._db.list_tables.return_value.tables = ["prose"]  # pyright: ignore[reportPrivateUsage]

    # 'abnf' doesn't exist, so it should gracefully return an empty list without crashing
    results = client.search_table("query", "abnf")
    assert results == []


@patch.object(HybridSearchClient, "_encode_query")
def test_search_table_success(mock_encode: MagicMock, tmp_path: Path) -> None:
    mock_encode.return_value = [0.1] * 256

    client = HybridSearchClient(db_path=tmp_path)
    client._db = MagicMock()  # pyright: ignore[reportPrivateUsage]
    client._db.list_tables.return_value.tables = ["prose"]  # pyright: ignore[reportPrivateUsage]

    # Mocking LanceDB's highly nested fluent API chain for vector text search:
    mock_table = MagicMock()
    mock_table.count_rows.return_value = 100

    mock_search = MagicMock()
    mock_vector = MagicMock()
    mock_text = MagicMock()
    mock_limit = MagicMock()

    mock_table.search.return_value = mock_search
    mock_search.vector.return_value = mock_vector
    mock_vector.text.return_value = mock_text
    mock_text.limit.return_value = mock_limit

    # Simulate LanceDB returning 2 raw dictionaries
    mock_limit.to_list.return_value = [
        {
            "chunk_id": "chunk-1",
            "rfc_number": 1000,
            "table_route": "prose",
            "hierarchy_path": "Root",
            "text_payload": "Text A",
            "_score": 0.95,
        },
        {
            "chunk_id": "chunk-2",
            "rfc_number": 2000,
            "text_payload": "Text B",
            "_score": 0.85,
            # Intentionally omit table_route and hierarchy_path to test fallbacks
        },
    ]

    client._db.open_table.return_value = mock_table  # pyright: ignore[reportPrivateUsage]

    results = client.search_table("TCP", "prose", limit=2)

    assert len(results) == 2
    assert isinstance(results[0], RetrievalResult)
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score == pytest.approx(0.95)

    # Verify graceful fallbacks for missing metadata columns
    assert results[1].chunk_id == "chunk-2"
    assert results[1].table_route == "prose"  # Defaulted to the queried table
    assert results[1].hierarchy_path == "Unknown"


@patch.object(HybridSearchClient, "search_table")
def test_search_multiple_no_truncation(
    mock_search_table: MagicMock, tmp_path: Path
) -> None:
    client = HybridSearchClient(db_path=tmp_path)

    # Simulate 'prose' returning 3 results, and 'security' returning 3 results
    def mock_table_return(
        _query: str, table_name: str, _limit: int
    ) -> list[RetrievalResult]:
        if table_name == "prose":
            return [
                RetrievalResult(
                    chunk_id=f"p-{i}",
                    rfc_number=1,
                    table_route="prose",
                    hierarchy_path="Path",
                    text_payload="Prose",
                    score=float(i),
                )
                for i in range(1, 4)
            ]
        if table_name == "security":
            return [
                RetrievalResult(
                    chunk_id=f"s-{i}",
                    rfc_number=1,
                    table_route="security",
                    hierarchy_path="Path",
                    text_payload="Security",
                    score=float(i) + 0.5,
                )
                for i in range(1, 4)
            ]
        return []

    mock_search_table.side_effect = mock_table_return

    # Call with limit=3
    combined = client.search_multiple("query", ["prose", "security"], limit=3)

    # 1. Verify determinism: It should sort the set("prose", "security") alphabetically
    assert mock_search_table.call_count == 2
    assert mock_search_table.call_args_list[0][0][1] == "prose"
    assert mock_search_table.call_args_list[1][0][1] == "security"

    # 2. PROVE THE FIX: It should return ALL 6 results without truncating down to 3!
    assert len(combined) == 6

    # 3. Verify it globally sorted the combined array by score (Highest to Lowest)
    assert combined[0].chunk_id == "s-3"  # Score 3.5
    assert combined[1].chunk_id == "p-3"  # Score 3.0
    assert combined[2].chunk_id == "s-2"  # Score 2.5
    assert combined[-1].chunk_id == "p-1"  # Score 1.0
