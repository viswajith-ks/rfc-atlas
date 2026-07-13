from unittest.mock import MagicMock, patch

import pytest

from rfc_atlas.retrieval.interceptor import ContextInterceptor
from rfc_atlas.retrieval.search_client import RetrievalResult


@pytest.fixture
def mock_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="rfc1000-sec1",
            rfc_number=1000,
            table_route="prose",
            hierarchy_path="Root",
            text_payload="This is standard text. No errata here.",
            score=0.9,
        ),
        RetrievalResult(
            chunk_id="rfc2000-sec2",
            rfc_number=2000,
            table_route="prose",
            hierarchy_path="Root",
            text_payload="The server runs on TCP port 80.",
            score=0.8,
        ),
    ]


@patch("rfc_atlas.retrieval.interceptor.TemporalLineageGraph")
@patch("rfc_atlas.retrieval.interceptor.ErrataLedger")
def test_enrich_results_no_modifications(
    mock_errata_ledger: MagicMock,
    mock_graph: MagicMock,
    mock_results: list[RetrievalResult],
) -> None:
    mock_graph.format_lineage_warning.return_value = None
    mock_graph.get_node.return_value = None  # Simulate active/clean standard
    mock_errata_ledger.get_errata.return_value = []

    enriched = ContextInterceptor.enrich_results(mock_results)

    assert enriched[0].is_obsolete is False
    assert enriched[0].has_errata is False
    assert enriched[0].text_payload == "This is standard text. No errata here."

    mock_graph.load.assert_called_once()
    mock_errata_ledger.load.assert_called_once()


@patch("rfc_atlas.retrieval.interceptor.TemporalLineageGraph")
@patch("rfc_atlas.retrieval.interceptor.ErrataLedger")
def test_temporal_lineage_injection(
    mock_errata_ledger: MagicMock,
    mock_graph: MagicMock,
    mock_results: list[RetrievalResult],
) -> None:
    mock_errata_ledger.get_errata.return_value = []

    # Return a mocked structured node unconditionally to guarantee truthy evaluation
    mock_node = MagicMock()
    mock_node.obsoleted_by = {2000}
    mock_graph.get_node.return_value = mock_node
    mock_graph.format_lineage_warning.return_value = "[TEMPORAL WARNING: OBSOLETE]"

    enriched = ContextInterceptor.enrich_results(mock_results)

    # 1. Verify chunk was flagged and mutated
    assert enriched[0].is_obsolete is True
    assert "[TEMPORAL WARNING" in enriched[0].text_payload
    assert enriched[0].text_payload.startswith("[TEMPORAL WARNING")


@patch("rfc_atlas.retrieval.interceptor.TemporalLineageGraph")
@patch("rfc_atlas.retrieval.interceptor.ErrataLedger")
def test_errata_correction_injection(
    mock_errata_ledger: MagicMock,
    mock_graph: MagicMock,
    mock_results: list[RetrievalResult],
) -> None:
    mock_graph.format_lineage_warning.return_value = None
    mock_graph.get_node.return_value = None

    # Unconditionally return the erratum mock
    mock_errata_ledger.get_errata.return_value = [
        {
            "errata_status_code": "Verified",
            "orig_text": "The server runs on TCP port 80.",
            "correct_text": "The server runs on TCP port 443.",
        }
    ]

    enriched = ContextInterceptor.enrich_results(mock_results)

    assert enriched[1].has_errata is True
    assert "TCP port 80." in enriched[1].text_payload
    assert "⚠️ [IETF ERRATA (VERIFIED)]" in enriched[1].text_payload
    assert "TCP port 443." in enriched[1].text_payload
    assert enriched[0].has_errata is False


@patch("rfc_atlas.retrieval.interceptor.TemporalLineageGraph")
@patch("rfc_atlas.retrieval.interceptor.ErrataLedger")
def test_errata_no_match(
    mock_errata_ledger: MagicMock,
    mock_graph: MagicMock,
    mock_results: list[RetrievalResult],
) -> None:
    mock_graph.format_lineage_warning.return_value = None
    mock_graph.get_node.return_value = None

    mock_errata_ledger.get_errata.return_value = [
        {
            "errata_status_code": "Verified",
            "orig_text": "Some text not in the chunk.",
            "correct_text": "Fixed text.",
        }
    ]

    enriched = ContextInterceptor.enrich_results(mock_results)

    assert enriched[0].has_errata is False
    assert "IETF ERRATA" not in enriched[0].text_payload


@patch("rfc_atlas.retrieval.interceptor.TemporalLineageGraph")
@patch("rfc_atlas.retrieval.interceptor.ErrataLedger")
def test_interceptor_idempotency_guard(
    mock_errata_ledger: MagicMock,
    mock_graph: MagicMock,
    mock_results: list[RetrievalResult],
) -> None:
    mock_node = MagicMock()
    mock_node.obsoleted_by = {3000}
    mock_graph.get_node.return_value = mock_node
    mock_graph.format_lineage_warning.return_value = "[TEMPORAL WARNING]"

    mock_errata_ledger.get_errata.return_value = [
        {
            "errata_status_code": "Verified",
            "orig_text": "The server runs on TCP port 80.",
            "correct_text": "The server runs on TCP port 443.",
        }
    ]

    # Pass 1
    enriched_first = ContextInterceptor.enrich_results(mock_results)

    # Store exact string snapshots from the first pass
    text_0_pass1 = enriched_first[0].text_payload
    text_1_pass1 = enriched_first[1].text_payload

    # Pass 2
    enriched_second = ContextInterceptor.enrich_results(enriched_first)

    text_0_pass2 = enriched_second[0].text_payload
    text_1_pass2 = enriched_second[1].text_payload

    # Compare exact strings to guarantee idempotency worked and returned early
    assert text_0_pass1 == text_0_pass2, "Idempotency failed: String 0 mutated"
    assert text_1_pass1 == text_1_pass2, "Idempotency failed: String 1 mutated"
