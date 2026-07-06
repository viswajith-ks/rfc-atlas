from typing import Any
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
    # Setup mocks to return empty/None (representing an active RFC with no typos)
    mock_graph.format_lineage_warning.return_value = None
    mock_errata_ledger.get_errata.return_value = []

    enriched = ContextInterceptor.enrich_results(mock_results)

    # Verify no flags were tripped
    assert not enriched[0].is_obsolete
    assert not enriched[0].has_errata
    assert enriched[0].text_payload == "This is standard text. No errata here."

    # Verify singletons were lazily loaded
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

    # Simulate a warning for RFC 1000
    def mock_warning(rfc_num: int) -> str | None:
        if rfc_num == 1000:
            return "[TEMPORAL WARNING: This chunk is from RFC 1000, which is OBSOLETE.]"
        return None

    mock_graph.format_lineage_warning.side_effect = mock_warning

    enriched = ContextInterceptor.enrich_results(mock_results)

    # 1. Verify chunk 1 was flagged and mutated
    assert enriched[0].is_obsolete is True
    assert "[TEMPORAL WARNING" in enriched[0].text_payload
    assert "This is standard text." in enriched[0].text_payload
    # Ensure the warning was PREPENDED to the top of the text
    assert enriched[0].text_payload.startswith("[TEMPORAL WARNING")

    # 2. Verify chunk 2 was untouched
    assert enriched[1].is_obsolete is False
    assert "[TEMPORAL WARNING" not in enriched[1].text_payload


@patch("rfc_atlas.retrieval.interceptor.TemporalLineageGraph")
@patch("rfc_atlas.retrieval.interceptor.ErrataLedger")
def test_errata_correction_injection(
    mock_errata_ledger: MagicMock,
    mock_graph: MagicMock,
    mock_results: list[RetrievalResult],
) -> None:
    mock_graph.format_lineage_warning.return_value = None

    # Simulate an erratum matching RFC 2000
    def mock_errata(rfc_num: int) -> list[dict[str, Any]]:
        if rfc_num == 2000:
            return [
                {
                    "errata_status_code": "Verified",
                    "orig_text": "The server runs on TCP port 80.",
                    "correct_text": "The server runs on TCP port 443.",
                }
            ]
        return []

    mock_errata_ledger.get_errata.side_effect = mock_errata

    enriched = ContextInterceptor.enrich_results(mock_results)

    # 1. Verify chunk 2 was flagged and mutated
    assert enriched[1].has_errata is True
    assert "TCP port 80." in enriched[1].text_payload
    assert "⚠️ [VERIFIED IETF ERRATA (Verified)]" in enriched[1].text_payload
    assert "TCP port 443." in enriched[1].text_payload

    # 2. Verify chunk 1 was untouched
    assert enriched[0].has_errata is False


@patch("rfc_atlas.retrieval.interceptor.TemporalLineageGraph")
@patch("rfc_atlas.retrieval.interceptor.ErrataLedger")
def test_errata_no_match(
    mock_errata_ledger: MagicMock,
    mock_graph: MagicMock,
    mock_results: list[RetrievalResult],
) -> None:
    mock_graph.format_lineage_warning.return_value = None

    # Simulate an erratum for RFC 1000, but the 'orig_text' DOES NOT exist in the chunk
    def mock_errata(rfc_num: int) -> list[dict[str, Any]]:
        if rfc_num == 1000:
            return [
                {
                    "errata_status_code": "Verified",
                    "orig_text": "Some text not in the chunk.",
                    "correct_text": "Fixed text.",
                }
            ]
        return []

    mock_errata_ledger.get_errata.side_effect = mock_errata

    enriched = ContextInterceptor.enrich_results(mock_results)

    # 1. Verify chunk 1 was NOT flagged because the text didn't match the payload
    assert enriched[0].has_errata is False
    assert "VERIFIED IETF ERRATA" not in enriched[0].text_payload
