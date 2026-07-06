from rfc_atlas.retrieval.context_builder import ContextBuilder
from rfc_atlas.retrieval.search_client import RetrievalResult


def test_format_context_empty_results() -> None:
    output = ContextBuilder.format_context([])
    assert output == "No relevant context found in the RFC Atlas database."


def test_format_context_with_results() -> None:
    mock_results = [
        RetrievalResult(
            chunk_id="rfc1234-sec1",
            rfc_number=1234,
            table_route="prose",
            hierarchy_path="1. Introduction",
            text_payload="This is the first chunk.",
            score=0.95,
        ),
        RetrievalResult(
            chunk_id="rfc5678-sec2",
            rfc_number=5678,
            table_route="abnf",
            hierarchy_path="2. Grammar > 2.1 Core",
            text_payload="core-rule = %x20-7E",
            score=0.85,
        ),
    ]

    output = ContextBuilder.format_context(mock_results)

    # 1. Verify Preamble
    assert "The following authoritative context blocks have been retrieved" in output

    # 2. Verify Chunk 1 formatting and citation header
    assert "### [SOURCE 1: RFC 1234 | Format: PROSE | Path: 1. Introduction]" in output
    assert "This is the first chunk." in output

    # 3. Verify Chunk 2 formatting and sequential numbering
    assert (
        "### [SOURCE 2: RFC 5678 | Format: ABNF | Path: 2. Grammar > 2.1 Core]"
        in output
    )
    assert "core-rule = %x20-7E" in output

    # 4. Verify Separator exists between blocks
    assert "-" * 65 in output
