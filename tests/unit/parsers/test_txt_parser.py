"""Unit tests for the LegacyTextParser heuristic engine.

This test suite validates the extraction and classification logic for plaintext
IETF RFC documents (RFC 1 - 8649). It ensures that the parser correctly handles
legacy pagination, structural breadcrumbs, and exact indentation requirements
for ABNF and ASCII artwork.
"""

from pathlib import Path

import pytest

from normalization.schema import CanonicalBlockDict
from parsers.txt_parser import LegacyTextParser


@pytest.fixture
def parser_instance(tmp_path: Path, synthetic_txt_content: str) -> LegacyTextParser:
    """Initializes a LegacyTextParser against a synthetic IETF plaintext fixture.

    Args:
        tmp_path (Path): Pytest fixture for generating temporary directories.
        synthetic_txt_content (str): The raw string content of the mock RFC.

    Returns:
        LegacyTextParser: An instantiated parser loaded with the mock document.
    """
    mock_file = tmp_path / "rfc9999.txt"
    mock_file.write_text(synthetic_txt_content, encoding="utf-8")
    return LegacyTextParser(mock_file)


def test_parser_initialization(parser_instance: LegacyTextParser) -> None:
    """Verifies standard parser bootstrapping and metadata extraction.

    Ensures that the parser safely extracts the numeric RFC ID from the filename
    and applies the correct parsing confidence threshold based on the RFC era
    (modern vs. early).
    """
    assert parser_instance.rfc_id == 9999
    assert parser_instance.base_confidence == 0.8


def test_pagination_stripping(parser_instance: LegacyTextParser) -> None:
    """Validates the context-aware pagination removal heuristic.

    Ensures that form-feed characters (\\x0c), running headers, and footers
    are removed from the raw text stream. This tests the algorithmic "glue"
    that decides whether to stitch cross-page sentences together or split
    them apart based on indentation and header context.
    """
    clean_text = parser_instance._strip_pagination(parser_instance.raw_text)  # pyright: ignore[reportPrivateUsage]
    assert "\x0c" not in clean_text
    assert (
        "RFC 9999                     Synthetic Protocol            February 2026"
        not in clean_text
    )


def test_document_parsing_integration(parser_instance: LegacyTextParser) -> None:
    """Validates the complete end-to-end extraction and routing pipeline.

    This test triggers the two-pass block evaluation engine to ensure:
    1. ABNF code blocks are detected via keyword/syntax scoring.
    2. ASCII tables are preserved via strict 4-space indentation matching
       (immune to invisible control-character corruption).
    3. Hierarchical lineage (breadcrumbs) correctly updates when navigating
       into specific sections (e.g., 'Security Considerations').
    """
    blocks: list[CanonicalBlockDict] = parser_instance.parse_document()

    # 1. Verify ABNF Classification (Tests regex scoring heuristic)
    abnf_blocks = [b for b in blocks if b["block_type"] == "abnf"]
    assert len(abnf_blocks) == 1, (
        "Failed to identify the ABNF code block. Check 4-space indent in fixture."
    )
    assert 'protocol = "SEP" SP 1*DIGIT' in abnf_blocks[0]["normalized_text"]

    # 2. Verify Table Classification (Tests control-character blindness and indentation)
    table_block = next(
        (b for b in blocks if "| State | Metric | Notes |" in b["normalized_text"]),
        None,
    )
    assert table_block is not None, "Table text was completely lost during parsing."
    assert table_block["block_type"] in ["artwork", "table"], (
        f"Table was misclassified as {table_block['block_type']}. "
        "Ensure the table lines in your synthetic_rfc_9999.txt start with exactly 4 spaces!"
    )

    # 3. Verify Hierarchy Path tracking (Tests stateful section depth tracking)
    security_blocks = [
        b for b in blocks if "Security Considerations" in b["hierarchy_path"]
    ]
    assert len(security_blocks) >= 1, "Failed to track header hierarchy to section 99"
