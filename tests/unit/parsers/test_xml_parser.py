"""Unit tests for the ModernRFCParser xml2rfc engine.

This test suite validates the extraction logic for modern xml2rfc v3
documents (RFC 8650+). It ensures the lxml-based engine correctly navigates
the document tree, extracts specific attribute tags, and accurately tracks
document hierarchy paths (Front/Middle/Back matter).
"""

from pathlib import Path

import pytest

from normalization.schema import CanonicalBlockDict
from parsers.xml_parser import ModernRFCParser


@pytest.fixture
def parser_instance(tmp_path: Path, synthetic_xml_content: str) -> ModernRFCParser:
    """Initializes a ModernRFCParser against a synthetic xml2rfc v3 fixture.

    Args:
        tmp_path (Path): Pytest fixture for generating temporary directories.
        synthetic_xml_content (str): The raw string content of the mock XML RFC.

    Returns:
        ModernRFCParser: An instantiated parser with a loaded lxml element tree.
    """
    # We must mock an XML file so lxml can parse it from disk
    mock_file = tmp_path / "rfc9999.xml"
    mock_file.write_text(synthetic_xml_content, encoding="utf-8")
    return ModernRFCParser(mock_file)


def test_parser_initialization(parser_instance: ModernRFCParser) -> None:
    """Verifies standard parser bootstrapping.

    Ensures that the parser successfully parses the XML tree and dynamically
    extracts the target RFC ID from the root `<rfc number="...">` attribute.
    """
    assert parser_instance.rfc_id == 9999


def test_document_parsing_integration(parser_instance: ModernRFCParser) -> None:
    """Validates the full XML tree traversal and canonical block formatting.

    This test ensures that:
    1. Native XML tags (`<sourcecode type="abnf">`) are perfectly mapped to
       their corresponding intermediate block types.
    2. HTML-style `<table>` structures are safely ingested.
    3. The recursive section tracker properly identifies structural boundaries
       like `<back>` to maintain exact vector provenance.
    """
    blocks: list[CanonicalBlockDict] = parser_instance.parse_document()

    # 1. Verify ABNF extraction from the <sourcecode type="abnf"> tag
    abnf_blocks = [b for b in blocks if b["block_type"] == "abnf"]
    assert len(abnf_blocks) == 1, (
        "Failed to route <sourcecode type='abnf'> to block_type='abnf'"
    )

    # 2. Verify Table extraction (Direct mapping from <table> tag)
    table_blocks = [b for b in blocks if b["block_type"] == "table"]
    assert len(table_blocks) == 1, "Failed to extract the <table> element"

    # 3. Verify Back Matter boundary tracking (Ensures recursive tree lineage)
    security_blocks = [
        b for b in blocks if "Security Considerations" in b["hierarchy_path"]
    ]
    assert len(security_blocks) >= 1, "Failed to track hierarchy into the <back> matter"
    # Ensure it appended the explicit "Back" partition to the hierarchy string
    assert security_blocks[0]["hierarchy_path"].startswith("Back > ")
