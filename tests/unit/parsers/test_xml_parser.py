from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from rfc_atlas.parsers.xml_parser import ModernRFCParser

if TYPE_CHECKING:
    from rfc_atlas.normalization.schema import CanonicalBlockDict


@pytest.fixture
def parser_instance(tmp_path: Path, synthetic_xml_content: str) -> ModernRFCParser:
    # We must mock an XML file so lxml can parse it from disk
    mock_file = tmp_path / "rfc9999.xml"
    mock_file.write_text(synthetic_xml_content, encoding="utf-8")
    return ModernRFCParser(mock_file)


def test_parser_initialization(parser_instance: ModernRFCParser) -> None:
    assert parser_instance.rfc_id == 9999


def test_document_parsing_integration(parser_instance: ModernRFCParser) -> None:
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
    back_matter_blocks = [b for b in blocks if "Back > " in b["hierarchy_path"]]
    assert len(back_matter_blocks) >= 1, (
        "Failed to track hierarchy into the <back> matter"
    )

    # Verify the section inside <back> starts with the correct partition
    assert back_matter_blocks[0]["hierarchy_path"].startswith("Back > ")

    security_blocks = [b for b in blocks if b["block_type"] == "security"]
    assert len(security_blocks) >= 1, (
        "Security Considerations section should produce 'security' typed blocks"
    )
