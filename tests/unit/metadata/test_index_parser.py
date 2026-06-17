"""Unit tests for the RFCIndexParser XML extraction and metadata compilation logic."""

import json
from pathlib import Path

import pytest
from lxml import etree

from metadata.index_parser import RFCIndexParser

# A minimal, valid XML snippet matching the official IETF rfc-index schema
SYNTHETIC_INDEX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rfc-index xmlns="http://www.rfc-editor.org/rfc-index">
    <rfc-entry>
        <doc-id>RFC1234</doc-id>
        <title>The First Mock Protocol</title>
        <author><name>A. Nonymous</name></author>
        <author><name>B. Tester</name></author>
        <date><month>January</month><year>2020</year></date>
        <current-status>PROPOSED STANDARD</current-status>
        <stream>IETF</stream>
        <obsoletes><doc-id>RFC1000</doc-id></obsoletes>
        <updates><doc-id>RFC1111</doc-id></updates>
        <updated-by><doc-id>RFC9999</doc-id></updated-by>
    </rfc-entry>
    <rfc-entry>
        <doc-id>RFC2345</doc-id>
        <title>Fuzzy Month Test</title>
        <date><month>Spring</month><year>2021</year></date>
        <current-status>INFORMATIONAL</current-status>
        <stream>IRTF</stream>
    </rfc-entry>
</rfc-index>
"""


@pytest.fixture
def parser_workspace(tmp_path: Path) -> tuple[Path, Path]:
    """Provides a temporary workspace with a mock XML index and an output target.

    Args:
        tmp_path (Path): Pytest-provided temporary directory path.

    Returns:
        tuple[Path, Path]: A tuple containing the (xml_source_path, json_output_path).
    """
    xml_path = tmp_path / "rfc-index.xml"
    json_path = tmp_path / "rfc_metadata_lookup.json"

    xml_path.write_text(SYNTHETIC_INDEX_XML, encoding="utf-8")

    return xml_path, json_path


def test_successful_parsing_and_relationships(
    parser_workspace: tuple[Path, Path],
) -> None:
    """Verifies standard extraction, relationship mapping, and atomic JSON serialization.

    Args:
        parser_workspace (tuple[Path, Path]): The mock XML and JSON paths.
    """
    xml_path, json_path = parser_workspace
    parser = RFCIndexParser(xml_path=xml_path, output_path=json_path)

    # Execute the streaming iterparse
    parser.parse()

    # 1. Verify Memory Ledger
    assert "1234" in parser.metadata_dict
    assert "2345" in parser.metadata_dict

    entry_1234 = parser.metadata_dict["1234"]
    assert entry_1234["title"] == "The First Mock Protocol"
    assert entry_1234["authors"] == ["A. Nonymous", "B. Tester"]
    assert entry_1234["stream"] == "IETF"
    assert entry_1234["status"] == "PROPOSED STANDARD"

    # Verify arrays extracted successfully
    assert entry_1234["obsoletes"] == [1000]
    assert entry_1234["updates"] == [1111]
    assert entry_1234["updated_by"] == [9999]

    # 2. Verify Disk Serialization
    assert json_path.exists()
    with json_path.open(encoding="utf-8") as f:
        disk_data = json.load(f)
        assert "1234" in disk_data
        assert "2345" in disk_data


def test_fuzzy_month_resolution(parser_workspace: tuple[Path, Path]) -> None:
    """Verifies that non-standard IETF date strings (e.g., 'Spring') resolve correctly.

    Args:
        parser_workspace (tuple[Path, Path]): The mock XML and JSON paths.
    """
    xml_path, json_path = parser_workspace
    parser = RFCIndexParser(xml_path=xml_path, output_path=json_path)
    parser.parse()

    # RFC 1234 is explicitly "January"
    date_1234 = parser.metadata_dict["1234"]["published_at"]
    assert date_1234 is not None
    assert date_1234["year"] == 2020
    assert date_1234["month"] == 1

    # RFC 2345 is "Spring" (resolves to 4 in the _FUZZY_MONTH_MAP)
    date_2345 = parser.metadata_dict["2345"]["published_at"]
    assert date_2345 is not None
    assert date_2345["year"] == 2021
    assert date_2345["month"] == 4


def test_missing_xml_file_handling(tmp_path: Path) -> None:
    """Verifies that the parser safely aborts if the foundational XML is missing.

    Args:
        tmp_path (Path): Pytest-provided temporary directory path.
    """
    missing_xml = tmp_path / "does_not_exist.xml"
    out_json = tmp_path / "out.json"

    parser = RFCIndexParser(xml_path=missing_xml, output_path=out_json)

    with pytest.raises(FileNotFoundError, match="Missing RFC index"):
        parser.parse()


def test_malformed_xml_handling(tmp_path: Path) -> None:
    """Verifies that the parser catches and surfaces structural XML corruption.

    Args:
        tmp_path (Path): Pytest-provided temporary directory path.
    """
    corrupt_xml = tmp_path / "corrupt.xml"
    out_json = tmp_path / "out.json"

    # Write a purposely broken XML string
    corrupt_xml.write_text(
        "<?xml version='1.0'?><rfc-index><bad-tag>No closure", encoding="utf-8"
    )

    parser = RFCIndexParser(xml_path=corrupt_xml, output_path=out_json)

    with pytest.raises(etree.ParseError):
        parser.parse()
