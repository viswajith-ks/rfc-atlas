import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from parsers.txt_parser import LegacyTextParser

if TYPE_CHECKING:
    from normalization.schema import CanonicalBlockDict


@pytest.fixture
def parser_instance(tmp_path: Path, synthetic_txt_content: str) -> LegacyTextParser:
    mock_file = tmp_path / "rfc9999.txt"
    mock_file.write_text(synthetic_txt_content, encoding="utf-8")
    return LegacyTextParser(mock_file)


def test_parser_initialization(parser_instance: LegacyTextParser) -> None:
    assert parser_instance.rfc_id == 9999
    assert math.isclose(parser_instance.base_confidence, 0.8)


def test_pagination_stripping(parser_instance: LegacyTextParser) -> None:
    clean_text = parser_instance._strip_pagination(parser_instance.raw_text)  # pyright: ignore[reportPrivateUsage]
    assert "\x0c" not in clean_text
    assert (
        "RFC 9999                     Synthetic Protocol            February 2026"
        not in clean_text
    )


def test_document_parsing_integration(parser_instance: LegacyTextParser) -> None:
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
    assert table_block["block_type"] in {"artwork", "table"}, (
        f"Table was misclassified as {table_block['block_type']}. "
        "Ensure the table lines in your synthetic_rfc_9999.txt start with exactly 4 spaces!"
    )

    # 3. Verify Hierarchy Path tracking (Tests stateful section depth tracking)
    security_blocks = [
        b for b in blocks if "Security Considerations" in b["hierarchy_path"]
    ]
    assert len(security_blocks) >= 1, "Failed to track header hierarchy to section 99"
