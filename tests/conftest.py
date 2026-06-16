"""Global test configuration and shared fixtures for the RFC Atlas test suite."""

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR: Path = Path(__file__).parent / "fixtures"
RAW_TXT_PATH: Path = FIXTURES_DIR / "raw_txt" / "synthetic_rfc_9999.txt"
RAW_XML_PATH: Path = FIXTURES_DIR / "raw_xml" / "synthetic_rfc_9999.xml"
EXPECTED_CHUNKS_PATH: Path = FIXTURES_DIR / "golden_outputs" / "expected_chunks.jsonl"
EXPECTED_TREE_PATH: Path = FIXTURES_DIR / "golden_outputs" / "expected_tree.json"


@pytest.fixture
def synthetic_txt_content() -> str:
    """Loads the raw string content of the synthetic TXT RFC.

    Returns:
        str: The raw plaintext of the synthetic document.
    """
    raw_content = RAW_TXT_PATH.read_text(encoding="utf-8")
    lines = raw_content.split("\n")
    return "\n".join(lines)


@pytest.fixture
def synthetic_xml_content() -> str:
    """Loads the raw string content of the synthetic XML RFC.

    Returns:
        str: The raw XML markup of the synthetic document.
    """
    return RAW_XML_PATH.read_text(encoding="utf-8")


@pytest.fixture
def expected_chunks() -> list[dict[str, Any]]:
    """Loads the golden expected output chunks from JSONL format.

    Returns:
        list[dict[str, Any]]: A list containing the validated expected chunk schemas.
    """
    chunks: list[dict[str, Any]] = []
    with open(EXPECTED_CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


@pytest.fixture
def expected_tree() -> dict[str, Any]:
    """Loads the golden expected CanonicalTree output.

    Returns:
        dict[str, Any]: A dictionary representing the expected hierarchical tree.
    """
    return json.loads(EXPECTED_TREE_PATH.read_text(encoding="utf-8"))
