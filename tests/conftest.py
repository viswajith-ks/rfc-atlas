"""Global test configuration and shared fixtures for the RFC Atlas test suite."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from normalization.schema import CanonicalBlockDict, IntermediateBlockType

FIXTURES_DIR: Path = Path(__file__).parent / "fixtures"
RAW_TXT_PATH: Path = FIXTURES_DIR / "raw_txt" / "synthetic_rfc_9999.txt"
RAW_XML_PATH: Path = FIXTURES_DIR / "raw_xml" / "synthetic_rfc_9999.xml"
EXPECTED_CHUNKS_PATH: Path = FIXTURES_DIR / "golden_outputs" / "expected_chunks.jsonl"
EXPECTED_TREE_PATH: Path = FIXTURES_DIR / "golden_outputs" / "expected_tree.json"


@pytest.fixture
def synthetic_txt_content() -> str:
    return RAW_TXT_PATH.read_text(encoding="utf-8")


@pytest.fixture
def synthetic_xml_content() -> str:
    return RAW_XML_PATH.read_text(encoding="utf-8")


# TODO: Integration stage - This fixture is currently orphaned but will be
# consumed when validating LanceDB vector insertion and embedding pipelines.
@pytest.fixture
def expected_chunks() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with Path(EXPECTED_CHUNKS_PATH).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


@pytest.fixture
def expected_tree() -> dict[str, Any]:
    return json.loads(EXPECTED_TREE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def mock_canonical_block() -> Callable[..., CanonicalBlockDict]:

    def _factory(
        h_path: str = "Document Root > 1. Introduction",
        b_type: IntermediateBlockType = "prose",
        text: str = "Sample text.",
        rfc_id: int = 9999,
    ) -> CanonicalBlockDict:
        return CanonicalBlockDict(
            rfc_id=rfc_id,
            hierarchy_path=h_path,
            block_type=b_type,
            source_type="txt",
            normalized_text=text,
            source_fragment=text,
            parsing_confidence=1.0,
            metadata={"element_id": None},
        )

    return _factory
