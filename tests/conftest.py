"""Global test configuration and shared fixtures for the RFC Atlas test suite."""

import json
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest

from rfc_atlas.graph.lineage import TemporalLineageGraph
from rfc_atlas.normalization.schema import CanonicalBlockDict, IntermediateBlockType
from rfc_atlas.vector_store.errata_ledger import ErrataLedger

FIXTURES_DIR: Path = Path(__file__).parent / "fixtures"
RAW_TXT_PATH: Path = FIXTURES_DIR / "raw_txt" / "synthetic_rfc_9999.txt"
RAW_XML_PATH: Path = FIXTURES_DIR / "raw_xml" / "synthetic_rfc_9999.xml"
EXPECTED_TREE_PATH: Path = FIXTURES_DIR / "golden_outputs" / "expected_tree.json"


@pytest.fixture
def synthetic_txt_content() -> str:
    return RAW_TXT_PATH.read_text(encoding="utf-8")


@pytest.fixture
def synthetic_xml_content() -> str:
    return RAW_XML_PATH.read_text(encoding="utf-8")


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


@pytest.fixture
def reset_singletons() -> Generator[None, None, None]:
    TemporalLineageGraph.is_instantiated = False
    TemporalLineageGraph._graph.clear()  # pyright: ignore[reportPrivateUsage]
    TemporalLineageGraph._is_loaded = False  # pyright: ignore[reportPrivateUsage]

    ErrataLedger.is_instantiated = False
    ErrataLedger._ledger.clear()  # pyright: ignore[reportPrivateUsage]
    ErrataLedger._is_loaded = False  # pyright: ignore[reportPrivateUsage]

    yield

    TemporalLineageGraph.is_instantiated = False
    TemporalLineageGraph._graph.clear()  # pyright: ignore[reportPrivateUsage]
    TemporalLineageGraph._is_loaded = False  # pyright: ignore[reportPrivateUsage]

    ErrataLedger.is_instantiated = False
    ErrataLedger._ledger.clear()  # pyright: ignore[reportPrivateUsage]
    ErrataLedger._is_loaded = False  # pyright: ignore[reportPrivateUsage]
