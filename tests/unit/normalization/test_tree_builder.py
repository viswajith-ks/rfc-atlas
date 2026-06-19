"""Unit tests for the CanonicalTreeBuilder assembly logic."""

import json
from pathlib import Path

import pytest

from metadata.schema import RFCIndexEntryDict
from normalization.schema import CanonicalBlockDict, IntermediateBlockType
from normalization.tree_builder import CanonicalTreeBuilder


@pytest.fixture
def mock_metadata_file(tmp_path: Path) -> Path:
    """Provides a temporary metadata lookup JSON file.

    Args:
        tmp_path (Path): Pytest-provided temporary directory path.

    Returns:
        Path: The absolute path to the generated mock metadata JSON file.
    """
    meta_path: Path = tmp_path / "mock_meta.json"
    mock_data: dict[str, RFCIndexEntryDict] = {
        "1234": {
            "rfc_number": 1234,
            "title": "The Test Protocol",
            "published_at": {"year": 2026, "month": 6},
            "status": "PROPOSED STANDARD",
            "stream": "IETF",
            "authors": ["A. Tester"],
            "obsoletes": [],
            "updates": [],
            "updated_by": [],
            "protocol_family": None,
        }
    }
    meta_path.write_text(json.dumps(mock_data), encoding="utf-8")
    return meta_path


@pytest.fixture
def tree_builder(mock_metadata_file: Path) -> CanonicalTreeBuilder:
    """Provides an instantiated CanonicalTreeBuilder.

    Args:
        mock_metadata_file (Path): The path fixture pointing to the mock metadata JSON.

    Returns:
        CanonicalTreeBuilder: An initialized builder ready for test execution.
    """
    return CanonicalTreeBuilder(metadata_lookup_path=mock_metadata_file)


def _mock_flat_block(
    h_path: str, b_type: IntermediateBlockType = "prose", text: str = "Sample"
) -> CanonicalBlockDict:
    """Helper to generate structurally compliant flat blocks for testing.

    Args:
        h_path (str): The mock hierarchy path (e.g., '1. Introduction').
        b_type (IntermediateBlockType): The intermediate block type designation. Defaults to 'prose'.
        text (str): The mock normalized text payload. Defaults to 'Sample'.

    Returns:
        CanonicalBlockDict: A strictly typed dictionary mimicking parser output.
    """
    return CanonicalBlockDict(
        rfc_id=1234,
        hierarchy_path=h_path,
        block_type=b_type,
        source_type="txt",
        normalized_text=text,
        source_fragment=text,
        parsing_confidence=1.0,
        metadata={"element_id": None, "normative_statements": []},
    )


def test_invalid_rfc_number_rejection(tree_builder: CanonicalTreeBuilder) -> None:
    """Verifies that the builder strictly rejects non-positive RFC numbers.

    Args:
        tree_builder (CanonicalTreeBuilder): The instantiated test fixture.
    """
    with pytest.raises(ValueError, match="Cannot build canonical tree"):
        tree_builder.build_tree(rfc_number=0, flat_blocks=[], source_type="txt")


def test_missing_metadata_fallback(tree_builder: CanonicalTreeBuilder) -> None:
    """Verifies that an unknown RFC gracefully falls back to placeholder metadata.

    Args:
        tree_builder (CanonicalTreeBuilder): The instantiated test fixture.
    """
    # RFC 9999 is NOT in our mock_metadata_file
    tree = tree_builder.build_tree(rfc_number=9999, flat_blocks=[], source_type="txt")

    assert tree.metadata.rfc_number == 9999
    assert tree.metadata.title == "RFC 9999"
    assert tree.metadata.status == "UNKNOWN"
    assert tree.metadata.published_at is None


def test_preface_routing(tree_builder: CanonicalTreeBuilder) -> None:
    """Verifies blocks labeled as 'Document Root' or 'Preface' route to preface_blocks.

    Args:
        tree_builder (CanonicalTreeBuilder): The instantiated test fixture.
    """
    blocks = [
        _mock_flat_block("Document Root", text="Preface Text 1"),
        _mock_flat_block("Preface", text="Preface Text 2"),
    ]
    tree = tree_builder.build_tree(
        rfc_number=1234, flat_blocks=blocks, source_type="txt"
    )

    assert len(tree.preface_blocks) == 2
    assert len(tree.sections) == 0


def test_section_block_counters_and_grouping(
    tree_builder: CanonicalTreeBuilder,
) -> None:
    """Verifies that identical sections are grouped and block IDs increment sequentially.

    Args:
        tree_builder (CanonicalTreeBuilder): The instantiated test fixture.
    """
    blocks = [
        _mock_flat_block("1. Introduction", text="First paragraph"),
        _mock_flat_block("1. Introduction", text="Second paragraph"),
        _mock_flat_block("2. Security", text="Security paragraph"),
    ]
    tree = tree_builder.build_tree(
        rfc_number=1234, flat_blocks=blocks, source_type="txt"
    )

    # It should collapse the 3 blocks into exactly 2 hierarchical sections
    assert len(tree.sections) == 2

    intro_section = tree.sections[0]
    assert intro_section.title == "Introduction"
    assert len(intro_section.blocks) == 2

    # Verify the block counter incremented correctly!
    # Format should be: rfc1234-sec1-blk1 and rfc1234-sec1-blk2
    assert intro_section.blocks[0].block_id == "rfc1234-sec1-blk1"
    assert intro_section.blocks[1].block_id == "rfc1234-sec1-blk2"


def test_unknown_section_deterministic_hashing(
    tree_builder: CanonicalTreeBuilder,
) -> None:
    """Verifies that unnumbered sections generate stable, deterministic ID hashes.

    Args:
        tree_builder (CanonicalTreeBuilder): The instantiated test fixture.
    """
    # Two unnumbered blocks with the exact same path should hash to the same section token
    blocks = [
        _mock_flat_block("Random Unnumbered Section", text="A"),
        _mock_flat_block("Random Unnumbered Section", text="B"),
    ]
    tree = tree_builder.build_tree(
        rfc_number=1234, flat_blocks=blocks, source_type="txt"
    )

    assert len(tree.sections) == 1
    sec = tree.sections[0]

    # Verify the fallback hashing worked and grouped them together
    assert "secunknown-" in sec.blocks[0].block_id
    assert sec.blocks[0].block_id.endswith("-blk1")
    assert sec.blocks[1].block_id.endswith("-blk2")

    # Ensure the hashes are completely identical since the rfc_number and path are identical
    hash1 = sec.blocks[0].block_id.split("-blk")[0]
    hash2 = sec.blocks[1].block_id.split("-blk")[0]
    assert hash1 == hash2
