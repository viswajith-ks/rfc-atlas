import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from rfc_atlas.normalization.schema import CanonicalBlockDict
from rfc_atlas.normalization.tree_builder import CanonicalTreeBuilder
from rfc_atlas.utils.exceptions import InvalidRFCNumberError

if TYPE_CHECKING:
    from rfc_atlas.metadata.schema import RFCIndexEntryDict


@pytest.fixture
def mock_metadata_file(tmp_path: Path) -> Path:
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
    return CanonicalTreeBuilder(metadata_lookup_path=mock_metadata_file)


def test_invalid_rfc_number_rejection(tree_builder: CanonicalTreeBuilder) -> None:
    with pytest.raises(InvalidRFCNumberError, match="Invalid RFC identifier"):
        tree_builder.build_tree(rfc_number=0, flat_blocks=[], source_type="txt")


def test_missing_metadata_fallback(tree_builder: CanonicalTreeBuilder) -> None:
    # RFC 9999 is NOT in our mock_metadata_file
    tree = tree_builder.build_tree(rfc_number=9999, flat_blocks=[], source_type="txt")

    assert tree.metadata.rfc_number == 9999
    assert tree.metadata.title == "RFC 9999"
    assert tree.metadata.status == "UNKNOWN"
    assert tree.metadata.published_at is None


def test_preface_routing(
    tree_builder: CanonicalTreeBuilder,
    mock_canonical_block: Callable[..., CanonicalBlockDict],
) -> None:
    blocks = [
        mock_canonical_block(h_path="Document Root", text="Preface Text 1"),
        mock_canonical_block(h_path="Preface", text="Preface Text 2"),
    ]
    tree = tree_builder.build_tree(
        rfc_number=1234, flat_blocks=blocks, source_type="txt"
    )

    assert len(tree.preface_blocks) == 2
    assert len(tree.sections) == 0


def test_section_block_counters_and_grouping(
    tree_builder: CanonicalTreeBuilder,
    mock_canonical_block: Callable[..., CanonicalBlockDict],
) -> None:
    blocks = [
        mock_canonical_block(h_path="1. Introduction", text="First paragraph"),
        mock_canonical_block(h_path="1. Introduction", text="Second paragraph"),
        mock_canonical_block(h_path="2. Security", text="Security paragraph"),
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
    mock_canonical_block: Callable[..., CanonicalBlockDict],
) -> None:
    # Two unnumbered blocks with the exact same path should hash to the same section token
    blocks = [
        mock_canonical_block(h_path="Random Unnumbered Section", text="A"),
        mock_canonical_block(h_path="Random Unnumbered Section", text="B"),
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
