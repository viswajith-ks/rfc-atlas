import json
from pathlib import Path
from typing import Any

import pytest

from chunking.recursive_splitter import BatchChunker, Block
from normalization.schema import NormativeStatement, RFCMetadata

# Constants synchronized with the module configuration
CHUNK_LIMIT: int = 2000
OVERLAP: int = 250


@pytest.fixture
def chunker(tmp_path: Path) -> BatchChunker:
    return BatchChunker(batch_id=1, tmp_dir=tmp_path)


def test_pass_through_physics(chunker: BatchChunker) -> None:
    short_text: str = "This is a short normative paragraph. It MUST NOT be split."
    chunks: list[str] = chunker.split_text_with_overlap(short_text)

    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_sliding_window_overlap_physics(chunker: BatchChunker) -> None:
    massive_text: str = "A" * 5000
    chunks: list[str] = chunker.split_text_with_overlap(massive_text)

    # Chunk 1: indices 0 -> 2000
    # Chunk 2: indices 1750 -> 3750 (2000 limit - 250 overlap rollback)
    # Chunk 3: indices 3500 -> 5000
    assert len(chunks) == 3
    assert len(chunks[0]) == 2000
    assert len(chunks[1]) == 2000
    assert len(chunks[2]) == 1500

    # Extract the trailing 250 characters of Chunk 1 and verify they
    # perfectly match the leading characters of Chunk 2.
    overlap_region: str = chunks[0][-OVERLAP:]
    assert chunks[1].startswith(overlap_region)


def test_safe_boundary_snapping(chunker: BatchChunker) -> None:
    text: str = ("A" * 1900) + "\n" + ("B" * 49) + " " + ("C" * 150)

    chunks: list[str] = chunker.split_text_with_overlap(text)

    assert len(chunks) > 1
    # Evaluates whether the algorithm correctly snapped to the newline at index 1900
    # instead of the space at index 1950.
    assert chunks[0].endswith("\n")
    assert len(chunks[0]) == 1901


def test_metadata_conservation_integrity(chunker: BatchChunker, tmp_path: Path) -> None:
    long_text: str = "The system MUST log all errors. " + ("A" * 2500)

    fake_block: Block = Block(
        block_id="test-block-001",
        block_type="paragraph",
        source_fragment="",
        normalized_text=long_text,
        parsing_confidence=1,
        normative_statements=[
            NormativeStatement(
                keyword="MUST",
                statement_text="The system MUST log all errors.",
                referenced_rfcs=[],
            )
        ],
    )

    output_file: Path = tmp_path / "prose_batch_1.jsonl"
    chunker.handles["prose"] = output_file.open("w", encoding="utf-8")

    chunker._chunk_and_route(  # pyright: ignore[reportPrivateUsage]
        block=fake_block,
        rfc_number=9999,
        h_path=["Test"],
        rfc_metadata=RFCMetadata(rfc_number=9999, title="test", source_type="xml"),
    )

    chunker.handles["prose"].close()

    results: list[dict[str, Any]] = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]

    assert len(results) >= 2

    # Chunk 1 possesses the sentence fragment, so it must retain the metadata payload.
    assert len(results[0]["normative_statements"]) == 1

    # Chunk 2 lacks the sentence fragment; the metadata array must be empty.
    assert len(results[1]["normative_statements"]) == 0, (
        "CRITICAL RAG BUG: Normative metadata was blindly copied to a chunk that doesn't contain the rule!"
    )
