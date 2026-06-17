"""Integration tests validating the end-to-end pipeline against locked Golden outputs."""

import json
import shutil
from pathlib import Path
from typing import Any

from normalization.normative_extractor import NormativeExtractor
from normalization.schema import CanonicalBlockDict
from normalization.tree_builder import CanonicalTreeBuilder
from parsers.txt_parser import LegacyTextParser


def test_golden_tree_assembly(expected_tree: dict[str, Any], tmp_path: Path) -> None:
    """Validates that parsing + extraction + assembly produces the exact Golden JSON tree.

    This acts as a strict regression test for Phase 1 and Phase 2. If any regex,
    spacing, or structural hash logic changes in the codebase, this test will fail.

    Args:
        expected_tree (dict[str, Any]): The loaded Golden Snapshot dictionary fixture.
        tmp_path (Path): Pytest-provided temporary directory path for safe file manipulation.

    Returns:
        None
    """
    # 1. Setup paths based on standard project structure
    project_root = Path(__file__).resolve().parent.parent.parent
    synthetic_txt = (
        project_root / "tests" / "fixtures" / "raw_txt" / "synthetic_rfc_9999.txt"
    )
    metadata_lookup = project_root / "data" / "metadata" / "rfc_metadata_lookup.json"

    # The parser strictly requires filenames in the format rfcXXXX.txt
    # We copy the fixture to a temporary compliant path to satisfy the regex
    compliant_txt = tmp_path / "rfc9999.txt"
    shutil.copy(synthetic_txt, compliant_txt)

    # We must ensure the metadata lookup exists for the tree builder to run.
    # If it doesn't exist, we fall back to a safely mocked temporary path.
    if not metadata_lookup.exists():
        metadata_lookup = tmp_path / "mock_metadata.json"
        metadata_lookup.write_text(json.dumps({}), encoding="utf-8")

    # 2. Run the Parser (using the correct method and strict single-arg init)
    parser = LegacyTextParser(compliant_txt)
    raw_blocks: list[CanonicalBlockDict] = parser.parse_document()

    # 3. Run the Extractor
    extractor = NormativeExtractor()
    enriched_blocks = extractor.process_blocks(raw_blocks)

    # 4. Run the Builder
    builder = CanonicalTreeBuilder(metadata_lookup_path=metadata_lookup)
    normalized_tree = builder.build_tree(
        rfc_number=9999, flat_blocks=enriched_blocks, source_type="txt"
    )

    # 5. Serialize and compare the dictionaries to ensure exact structural parity
    # We dump and load to convert Pydantic models to pure Python dicts for clean == comparison.
    actual_tree_dict = json.loads(normalized_tree.model_dump_json(exclude_none=True))

    # We ignore the 'metadata' block in the comparison because published_at/status
    # might fall back to "UNKNOWN" depending on the local lookup file presence.
    # We only care that the physical document structure and normative extraction matches.
    assert actual_tree_dict["sections"] == expected_tree["sections"]
    assert actual_tree_dict["preface_blocks"] == expected_tree["preface_blocks"]
