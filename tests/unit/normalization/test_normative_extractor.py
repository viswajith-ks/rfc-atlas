"""Unit tests for the BCP-14 Normative Requirement Extractor.

Validates exact word-boundary regex matches, keyword normalization (e.g., SHALL -> MUST),
sentence-boundary splitting, and strict structural exemption logic.
"""

from typing import cast

import pytest

from normalization.normative_extractor import NormativeExtractor
from normalization.schema import CanonicalBlockDict, NormativeKeyword


@pytest.fixture
def extractor() -> NormativeExtractor:
    """Provides a fresh instance of the NormativeExtractor."""
    return NormativeExtractor()


def _mock_block(
    block_type: str = "prose",
    hierarchy_path: str = "Document Root > 1. Introduction",
    text: str = "",
) -> CanonicalBlockDict:
    """Helper to generate structurally compliant mock blocks for testing."""
    return cast(
        CanonicalBlockDict,
        {
            "rfc_id": 9999,
            "hierarchy_path": hierarchy_path,
            "block_type": block_type,
            "source_type": "txt",
            "normalized_text": text,
            "source_fragment": text,
            "parsing_confidence": 1.0,
            "metadata": {"element_id": None},
        },
    )


@pytest.mark.parametrize(
    "path, expected_exempt",
    [
        # Standard non-exempt paths
        ("Document Root > 1. Introduction", False),
        ("Document Root > 2. Protocol Details", False),
        ("Document Root > Security Considerations", False),
        # Explicitly exempt sections
        ("Document Root > Abstract", True),
        ("Document Root > Table of Contents", True),
        ("Document Root > References", True),
        ("Document Root > Acknowledgments", True),
        # Back-matter catch-all
        ("Document Root > Back > Some Random Appendix", True),
        # Appendix stripping logic
        ("Document Root > Appendix A References", True),
        ("Document Root > Appendix B.1 Acknowledgements", True),
        (
            "Document Root > Appendix C Cool Protocol Stuff",
            False,
        ),  # Appendix but not an exempt title
    ],
)
def test_exemption_logic(
    extractor: NormativeExtractor, path: str, expected_exempt: bool
) -> None:
    """Ensure structural paths correctly trigger extraction bypasses."""
    # pylint/pyright bypass for testing private methods
    assert extractor._is_exempt(path) == expected_exempt  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "raw_keyword, expected_normalized",
    [
        ("MUST", "MUST"),
        ("SHALL", "MUST"),
        ("REQUIRED", "MUST"),
        ("MUST NOT", "MUST NOT"),
        ("SHALL NOT", "MUST NOT"),
        ("SHOULD", "SHOULD"),
        ("RECOMMENDED", "SHOULD"),
        ("SHOULD NOT", "SHOULD NOT"),
        ("NOT RECOMMENDED", "SHOULD NOT"),
        ("MAY", "MAY"),
        ("OPTIONAL", "MAY"),
    ],
)
def test_keyword_normalization(
    extractor: NormativeExtractor,
    raw_keyword: str,
    expected_normalized: NormativeKeyword,
) -> None:
    """Ensure legacy BCP-14 terms map strictly to the core keyword constraints."""
    text = f"The system {raw_keyword} do the thing."
    block = _mock_block(text=text)

    results = extractor.process_blocks([block])

    statements = results[0]["metadata"].get("normative_statements", [])
    assert len(statements) == 1
    assert statements[0]["keyword"] == expected_normalized
    assert statements[0]["statement_text"] == text


def test_sentence_boundary_isolation(extractor: NormativeExtractor) -> None:
    """Ensure the regex splits sentences and only extracts the sentence containing the keyword."""
    text = (
        "This is the first sentence. "
        "The system MUST log all critical errors! "
        "Does the system crash? "
        "It SHOULD NOT crash."
    )
    block = _mock_block(text=text)

    results = extractor.process_blocks([block])
    statements = results[0]["metadata"].get("normative_statements", [])

    assert len(statements) == 2
    assert statements[0]["keyword"] == "MUST"
    assert statements[0]["statement_text"] == "The system MUST log all critical errors!"

    assert statements[1]["keyword"] == "SHOULD NOT"
    assert statements[1]["statement_text"] == "It SHOULD NOT crash."


def test_word_boundary_safety(extractor: NormativeExtractor) -> None:
    """Ensure partial string matches (like MUSTARD) do not trigger false positives."""
    text = "Colonel MUSTARD SHALLOT eat the OPTIONALity."
    block = _mock_block(text=text)

    results = extractor.process_blocks([block])
    statements = results[0]["metadata"].get("normative_statements", [])

    # "MUSTARD", "SHALLOT", and "OPTIONALity" should all be ignored due to \b boundaries.
    assert len(statements) == 0


def test_block_type_filtering(extractor: NormativeExtractor) -> None:
    """Ensure syntax, code, and artwork blocks are bypassed to prevent false flags in comments."""
    code_text = "if (error) { // The system MUST log this }"

    code_block = _mock_block(block_type="sourcecode", text=code_text)
    prose_block = _mock_block(block_type="prose", text=code_text)

    results = extractor.process_blocks([code_block, prose_block])

    # The sourcecode block should be completely ignored
    assert "normative_statements" not in results[0]["metadata"]

    # The prose block should successfully extract the comment
    assert len(results[1]["metadata"].get("normative_statements", [])) == 1
