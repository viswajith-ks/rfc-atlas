import json
from pathlib import Path

import pytest

from rfc_atlas.utils.exceptions import SingletonViolationError
from rfc_atlas.vector_store.errata_ledger import ErrataLedger

pytestmark = pytest.mark.usefixtures("_reset_ledger")

# Mock data containing valid, rejected, and malformed errata
MOCK_ERRATA_DATA = [
    {
        "doc-id": "RFC1234",
        "errata_status_code": "Verified",
        "errata_type_code": "Technical",
        "section": "1.1",
        "orig_text": "The protocol runs on TCP port 80.",
        "correct_text": "The protocol runs on TCP port 443.",
    },
    {
        "doc-id": "RFC1234",
        "errata_status_code": "Reported",
        "errata_type_code": "Editorial",
        "section": "2.0",
        "orig_text": "hte data",
        "correct_text": "the data",
    },
    {
        "doc-id": "RFC1234",
        "errata_status_code": "Rejected",
        "errata_type_code": "Technical",
        "section": "3.0",
        "orig_text": "Bad claim",
        "correct_text": "Worse claim",
    },
    {
        "doc-id": "RFCBOGUS",
        "errata_status_code": "Verified",
    },
]


@pytest.fixture
def mock_errata_file(tmp_path: Path) -> Path:
    errata_path = tmp_path / "errata.json"
    errata_path.write_text(json.dumps(MOCK_ERRATA_DATA), encoding="utf-8")
    return errata_path


@pytest.fixture
def _reset_ledger() -> None:  # pyright: ignore[reportUnusedFunction]
    """Fixture to ensure the Singleton state is wiped clean before each test."""
    ErrataLedger._is_instantiated = False  # pyright: ignore[reportPrivateUsage]
    ErrataLedger._ledger.clear()  # pyright: ignore[reportPrivateUsage]
    ErrataLedger._is_loaded = False  # pyright: ignore[reportPrivateUsage]


def test_errata_ledger_singleton_physics() -> None:
    """Asserts that secondary instantiation raises a SingletonViolationError."""
    _ = ErrataLedger()
    with pytest.raises(SingletonViolationError):
        _ = ErrataLedger()


def test_errata_parsing_and_filtering(mock_errata_file: Path) -> None:
    """Asserts that IDs are correctly stripped, cast, and statuses are filtered."""
    ErrataLedger.force_reload(mock_errata_file)

    records_1234 = ErrataLedger.get_errata(1234)
    records_9999 = ErrataLedger.get_errata(9999)

    assert records_9999 == []
    assert len(records_1234) == 2

    statuses = [r["errata_status_code"] for r in records_1234]
    assert "Verified" in statuses
    assert "Reported" in statuses
    assert "Rejected" not in statuses


def test_errata_missing_or_corrupted_file(tmp_path: Path) -> None:
    """Asserts that bad files degrade gracefully without crashing the pipeline."""
    missing_file = tmp_path / "does_not_exist.json"

    ErrataLedger.force_reload(missing_file)
    assert ErrataLedger.get_errata(1234) == []

    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("THIS IS NOT JSON", encoding="utf-8")

    ErrataLedger.force_reload(corrupt_file)
    assert ErrataLedger.get_errata(1234) == []


def test_format_errata_for_llm(mock_errata_file: Path) -> None:
    """Asserts that the text injection string is perfectly formatted for Gemini."""
    ErrataLedger.force_reload(mock_errata_file)

    injection_string = ErrataLedger.format_errata_for_llm(1234)

    assert injection_string is not None
    assert "[CRITICAL ERRATA DETECTED FOR RFC 1234]" in injection_string
    assert "- Section 1.1: Technical Error (Verified)" in injection_string
    assert "ORIGINAL TEXT:\n  The protocol runs on TCP port 80." in injection_string
    assert "CORRECTED TEXT:\n  The protocol runs on TCP port 443." in injection_string

    assert ErrataLedger.format_errata_for_llm(9999) is None
