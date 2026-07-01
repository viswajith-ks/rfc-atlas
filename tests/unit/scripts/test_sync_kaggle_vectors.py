import builtins
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.sync_kaggle_vectors import (
    _PROJECT_ROOT,  # pyright: ignore[reportPrivateUsage]
    KaggleOrchestrator,
)


@pytest.fixture
def mock_orchestrator() -> KaggleOrchestrator:
    with patch("scripts.sync_kaggle_vectors.KaggleApi") as mock_api:
        # Mock the config value resolution so _resolve_username passes
        mock_api.return_value.config_values = {"username": "test_user"}
        return KaggleOrchestrator()


def test_ephemeral_cleanup_input_restoration(
    mock_orchestrator: KaggleOrchestrator,
) -> None:
    original_input = builtins.input

    # We intentionally trigger a failure in the API to test the finally/exception block
    mock_orchestrator.api.dataset_delete.side_effect = ValueError("Kaggle died")  # pyright: ignore[reportAttributeAccessIssue]

    mock_orchestrator._cleanup_ephemeral_assets()  # pyright: ignore[reportPrivateUsage]

    # CRITICAL: Input must be restored to normal so the user's terminal isn't broken
    assert builtins.input is original_input


def test_build_pack_bundle_ignores(mock_orchestrator: KaggleOrchestrator) -> None:
    # We don't want to zip the entire real repo in a unit test because it's slow.
    # But we can test the `_should_ignore` logic directly

    # 1. Should Ignore
    assert (
        mock_orchestrator._should_ignore(_PROJECT_ROOT / ".venv" / "bin" / "python")  # pyright: ignore[reportPrivateUsage]
        is True
    )
    assert (
        mock_orchestrator._should_ignore(_PROJECT_ROOT / "__pycache__" / "test.pyc")  # pyright: ignore[reportPrivateUsage]
        is True
    )
    assert (
        mock_orchestrator._should_ignore(_PROJECT_ROOT / "data" / "raw" / "rfc.txt")  # pyright: ignore[reportPrivateUsage]
        is True
    )
    assert (
        mock_orchestrator._should_ignore(_PROJECT_ROOT / "tests" / "unit" / "test_x.py")  # pyright: ignore[reportPrivateUsage]
        is True
    )

    # 2. Should Include
    assert (
        mock_orchestrator._should_ignore(  # pyright: ignore[reportPrivateUsage]
            _PROJECT_ROOT / "scripts" / "sync_kaggle_vectors.py"
        )
        is False
    )
    assert (
        mock_orchestrator._should_ignore(_PROJECT_ROOT / "vector_store" / "schema.py")  # pyright: ignore[reportPrivateUsage]
        is False
    )


@patch("scripts.sync_kaggle_vectors.time.sleep", return_value=None)
def test_polling_state_machine(
    mock_sleep: MagicMock, mock_orchestrator: KaggleOrchestrator, tmp_path: Path
) -> None:

    # Mock the status endpoint to simulate Kaggle processing: Running -> Running -> Complete
    mock_status_1 = MagicMock()
    mock_status_1.status = "running"

    mock_status_2 = MagicMock()
    mock_status_2.status = "running"

    mock_status_3 = MagicMock()
    mock_status_3.status = "complete"

    mock_orchestrator.api.kernels_status.side_effect = [  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        mock_status_1,
        mock_status_2,
        mock_status_3,
    ]

    # Combine context managers to avoid nested 'with' blocks
    with (
        patch("scripts.sync_kaggle_vectors.LOCAL_EMBEDDINGS_DIR", tmp_path),
        patch.object(mock_orchestrator, "_cleanup_ephemeral_assets"),
    ):
        # Fake a downloaded parquet file so it doesn't trigger the "no parquets" crash
        (tmp_path / "fake.parquet").touch()

        # If this doesn't raise SystemExit, the state machine successfully broke out of the loop
        mock_orchestrator.poll_and_fetch_pipeline()

        # Assert the API was pinged exactly 3 times before breaking out
        assert mock_orchestrator.api.kernels_status.call_count == 3  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        mock_orchestrator.api.kernels_output.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

        # ACTUALLY USE THE MOCK: Assert the script slept exactly 2 times between the 3 pings
        assert mock_sleep.call_count == 2
