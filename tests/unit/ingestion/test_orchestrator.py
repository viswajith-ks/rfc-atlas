"""Unit tests for the multi-core Pipeline Orchestrator.

This suite validates the protective Singleton lock, the era-based XML
superiority routing logic, and the Pebble multiprocessing crash handlers
(simulating hard exceptions and timeouts) without requiring actual heavy I/O.
"""

from collections.abc import Generator
from concurrent.futures import Future, TimeoutError
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import ingestion.orchestrator
from ingestion.orchestrator import PipelineOrchestrator


@pytest.fixture
def orchestrator(tmp_path: Path) -> Generator[PipelineOrchestrator, None, None]:
    """Provides a fresh, isolated PipelineOrchestrator instance securely locked per test.

    This fixture creates the required directory structures and a mock metadata
    ledger to satisfy the CanonicalTreeBuilder initialization constraints. It safely
    yields the orchestrator instance and guarantees that both the Singleton lock
    and the global Copy-on-Write memory cache are purged during teardown, preventing
    cross-test contamination.

    Args:
        tmp_path (Path): Pytest-provided temporary directory path.

    Yields:
        Generator[PipelineOrchestrator, None, None]: An initialized orchestrator instance.
    """
    PipelineOrchestrator.reset_state()

    txt_dir = tmp_path / "txt"
    xml_dir = tmp_path / "xml"
    txt_dir.mkdir()
    xml_dir.mkdir()

    # Create a fake metadata ledger so the CoW CanonicalTreeBuilder doesn't crash
    meta_path = tmp_path / "meta.json"
    meta_path.write_text("{}", encoding="utf-8")

    orch = PipelineOrchestrator(
        raw_txt_dir=txt_dir,
        raw_xml_dir=xml_dir,
        output_dir=tmp_path / "out",
        manifest_dir=tmp_path / "manifest",
        raw_index_path=tmp_path / "index.xml",
        metadata_path=meta_path,
    )

    yield orch

    # Teardown: Release the Singleton lock and reset the global memory cache
    PipelineOrchestrator.reset_state()
    ingestion.orchestrator._worker_tree_builder = None  # pyright: ignore[reportPrivateUsage]


def test_singleton_lock(tmp_path: Path, orchestrator: PipelineOrchestrator) -> None:
    """Verifies that the orchestrator enforces strict Singleton instantiation.

    Because the ingestion pipeline relies heavily on Linux-native Copy-on-Write
    via `fork()` to share the massive metadata index across worker processes,
    instantiating multiple orchestrators in the same runtime will corrupt the
    worker pools. This test proves that the system aggressively aborts if
    a duplicate instantiation is attempted.
    """
    with pytest.raises(
        RuntimeError, match="PipelineOrchestrator is a strict Singleton"
    ):
        PipelineOrchestrator(
            raw_txt_dir=tmp_path,
            raw_xml_dir=tmp_path,
            output_dir=tmp_path,
            manifest_dir=tmp_path,
            raw_index_path=tmp_path,
            metadata_path=tmp_path,
        )


def test_xml_superiority_routing(orchestrator: PipelineOrchestrator) -> None:
    """Verifies the dynamic era routing logic for legacy backports.

    Ensures that if an RFC possesses both a legacy TXT source file and a
    modern high-fidelity XML counterpart, the orchestrator successfully
    identifies the XML overlap and explicitly excludes the TXT version from
    the legacy ingestion queue to prevent duplicate extraction boundaries.
    """
    (orchestrator.raw_txt_dir / "rfc1000.txt").touch()
    (orchestrator.raw_txt_dir / "rfc1001.txt").touch()
    (orchestrator.raw_xml_dir / "rfc1000.xml").touch()

    xml_covered = orchestrator._get_xml_covered_rfcs()  # pyright: ignore[reportPrivateUsage]

    assert 1000 in xml_covered, (
        "RFC 1000 has an XML file, it MUST be marked as covered."
    )
    assert 1001 not in xml_covered, "RFC 1001 is TXT only, it MUST NOT be covered."


@patch("ingestion.orchestrator.ProcessPool")
@patch("ingestion.orchestrator.wait")
def test_orchestrator_crash_handlers_and_telemetry(
    mock_wait: MagicMock,
    mock_pool_class: MagicMock,
    orchestrator: PipelineOrchestrator,
) -> None:
    """Validates multiprocessing crash recovery and telemetry ledger fidelity.

    This test uses Python `unittest.mock` components to hijack the Pebble worker
    pool and inject artificial chaos (a standard runtime exception and a hard
    process timeout). It verifies that the orchestrator catches the failures,
    prevents pipeline halting, and accurately records the anomalies into the
    compiled telemetry manifest.
    """
    f_success = MagicMock()
    f_success.result.return_value = (
        "success",
        None,
        {
            "file": "rfc1.txt",
            "status": "success",
            "total_blocks": 5,
            "normative_rules": 0,
            "total_chars": 1000,
            "max_block_chars": 200,
            "min_block_chars": 50,
        },
    )

    f_fail = MagicMock()
    f_fail.result.return_value = (
        "failed",
        "Simulated hard worker crash inside parsing",
        None,
    )

    f_timeout = MagicMock()
    f_timeout.result.side_effect = TimeoutError("Simulated Process Timeout")

    mock_pool_instance = mock_pool_class.return_value.__enter__.return_value
    mock_pool_instance.schedule.side_effect = [f_success, f_fail, f_timeout]

    # Hijack concurrent.futures.wait using the _ variable to absorb unused kwargs
    def mock_wait_resolved(
        fs: set[Future[Any]],
        **_: Any,  # noqa: ANN401
    ) -> tuple[set[Future[Any]], set[Future[Any]]]:
        return set(fs), set()

    mock_wait.side_effect = mock_wait_resolved

    fake_files = [Path("rfc1.txt"), Path("rfc2.txt"), Path("rfc3.txt")]
    success_count, failed_count = orchestrator._run_parallel_ingestion_pool(  # pyright: ignore[reportPrivateUsage]
        target_files=fake_files, source_type="txt", log_prefix="Test"
    )

    assert success_count == 1
    assert failed_count == 2
    assert len(orchestrator.telemetry_manifest) == 3

    statuses = [record["status"] for record in orchestrator.telemetry_manifest]
    assert statuses.count("success") == 1
    assert statuses.count("failed") == 2

    timeout_records = [
        r
        for r in orchestrator.telemetry_manifest
        if "exceeded per-file timeout" in str(r.get("error"))
    ]
    assert len(timeout_records) == 1, (
        "The TimeoutError was not caught and logged properly!"
    )
