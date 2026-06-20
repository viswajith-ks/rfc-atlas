from collections.abc import Generator
from concurrent.futures import Future, TimeoutError
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import ingestion.orchestrator
from ingestion.orchestrator import PipelineConfig, PipelineOrchestrator


@pytest.fixture
def orchestrator(tmp_path: Path) -> Generator[PipelineOrchestrator, None, None]:
    PipelineOrchestrator.reset_state()

    txt_dir = tmp_path / "txt"
    xml_dir = tmp_path / "xml"
    txt_dir.mkdir()
    xml_dir.mkdir()

    meta_path = tmp_path / "meta.json"
    meta_path.write_text("{}", encoding="utf-8")

    config = PipelineConfig(
        raw_txt_dir=txt_dir,
        raw_xml_dir=xml_dir,
        output_dir=tmp_path / "out",
        manifest_dir=tmp_path / "manifest",
        raw_index_path=tmp_path / "index.xml",
        metadata_path=meta_path,
    )

    orch = PipelineOrchestrator(config)

    yield orch

    PipelineOrchestrator.reset_state()
    ingestion.orchestrator._worker_tree_builder = None  # pyright: ignore[reportPrivateUsage]


def test_singleton_lock(tmp_path: Path, orchestrator: PipelineOrchestrator) -> None:
    config = PipelineConfig(
        raw_txt_dir=tmp_path,
        raw_xml_dir=tmp_path,
        output_dir=tmp_path,
        manifest_dir=tmp_path,
        raw_index_path=tmp_path,
        metadata_path=tmp_path,
    )

    with pytest.raises(
        RuntimeError, match="PipelineOrchestrator is a strict Singleton"
    ):
        PipelineOrchestrator(config)


def test_xml_superiority_routing(orchestrator: PipelineOrchestrator) -> None:
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
