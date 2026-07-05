import json
from pathlib import Path
from typing import Any

import pytest

from rfc_atlas.graph.lineage import TemporalLineageGraph
from rfc_atlas.utils.exceptions import SingletonViolationError

# Force resetting the singleton before every test
pytestmark = pytest.mark.usefixtures("_reset_graph")


@pytest.fixture
def _reset_graph() -> None:  # pyright: ignore[reportUnusedFunction]
    TemporalLineageGraph._is_instantiated = False  # pyright: ignore[reportPrivateUsage]
    TemporalLineageGraph._graph.clear()  # pyright: ignore[reportPrivateUsage]
    TemporalLineageGraph._is_loaded = False  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def mock_metadata_file(tmp_path: Path) -> Path:
    metadata_path = tmp_path / "rfc_metadata_lookup.json"

    mock_data: dict[str, dict[str, Any]] = {
        "1000": {
            "title": "Legacy Protocol",
            "status": "UNKNOWN",
            "obsoletes": [],
            "updates": [],
            "updated_by": [],
        },
        "2000": {
            "title": "Bridge Protocol",
            "status": "UNKNOWN",
            "obsoletes": [1000],
            "updates": [],
            "updated_by": [],
        },
        "3000": {
            "title": "Modern Protocol A",
            "status": "UNKNOWN",
            "obsoletes": [2000],
            "updates": [4000],
            "updated_by": [],
        },
        "3001": {
            "title": "Modern Protocol B",
            "status": "UNKNOWN",
            "obsoletes": [2000],
            "updates": [],
            "updated_by": [],
        },
        "4000": {
            "title": "Active Protocol",
            "status": "UNKNOWN",
            "obsoletes": [],
            "updates": [],
            "updated_by": [3000],
        },
        "5000": {
            "title": "Cycle Node A",
            "status": "UNKNOWN",
            "obsoletes": [5001],
            "updates": [],
            "updated_by": [],
        },
        "5001": {
            "title": "Cycle Node B",
            "status": "UNKNOWN",
            "obsoletes": [5000],
            "updates": [],
            "updated_by": [],
        },
    }

    metadata_path.write_text(json.dumps(mock_data), encoding="utf-8")
    return metadata_path


def test_lineage_graph_singleton_physics() -> None:
    _ = TemporalLineageGraph()
    with pytest.raises(SingletonViolationError):
        _ = TemporalLineageGraph()


def test_bidirectional_edge_computation(mock_metadata_file: Path) -> None:
    TemporalLineageGraph.force_reload(mock_metadata_file)

    node_1000 = TemporalLineageGraph.get_node(1000)
    assert node_1000 is not None

    # The JSON only said 2000 obsoletes 1000.
    # The engine MUST have dynamically calculated that 1000 is obsoleted_by 2000.
    assert 2000 in node_1000.obsoleted_by

    node_4000 = TemporalLineageGraph.get_node(4000)
    assert node_4000 is not None
    # JSON says 3000 updates 4000, and 4000 updated_by 3000. It should handle it seamlessly.
    assert 3000 in node_4000.updated_by


def test_bfs_modern_equivalent_resolution(mock_metadata_file: Path) -> None:
    TemporalLineageGraph.force_reload(mock_metadata_file)

    # 1. Ask for an active node (Should return itself)
    assert TemporalLineageGraph.resolve_modern_equivalents(3000) == {3000}

    # 2. Ask for a multi-generational legacy node (1000 -> 2000 -> [3000, 3001])
    modern_ids = TemporalLineageGraph.resolve_modern_equivalents(1000)
    assert modern_ids == {3000, 3001}


def test_bfs_cyclic_dependency_protection(mock_metadata_file: Path) -> None:
    TemporalLineageGraph.force_reload(mock_metadata_file)

    # 5000 and 5001 obsolete each other in the mock.
    # If the BFS lacks cycle detection, this call will hang the test suite forever.
    modern_ids = TemporalLineageGraph.resolve_modern_equivalents(5000)

    # Because both are superseded by each other and neither is "active",
    # the resolved set of modern endpoints is correctly empty.
    assert modern_ids == set()


def test_format_lineage_warning(mock_metadata_file: Path) -> None:
    TemporalLineageGraph.force_reload(mock_metadata_file)

    # 1. Obsolete Warning (Multi-generation)
    warn_1000 = TemporalLineageGraph.format_lineage_warning(1000)
    assert warn_1000 is not None
    assert "OBSOLETE" in warn_1000
    assert "superseded by RFC 2000" in warn_1000
    assert "resolving this lineage are RFC 3000, RFC 3001" in warn_1000

    # 2. Updated Warning
    warn_4000 = TemporalLineageGraph.format_lineage_warning(4000)
    assert warn_4000 is not None
    assert "UPDATED by RFC 3000" in warn_4000

    # 3. Active Standard (No warning required)
    assert TemporalLineageGraph.format_lineage_warning(3000) is None


def test_missing_or_corrupted_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "does_not_exist.json"

    TemporalLineageGraph.force_reload(missing_file)
    assert TemporalLineageGraph.get_node(1234) is None
    assert TemporalLineageGraph.resolve_modern_equivalents(1234) == set()

    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("THIS IS NOT JSON", encoding="utf-8")

    TemporalLineageGraph.force_reload(corrupt_file)
    assert TemporalLineageGraph.get_node(1234) is None
