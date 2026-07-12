"""In-memory temporal lineage graph for O(1) RFC relationship traversal.

Replaces external graph databases (e.g., Neo4j) to map obsoletions, updates,
and historical lineage directly in Python memory for RAG context injection.
"""

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, TypeAlias

from rfc_atlas.utils.patterns import StaticSingleton

logger = logging.getLogger(__name__)

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONNode: TypeAlias = JSONPrimitive | list["JSONNode"] | dict[str, "JSONNode"]

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_PATH = _PROJECT_ROOT / "data" / "metadata" / "rfc_metadata_lookup.json"


@dataclass
class LineageNode:
    """Represents a single RFC entity within the temporal graph."""

    rfc_number: int
    title: str
    status: str
    obsoletes: set[int] = field(default_factory=set[int])
    obsoleted_by: set[int] = field(default_factory=set[int])
    updates: set[int] = field(default_factory=set[int])
    updated_by: set[int] = field(default_factory=set[int])


class TemporalLineageGraph(StaticSingleton):
    """Singleton graph engine for resolving standard lifecycles dynamically."""

    _graph: ClassVar[dict[int, LineageNode]] = {}
    _is_loaded: ClassVar[bool] = False

    @classmethod
    def _get_or_create_node(cls, rfc_number: int) -> LineageNode:
        """Retrieves an existing node or initializes a new one.

        Args:
            rfc_number (int): Numeric identifier of the target RFC.

        Returns:
            LineageNode: The corresponding graph node.
        """
        if rfc_number not in cls._graph:
            cls._graph[rfc_number] = LineageNode(
                rfc_number=rfc_number, title=f"RFC {rfc_number}", status="UNKNOWN"
            )
        return cls._graph[rfc_number]

    @staticmethod
    def _extract_int_set(raw_list: JSONNode) -> set[int]:
        """Safely extracts a set of integers from an untyped JSON list.

        Args:
            raw_list (JSONNode): Raw JSON node, expected to be a list of integers.

        Returns:
            set[int]: A clean set of validated integers, or an empty set if invalid.
        """
        if isinstance(raw_list, list):
            return {x for x in raw_list if isinstance(x, int)}
        return set()

    @classmethod
    def _populate_nodes(cls, raw_data: dict[str, JSONNode]) -> None:
        """Parses the raw JSON dictionary and hydrates the foundational graph nodes.

        Iterates over the loaded metadata, instantiates LineageNode objects,
        and registers their explicit forward edges (obsoletes, updates).

        Args:
            raw_data (dict[str, JSONNode]): The parsed JSON metadata dictionary.
        """
        for rfc_str, entry in raw_data.items():
            if not isinstance(entry, dict):
                continue
            try:
                rfc_num = int(rfc_str)
            except ValueError:
                continue

            node = cls._get_or_create_node(rfc_num)
            node.title = str(entry.get("title", f"RFC {rfc_num}"))
            node.status = str(entry.get("status", "UNKNOWN"))

            node.obsoletes.update(cls._extract_int_set(entry.get("obsoletes")))
            node.updates.update(cls._extract_int_set(entry.get("updates")))
            node.updated_by.update(cls._extract_int_set(entry.get("updated_by")))

    @classmethod
    def _compute_bidirectional_edges(cls) -> None:
        """Calculates and injects all missing reverse edges into the graph.

        Traverses all explicit forward edges (e.g., Node A obsoletes Node B)
        and explicitly wires the corresponding reverse edges (Node B is
        obsoleted_by Node A) to guarantee O(1) bi-directional traversal.
        """
        for rfc_num, node in cls._graph.items():
            for target in node.obsoletes:
                cls._get_or_create_node(target).obsoleted_by.add(rfc_num)

            for target in node.updates:
                cls._get_or_create_node(target).updated_by.add(rfc_num)

    @classmethod
    def load(cls, filepath: Path | str = DEFAULT_METADATA_PATH) -> None:
        """Parses the raw JSON dictionary and computes bidirectional graph edges.

        Args:
            filepath (Path | str): Path to the compiled metadata lookup cache.
        """
        if cls._is_loaded:
            return

        path = Path(filepath)
        if not path.exists():
            logger.warning(
                "Metadata lookup not found at %s. Graph expansion disabled.", path
            )
            cls._is_loaded = True
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                raw_data: dict[str, JSONNode] = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.critical(
                "Failed to read %s. Graph expansion permanently disabled.",
                path,
                exc_info=True,
            )
            cls._is_loaded = True
            return

        cls._populate_nodes(raw_data)
        cls._compute_bidirectional_edges()

        cls._is_loaded = True
        logger.info(
            "TemporalLineageGraph initialized. Mapped temporal edges for %d RFCs.",
            len(cls._graph),
        )

    @classmethod
    def force_reload(cls, filepath: Path | str = DEFAULT_METADATA_PATH) -> None:
        """Wipes internal graph memory and forces an immediate disk re-read.

        Args:
            filepath (Path | str): Path to the target metadata lookup JSON.
        """
        cls._graph.clear()
        cls._is_loaded = False
        cls.load(filepath)

    @classmethod
    def get_node(cls, rfc_number: int) -> LineageNode | None:
        """O(1) lookup returning the localized graph node for a given RFC integer.

        Args:
            rfc_number (int): Numeric identifier of the target RFC.

        Returns:
            LineageNode | None: The graph node, or None if undiscovered.
        """
        if not cls._is_loaded:
            cls.load()
        return cls._graph.get(rfc_number)

    @classmethod
    def resolve_modern_equivalents(cls, rfc_number: int) -> set[int]:
        """Recursively traverses obsoletion edges to find the active modern standards.

        Performs a Breadth-First Search (BFS) to safely handle complex
        multi-generational obsoletion chains without falling into infinite cyclic traps.

        Args:
            rfc_number (int): The starting RFC identifier.

        Returns:
            set[int]: A set of currently active RFC integers that supersede the input.
        """
        if not cls._is_loaded:
            cls.load()

        node = cls._graph.get(rfc_number)
        if not node:
            return set()

        if not node.obsoleted_by:
            return {rfc_number}

        modern_rfcs: set[int] = set()
        visited: set[int] = {rfc_number}
        queue: deque[int] = deque(node.obsoleted_by)

        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            current_node = cls._graph.get(current_id)
            if not current_node:
                continue

            if not current_node.obsoleted_by:
                modern_rfcs.add(current_id)
            else:
                queue.extend(current_node.obsoleted_by)

        return modern_rfcs

    @classmethod
    def format_lineage_warning(cls, rfc_number: int) -> str | None:
        """Compiles graph edges into a dense textual injection string for the LLM.

        Evaluates the standard's current lifecycle status and crafts a high-visibility
        context window warning if the chunk is obsolete or has pending updates.

        Args:
            rfc_number (int): Numeric identifier of the target RFC.

        Returns:
            str | None: A formatted string containing lineage warnings, or None if
                the standard is fully active and unmodified.
        """
        node = cls.get_node(rfc_number)
        if not node:
            return None

        warnings: list[str] = []

        if node.obsoleted_by:
            modern_ids = cls.resolve_modern_equivalents(rfc_number)
            direct_obs = ", ".join(f"RFC {i}" for i in sorted(node.obsoleted_by))
            active_std = ", ".join(f"RFC {i}" for i in sorted(modern_ids))

            msg = (
                f"[TEMPORAL WARNING: This chunk is from RFC {rfc_number}, "
                f"which is OBSOLETE. It was superseded by {direct_obs}."
            )
            if modern_ids and modern_ids != node.obsoleted_by:
                msg += (
                    " The current active standard(s) resolving this lineage are "
                    f"{active_std}."
                )
            msg += " Evaluate this text with historical caution.]"
            warnings.append(msg)

        elif node.updated_by:
            updates = ", ".join(f"RFC {i}" for i in sorted(node.updated_by))
            warnings.append(
                f"[TEMPORAL NOTICE: This chunk is from RFC {rfc_number}, "
                f"which is active but has been UPDATED by {updates}. "
                "Please consider these later modifications.]"
            )

        if warnings:
            return "\n".join(warnings)

        return None
