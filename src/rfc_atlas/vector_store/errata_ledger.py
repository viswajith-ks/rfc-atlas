"""In-memory singleton ledger for O(1) IETF Errata lookups.

Bridging the data contract gap between LanceDB integer IDs and the
official IETF string-based ("RFC1234") errata manifest.
"""

import json
import logging
from pathlib import Path
from typing import ClassVar, TypeAlias

from rfc_atlas.utils.patterns import StaticSingleton

logger = logging.getLogger(__name__)

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONNode: TypeAlias = JSONPrimitive | list["JSONNode"] | dict[str, "JSONNode"]

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ERRATA_PATH = _PROJECT_ROOT / "data" / "metadata" / "errata.json"


class ErrataLedger(StaticSingleton):
    """Singleton ledger that loads and normalizes IETF errata into memory."""

    _ledger: ClassVar[dict[int, list[dict[str, JSONNode]]]] = {}
    _is_loaded: ClassVar[bool] = False

    @classmethod
    def _process_entry(cls, entry: dict[str, JSONNode]) -> None:
        """Processes a single errata entry and adds it to the ledger if valid.

        Args:
            entry (dict[str, JSONNode]): Raw JSON dictionary representing an erratum.
        """
        doc_id = entry.get("doc-id", "")

        if not (isinstance(doc_id, str) and doc_id.upper().startswith("RFC")):
            return

        try:
            rfc_num = int(doc_id[3:].strip())
        except ValueError:
            return

        status = entry.get("errata_status_code", "")

        if isinstance(status, str) and status in {
            "Verified",
            "Reported",
            "Held for Document Update",
        }:
            if rfc_num not in cls._ledger:
                cls._ledger[rfc_num] = []
            cls._ledger[rfc_num].append(entry)

    @classmethod
    def load(cls, filepath: Path | str = DEFAULT_ERRATA_PATH) -> None:
        """Parses the raw JSON array and groups errata by integer RFC number.

        Args:
            filepath (Path | str): Path to the target errata JSON file. Defaults to
                the standard data directory path.
        """
        if cls._is_loaded:
            return

        path = Path(filepath)
        if not path.exists():
            logger.warning(
                "Errata file not found at %s. Errata injection disabled.", path
            )
            cls._is_loaded = True
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                raw_data: JSONNode = json.load(f)
        except json.JSONDecodeError:
            logger.critical(
                "Syntax corruption in %s. Errata injection permanently disabled.",
                path,
                exc_info=True,
            )
            cls._is_loaded = True
            return
        except OSError:
            logger.critical(
                "I/O failure reading %s. Errata injection permanently disabled.",
                path,
                exc_info=True,
            )
            cls._is_loaded = True
            return

        if isinstance(raw_data, list):
            for entry in raw_data:
                if isinstance(entry, dict):
                    cls._process_entry(entry)

        cls._is_loaded = True
        logger.info(
            "ErrataLedger initialized. Tracking corrections for %d RFCs.",
            len(cls._ledger),
        )

    @classmethod
    def force_reload(cls, filepath: Path | str = DEFAULT_ERRATA_PATH) -> None:
        """Wipes internal ledger memory and forces an immediate disk re-read.

        Args:
            filepath (Path | str): Path to the target errata JSON file.
        """
        cls._ledger.clear()
        cls._is_loaded = False
        cls.load(filepath)

    @classmethod
    def get_errata(cls, rfc_number: int) -> list[dict[str, JSONNode]]:
        """O(1) lookup returning all active errata for a given RFC integer.

        Args:
            rfc_number (int): Numeric identifier of the target RFC.

        Returns:
            list[dict[str, JSONNode]]: A list of errata dictionary records.
        """
        if not cls._is_loaded:
            cls.load()
        return cls._ledger.get(rfc_number, [])

    @classmethod
    def format_errata_for_llm(cls, rfc_number: int) -> str | None:
        """Compiles errata into a dense textual injection string for the LLM.

        Args:
            rfc_number (int): Numeric identifier of the target RFC.

        Returns:
            str | None: A formatted string containing errata details, or None if
                no errata exist.
        """
        errata_list = cls.get_errata(rfc_number)
        if not errata_list:
            return None

        output = [f"\n[CRITICAL ERRATA DETECTED FOR RFC {rfc_number}]"]

        for e in errata_list:
            section = e.get("section", "Unknown")
            err_type = e.get("errata_type_code", "Technical")
            status = e.get("errata_status_code")
            orig = e.get("orig_text", "")
            corr = e.get("correct_text", "")

            output.extend([
                f"- Section {section}: {err_type} Error ({status})",
                f"  ORIGINAL TEXT:\n  {orig}",
                f"  CORRECTED TEXT:\n  {corr}\n",
            ])

        return "\n".join(output)
