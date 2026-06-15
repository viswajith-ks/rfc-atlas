"""Parser for compiling the global IETF RFC XML index into local metadata lookups."""

import json
import logging
import os
import re
from pathlib import Path

from lxml import etree
from lxml.etree import _Element  # pyright: ignore[reportPrivateUsage]

from metadata.schema import RFCIndexEntryDict, RFCPublicationDateDict
from utils.xml_utils import (
    find_child_by_local_name,
    get_child_text_by_local_name,
    get_local_name,
)

logger = logging.getLogger(__name__)

_FUZZY_MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
    "spring": 4,
    "summer": 7,
    "fall": 10,
    "autumn": 10,
    "winter": 1,
}


class RFCIndexParser:
    """Parses rfc-index.xml into a normalized structural JSON lookup table."""

    def __init__(self, xml_path: Path, output_path: Path) -> None:
        """Initializes the index parser with source and destination targets.

        Args:
            xml_path (Path): Path to the raw rfc-index.xml source file.
            output_path (Path): Target path for the output JSON lookup file.
        """
        self.xml_path = xml_path
        self.output_path = output_path
        self.metadata_dict: dict[str, RFCIndexEntryDict] = {}

    def _clean_doc_id(self, doc_id: str) -> int:
        """Extracts the numeric identifier from a raw document ID string.

        Args:
            doc_id (str): Raw document identifier (e.g., 'RFC2616').

        Returns:
            int: The isolated numeric component (e.g., 2616), or 0 if invalid.
        """
        if not doc_id:
            return 0

        # Matches the literal string "RFC" followed by one or more digits anywhere in the string.
        # Used to extract the ID from XML index doc-id tags (e.g., "RFC2616").
        match = re.search(r"RFC(\d+)", doc_id, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    def _parse_rfc_entry(self, entry: _Element) -> "RFCIndexEntryDict | None":
        """Extracts standard attributes and edge relations from a single rfc-entry XML node.

        Args:
            entry: The lxml etree element representing the <rfc-entry>.

        Returns:
            RFCIndexEntryDict: The compiled schema-compliant dictionary, or None if invalid.
        """
        doc_id_text = get_child_text_by_local_name(entry, "doc-id")
        if not doc_id_text:
            return None

        rfc_num = self._clean_doc_id(doc_id_text)
        if rfc_num == 0:
            return None

        title = get_child_text_by_local_name(entry, "title") or "Unknown Title"
        status = get_child_text_by_local_name(entry, "current-status") or "UNKNOWN"
        stream = get_child_text_by_local_name(entry, "stream") or "IETF"

        published_at: RFCPublicationDateDict | None = None
        date_node = find_child_by_local_name(entry, "date")

        if date_node is not None:
            month_str = get_child_text_by_local_name(date_node, "month")
            year_str = get_child_text_by_local_name(date_node, "year")

            if year_str and year_str.isdigit():
                year_val = int(year_str)
                month_val = None

                if month_str:
                    clean_str = month_str.strip().lower()
                    for key, num in _FUZZY_MONTH_MAP.items():
                        if key in clean_str:
                            month_val = num
                            break

                published_at = {
                    "year": year_val,
                    "month": month_val,
                }

        authors: list[str] = []
        for author_node in entry:
            if get_local_name(author_node) == "author":
                name = get_child_text_by_local_name(author_node, "name")
                if name:
                    authors.append(name)

        obsoletes: list[int] = []
        obs_node = find_child_by_local_name(entry, "obsoletes")
        if obs_node is not None:
            for doc in obs_node:
                if get_local_name(doc) == "doc-id" and doc.text:
                    obsoletes.append(self._clean_doc_id(doc.text))

        updates: list[int] = []
        up_node = find_child_by_local_name(entry, "updates")
        if up_node is not None:
            for doc in up_node:
                if get_local_name(doc) == "doc-id" and doc.text:
                    updates.append(self._clean_doc_id(doc.text))

        updated_by: list[int] = []
        upb_node = find_child_by_local_name(entry, "updated-by")
        if upb_node is not None:
            for doc in upb_node:
                if get_local_name(doc) == "doc-id" and doc.text:
                    updated_by.append(self._clean_doc_id(doc.text))

        return {
            "rfc_number": rfc_num,
            "title": title,
            "published_at": published_at,
            "status": status,
            "stream": stream,
            "authors": authors,
            "obsoletes": obsoletes,
            "updates": updates,
            "updated_by": updated_by,
            "protocol_family": None,
        }

    def parse(self) -> None:
        """Parses the global RFC index XML and compiles the metadata ledger."""
        logger.info(f"Parsing RFC Index XML from {self.xml_path}...")

        context = etree.iterparse(
            str(self.xml_path),
            events=("end",),
            tag="rfc-entry",
            huge_tree=True,
        )

        for _, elem in context:
            entry_data = self._parse_rfc_entry(elem)
            if entry_data is not None:
                self.metadata_dict[str(entry_data["rfc_number"])] = entry_data

            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        self._save()

    def _save(self) -> None:
        """Serializes the compiled metadata dictionary to disk atomically."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.output_path.with_suffix(".json.tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata_dict, f, indent=2)

        os.replace(tmp_path, self.output_path)
