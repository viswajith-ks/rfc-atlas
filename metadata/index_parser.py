"""Parser for compiling the global IETF RFC XML index into local metadata lookups."""

import json
import logging
import re
from pathlib import Path

from lxml import etree
from lxml.etree import _Element  # pyright: ignore[reportPrivateUsage]

from metadata.schema import RFCIndexEntryDict, RFCPublicationDateDict
from utils.exceptions import CorpusDependencyError, MalformedIndexXMLError
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

    @staticmethod
    def _clean_doc_id(doc_id: str) -> int:
        """Extracts the numeric identifier from a raw document ID string.

        Args:
            doc_id (str): Raw document identifier (e.g., 'RFC2616').

        Returns:
            int: The isolated numeric component (e.g., 2616), or 0 if invalid.
        """
        if not doc_id:
            return 0

        # Matches the literal string "RFC" followed by one or more digits anywhere
        # in the string. Used to extract the ID from XML index doc-id tags.
        match = re.search(r"RFC(\d+)", doc_id, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _extract_date(entry: _Element) -> RFCPublicationDateDict | None:
        """Extracts and normalizes the publication date from an RFC entry node.

        Args:
            entry (_Element): The parent <rfc-entry> node.

        Returns:
            RFCPublicationDateDict | None:
                Normalized date mapping or None if unparseable.
        """
        date_node = find_child_by_local_name(entry, "date")
        if date_node is None:
            return None

        year_str = get_child_text_by_local_name(date_node, "year")
        if not year_str or not year_str.isdigit():
            return None

        year_val = int(year_str)
        month_val = None
        month_str = get_child_text_by_local_name(date_node, "month")

        if month_str:
            clean_str = month_str.strip().lower()
            for key, num in _FUZZY_MONTH_MAP.items():
                if clean_str.startswith(key):
                    month_val = num
                    break

        return {"year": year_val, "month": month_val}

    @staticmethod
    def _extract_authors(entry: _Element) -> list[str]:
        """Extracts the list of author names from an RFC entry node.

        Args:
            entry (_Element): The parent <rfc-entry> node.

        Returns:
            list[str]: Collection of extracted author names.
        """
        authors: list[str] = []
        for author_node in entry:
            if get_local_name(author_node) == "author":
                name = get_child_text_by_local_name(author_node, "name")
                if name:
                    authors.append(name)
        return authors

    def _extract_doc_relations(self, entry: _Element, tag_name: str) -> list[int]:
        """Extracts a list of related document IDs for a given relational tag.

        Args:
            entry (_Element): The parent <rfc-entry> node.
            tag_name (str):
                The specific XML relation tag (e.g., 'obsoletes', 'updates').

        Returns:
            list[int]: Collection of numeric RFC identifiers.
        """
        relations: list[int] = []
        rel_node = find_child_by_local_name(entry, tag_name)
        if rel_node is not None:
            relations.extend(
                rfc_id
                for doc in rel_node
                if get_local_name(doc) == "doc-id"
                and doc.text
                and (rfc_id := self._clean_doc_id(doc.text))
            )
        return relations

    def _parse_rfc_entry(self, entry: _Element) -> RFCIndexEntryDict | None:
        """Extracts attributes and edge relations from a single rfc-entry XML node.

        Args:
            entry (_Element): The parent <rfc-entry> node.

        Returns:
            RFCIndexEntryDict | None:
                The compiled schema-compliant dictionary, or None if invalid.
        """
        doc_id_text = get_child_text_by_local_name(entry, "doc-id")
        if not doc_id_text:
            return None

        rfc_num = self._clean_doc_id(doc_id_text)
        if rfc_num == 0:
            return None

        title = get_child_text_by_local_name(entry, "title") or "Unknown Title"
        status = get_child_text_by_local_name(entry, "current-status") or "UNKNOWN"
        stream = get_child_text_by_local_name(entry, "stream") or "UNKNOWN"

        return {
            "rfc_number": rfc_num,
            "title": title,
            "published_at": self._extract_date(entry),
            "status": status,
            "stream": stream,
            "authors": self._extract_authors(entry),
            "obsoletes": self._extract_doc_relations(entry, "obsoletes"),
            "updates": self._extract_doc_relations(entry, "updates"),
            "updated_by": self._extract_doc_relations(entry, "updated-by"),
            "protocol_family": None,
        }

    def _process_element(self, elem: _Element) -> None:
        """Processes a single XML node and aggressively frees memory."""
        tag = elem.tag.split("}")[-1] if "}" in str(elem.tag) else str(elem.tag)

        if tag == "rfc-entry":
            entry_data = self._parse_rfc_entry(elem)
            if entry_data is not None:
                self.metadata_dict[str(entry_data["rfc_number"])] = entry_data

            elem.clear()
            parent = elem.getparent()
            if parent is not None:
                while elem.getprevious() is not None:
                    del parent[0]

    def parse(self) -> None:
        """Parses the global RFC index XML and compiles the metadata ledger.

        Raises:
            CorpusDependencyError: If the source XML file does not exist on disk.
            MalformedIndexXMLError: If the XML source text is structurally malformed.
        """
        if not self.xml_path.exists():
            raise CorpusDependencyError(self.xml_path, "RFC Index XML")

        logger.info("Parsing RFC Index XML from %s...", self.xml_path)

        try:
            context = etree.iterparse(
                str(self.xml_path),
                events=("end",),
                huge_tree=True,
            )
            for _, elem in context:
                self._process_element(elem)
        except etree.ParseError as e:
            logger.exception(
                "RFC index XML is structurally malformed: %s", self.xml_path
            )
            raise MalformedIndexXMLError(self.xml_path, e) from e

        self._save()

    def _save(self) -> None:
        """Serializes the compiled metadata dictionary to disk atomically."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.output_path.with_suffix(".json.tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata_dict, f, indent=2)

        Path(tmp_path).replace(self.output_path)
