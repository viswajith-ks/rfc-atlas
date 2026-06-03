"""RFC Index Parser Engine

This module provides the ingestion logic for the canonical IETF rfc-index.xml file.
It extracts ground-truth temporal and relational context (such as publication dates,
authors, and graph edges like obsoletes/updates) into a highly optimized JSON lookup
dictionary for downstream assembly.
"""

import json
import os
import re
import xml.etree.ElementTree as ET


class RFCIndexParser:
    """Parses the canonical IETF rfc-index.xml file into a fast JSON lookup dictionary.
    Namespace-agnostic to handle historical variations in the IETF schema.
    """

    def __init__(self, xml_path: str, output_path: str):
        """Initializes the index parser.

        Args:
            xml_path (str): The file path to the raw rfc-index.xml file.
            output_path (str): The destination file path for the compiled JSON dictionary.
        """
        self.xml_path = xml_path
        self.output_path = output_path
        self.metadata_dict = {}

    def _clean_doc_id(self, doc_id: str) -> int:
        """Converts 'RFC0001' or 'RFC2616' into integer 1 or 2616."""
        if not doc_id:
            return 0
        match = re.search(r"\d+", doc_id)
        return int(match.group()) if match else 0

    def parse(self):
        """Executes the XML parsing sequence.

        Reads the namespace-agnostic elements, extracts core publication metadata and
        relational graph edges, and loads them into the internal state dictionary.
        """
        print(f"Parsing RFC Index XML from {self.xml_path}...")

        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"Missing {self.xml_path}")

        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"XML Parsing Error: {e}")
            return

        # Helpers to bypass XML Namespaces (e.g., {http://...}rfc-entry)
        def find_child(parent, tag_suffix):
            for child in parent:
                if child.tag.endswith(tag_suffix):
                    return child
            return None

        def get_text(parent, tag_suffix):
            child = find_child(parent, tag_suffix)
            return child.text.strip() if child is not None and child.text else None

        # Find all entry nodes
        entries = [elem for elem in root if elem.tag.endswith("rfc-entry")]

        for entry in entries:
            doc_id_text = get_text(entry, "doc-id")
            if not doc_id_text:
                continue

            rfc_num = self._clean_doc_id(doc_id_text)
            if rfc_num == 0:
                continue

            # Extract basic metadata
            title = get_text(entry, "title") or "Unknown Title"
            status = get_text(entry, "current-status") or "UNKNOWN"
            stream = get_text(entry, "stream") or "IETF"

            # Extract date
            published_at = None
            date_node = find_child(entry, "date")
            if date_node is not None:
                month = get_text(date_node, "month") or "January"
                year = get_text(date_node, "year")
                if year:
                    published_at = f"{month} {year}"

            # Extract authors
            authors = []
            for author_node in entry:
                if author_node.tag.endswith("author"):
                    name = get_text(author_node, "name")
                    if name:
                        authors.append(name)

            # Extract graph edges (Obsoletes / Updates)
            obsoletes = []
            obs_node = find_child(entry, "obsoletes")
            if obs_node is not None:
                for doc in obs_node:
                    if doc.tag.endswith("doc-id") and doc.text:
                        obsoletes.append(self._clean_doc_id(doc.text))

            updates = []
            up_node = find_child(entry, "updates")
            if up_node is not None:
                for doc in up_node:
                    if doc.tag.endswith("doc-id") and doc.text:
                        updates.append(self._clean_doc_id(doc.text))

            updated_by = []
            upb_node = find_child(entry, "updated-by")
            if upb_node is not None:
                for doc in upb_node:
                    if doc.tag.endswith("doc-id") and doc.text:
                        updated_by.append(self._clean_doc_id(doc.text))

            # Compile
            self.metadata_dict[str(rfc_num)] = {
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

        self._save()

    def _save(self):
        """Serializes the internal metadata dictionary to a JSON file.
        Creates parent directories if they do not exist.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_dict, f, indent=2)
        print(f"Successfully compiled metadata for {len(self.metadata_dict)} RFCs.")
        print(f"Saved lookup table to {self.output_path}")
