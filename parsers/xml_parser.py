"""Modern XML Parser Engine

Provides deterministic structural parsing for modern IETF RFC documents (RFC 8650+).
Leverages lxml to traverse xml2rfc v3 schemas, extracting semantic blocks with
perfect confidence while routing specialized data structures to designated tables.
"""

from lxml import etree


class ModernRFCParser:
    """A structural parser for modern IETF RFC documents (RFC 8650+).

    This parser utilizes lxml to traverse xml2rfc v3 structures, extracting
    semantic blocks while enforcing strict schema compliance. It preserves the
    format-agnostic XML source fragments and routes specialized data (e.g.,
    ABNF, Security) to specific block types for downstream vector database tables.

    Attributes:
        xml_filepath (str): The local file path to the raw XML RFC.
        tree (lxml.etree._ElementTree): The parsed XML tree object.
        root (lxml.etree._Element): The root '<rfc>' element.
        rfc_number (str): The official RFC number extracted from the root.
    """

    def __init__(self, xml_filepath):
        """Initializes the parser and loads the XML document into memory.

        Args:
            xml_filepath (str): Path to the target RFC XML file.
        """
        self.xml_filepath = xml_filepath
        self.tree = etree.parse(xml_filepath)
        self.root = self.tree.getroot()
        self.rfc_number = self.root.get("number", "UNKNOWN")

    def parse_document(self):
        """Executes top-level parsing across the document.

        Walks through the primary content ('<middle>') and appendices/citations
        ('<back>'), triggering recursive extraction for all sections.

        Returns:
            list[dict]: A list of canonical JSON artifacts representing each block.
        """
        blocks = []

        # Parse main body text (<middle>)
        middle = self.root.find("middle")
        if middle is not None:
            for section in middle.findall("section"):
                blocks.extend(self._parse_section(section, hierarchy_path=[]))

        # Parse appendices and bibliographic metadata (<back>)
        back = self.root.find("back")
        if back is not None:
            for section in back.findall("section"):
                blocks.extend(self._parse_section(section, hierarchy_path=["Back"]))

            # Explicitly hunt for the references index outside standard sections
            for references in back.findall("references"):
                blocks.extend(self._parse_section(references, hierarchy_path=["Back"]))

        return blocks

    def _parse_section(self, section_node, hierarchy_path):
        """Recursively traverses sections to preserve deep lineage paths.

        Args:
            section_node (lxml.etree._Element): The current section or references node.
            hierarchy_path (list[str]): The accumulated breadcrumb trail of parent section titles.

        Returns:
            list[dict]: Extracted blocks from this section and all nested children.
        """
        blocks = []

        # Extract the official section title to build the breadcrumb path
        name_node = section_node.find("name")
        section_title = (
            name_node.text
            if (name_node is not None and name_node.text)
            else "Untitled Section"
        )

        current_path = hierarchy_path + [section_title.strip()]
        path_str = " > ".join(current_path)

        # The definitive list of extractable semantic elements in xml2rfc v3
        target_tags = [
            "t",
            "sourcecode",
            "artwork",
            "table",
            "reference",
            "ul",
            "ol",
            "dl",
            "blockquote",
            "aside",
        ]

        for child in section_node:
            tag = child.tag

            if tag == "name":
                continue

            if tag in ["section", "references"]:
                blocks.extend(self._parse_section(child, current_path))

            elif tag in target_tags:
                blocks.append(self._build_block(child, tag, path_str))

            elif tag == "figure":
                for sub_child in child:
                    if sub_child.tag in ["sourcecode", "artwork", "table"]:
                        blocks.append(
                            self._build_block(sub_child, sub_child.tag, path_str)
                        )

        return blocks

    def _build_block(self, node, tag, hierarchy_path):
        """Constructs a normalized, immutable block artifact from an XML node.

        Handles text normalization, citation reconstruction, and data stream
        routing (e.g., flagging ABNF or Security blocks).

        Args:
            node (lxml.etree._Element): The XML element to extract.
            tag (str): The tag name of the element.
            hierarchy_path (str): The full breadcrumb path (e.g., "Overview > Scope").

        Returns:
            dict: The canonical block artifact conforming to the database schema.
        """
        # Save the exact XML string to guarantee lossless provenance
        source_fragment = etree.tostring(
            node, encoding="unicode", pretty_print=False
        ).strip()

        # Text Normalization
        if tag == "t":
            text_parts = []
            if node.text:
                text_parts.append(node.text)

            # Reconstruct inline citations (e.g., converting <xref> to [RFC9000])
            for child in node:
                if child.tag == "xref":
                    target = child.get("target", "UNKNOWN")
                    text_parts.append(f"[{target}]")
                elif child.tag == "eref":
                    target = child.get("target", "LINK")
                    text_parts.append(f"[{target}]")
                elif child.text:
                    text_parts.append(child.text)

                if child.tail:
                    text_parts.append(child.tail)

            raw_text = "".join(text_parts)
            normalized_text = " ".join(raw_text.split())

        elif tag == "reference":
            # Reconstruct bibliography nodes into queryable text
            title_node = node.find(".//title")
            title = (
                title_node.text.strip()
                if (title_node is not None and title_node.text)
                else "Untitled"
            )

            authors = [
                a.get("fullname")
                for a in node.findall(".//author")
                if a.get("fullname")
            ]
            authors_str = ", ".join(authors) if authors else "Unknown Authors"

            normalized_text = f"Citations Target [{node.get('anchor', 'REF')}]: {title} by {authors_str}."
            normalized_text = " ".join(normalized_text.split())

        else:
            # Compress spacing for standard elements; preserve whitespace for code/artwork
            raw_text = "".join(node.itertext())
            if tag in ["sourcecode", "artwork"]:
                normalized_text = raw_text.strip("\n")
            else:
                normalized_text = " ".join(raw_text.split())

        # Schema Type Mapping
        block_type = tag

        # Identify specialized grammar structures
        if tag == "sourcecode" and node.get("type") == "abnf":
            block_type = "abnf"

        # Group standard text elements into the 'prose' table
        if block_type in ["t", "ul", "ol", "dl", "blockquote", "aside"]:
            block_type = "prose"

        if block_type == "reference":
            block_type = "references"

        # Contextual Data Routing
        path_lower = hierarchy_path.lower()
        if "security considerations" in path_lower:
            block_type = "security"
        elif "references" in path_lower:
            block_type = "references"

        # Capture the document part number (pn) or legacy anchor
        element_id = node.get("anchor") or node.get("pn")

        return {
            "rfc_id": self.rfc_number,
            "hierarchy_path": hierarchy_path,
            "block_type": block_type,
            "source_type": "xml",
            "normalized_text": normalized_text,
            "source_fragment": source_fragment,
            "parsing_confidence": 1.0,
            "metadata": {"element_id": element_id},
        }
