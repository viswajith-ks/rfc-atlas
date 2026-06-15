"""Modern XML parser for extracting canonical structured blocks from xml2rfc v3 documents."""

import logging
from pathlib import Path
from typing import cast, get_args

from lxml import etree
from lxml.etree import _Element  # pyright: ignore[reportPrivateUsage]

from normalization.schema import (
    XML_TAG_TO_INTERMEDIATE_TYPE_MAP,
    BlockMetadataDict,
    CanonicalBlockDict,
    IntermediateBlockType,
    ReferenceCategory,
    ReferenceMetadataDict,
    SourcecodeFormat,
)
from parsers.base import refine_block_type
from utils.xml_utils import (
    find_child_by_local_name,
    find_children_by_local_name,
    get_local_name,
)

logger = logging.getLogger(__name__)


class ModernRFCParser:
    """Structural parser for xml2rfc v3 compliant RFC documents (RFC 8650+)."""

    _TARGET_TAGS = frozenset(
        ["t", "sourcecode", "artwork", "table", "reference", "ul", "ol", "dl"]
    )

    _INFORMATIVE_TOKENS = frozenset(
        {
            "informative",
            "bibliography",
            "non-normative",
            "other",
            "background",
            "reading",
        }
    )
    _NORMATIVE_TOKENS = frozenset(
        {
            "normative",
            "core",
            "required",
            "requirement",
            "standards",
            "specifications",
            "mandatory",
        }
    )

    def __init__(self, xml_filepath: Path) -> None:
        """Initializes the parser and loads the XML document into memory.

        Args:
            xml_filepath (Path): Path to the target RFC XML file.

        Raises:
            ValueError: If a valid numeric RFC ID cannot be extracted from the root element.
            etree.ParseError: If the target XML stream is structurally malformed.
        """
        self.xml_filepath = xml_filepath

        try:
            self.tree = etree.parse(self.xml_filepath)
            self.root = self.tree.getroot()
        except etree.ParseError as e:
            logger.error(
                f"CRITICAL ERROR: Modern XML Ingestion Parsing Failed for document: {self.xml_filepath}. Error: {e}"
            )
            raise

        raw_id = self.root.get("number")
        if not raw_id or not raw_id.isdigit():
            raise ValueError(
                f"Missing or invalid numeric RFC ID in XML root: {xml_filepath}"
            )

        self.rfc_id = int(raw_id)

    def parse_document(self) -> list[CanonicalBlockDict]:
        """Executes top-level parsing across the front, middle, and back sections of the document.

        Returns:
            list[CanonicalBlockDict]: A flat list of canonical intermediate block dictionaries.
        """
        blocks: list[CanonicalBlockDict] = []

        front = find_child_by_local_name(self.root, "front")
        if front is not None:
            for abstract in find_children_by_local_name(front, "abstract"):
                blocks.extend(self._parse_section(abstract, hierarchy_path=["Preface"]))

            for note in find_children_by_local_name(front, "note"):
                title = note.get("title", "Note")
                blocks.extend(
                    self._parse_section(note, hierarchy_path=["Preface", title])
                )

            for boilerplate in find_children_by_local_name(front, "boilerplate"):
                blocks.extend(
                    self._parse_section(
                        boilerplate, hierarchy_path=["Preface", "Boilerplate"]
                    )
                )

        middle = find_child_by_local_name(self.root, "middle")
        if middle is not None:
            for section in find_children_by_local_name(middle, "section"):
                blocks.extend(self._parse_section(section, hierarchy_path=[]))

        back = find_child_by_local_name(self.root, "back")
        if back is not None:
            for section in find_children_by_local_name(back, "section"):
                blocks.extend(self._parse_section(section, hierarchy_path=["Back"]))

            for references in find_children_by_local_name(back, "references"):
                blocks.extend(self._parse_section(references, hierarchy_path=["Back"]))

        return blocks

    def _parse_section(
        self,
        section_node: _Element,
        hierarchy_path: list[str],
        parent_section_number: str | None = None,
    ) -> list[CanonicalBlockDict]:
        """Recursively traverses sections and structural containers to preserve block lineage.

        Args:
            section_node (_Element): The current section or structural container node.
            hierarchy_path (list[str]): Accumulated list of parent section titles.
            parent_section_number (str | None): Tracked section number passed from parent nodes.

        Returns:
            list[CanonicalBlockDict]: Extracted blocks from this container and all nested children.
        """
        blocks: list[CanonicalBlockDict] = []
        node_tag = get_local_name(section_node)

        if node_tag in ["blockquote", "aside"]:
            current_path = hierarchy_path + [node_tag.capitalize()]
            path_str = " > ".join(current_path)
            section_number = parent_section_number
        else:
            pn = section_node.get("pn")
            section_number = parent_section_number

            if pn and pn.startswith("section-"):
                sec_num_part = pn[len("section-") :]
                if sec_num_part.startswith("appendix-"):
                    section_number = sec_num_part[len("appendix-") :].upper()
                else:
                    section_number = sec_num_part

            name_node = find_child_by_local_name(section_node, "name")
            if name_node is not None:
                raw_title = "".join(str(t) for t in name_node.itertext())
                section_title = " ".join(raw_title.split()) or "Untitled Section"
            else:
                section_title = "Untitled Section"

            current_path = hierarchy_path + [section_title.strip()]
            path_str = " > ".join(current_path)

        for child in section_node:
            tag = get_local_name(child)

            if tag == "name":
                continue

            if tag in ["section", "references", "blockquote", "aside"]:
                blocks.extend(self._parse_section(child, current_path, section_number))

            elif tag in self._TARGET_TAGS:
                blocks.append(self._build_block(child, tag, path_str, section_number))

            elif tag == "figure":
                figure_title = ""
                figure_name_node = find_child_by_local_name(child, "name")

                if figure_name_node is not None:
                    raw_title = "".join(str(t) for t in figure_name_node.itertext())
                    figure_title = " ".join(raw_title.split()).strip()

                for sub_child in child:
                    sub_tag = get_local_name(sub_child)

                    if sub_tag in [
                        "sourcecode",
                        "artwork",
                        "table",
                        "t",
                        "ul",
                        "ol",
                        "dl",
                    ]:
                        block = self._build_block(
                            sub_child, sub_tag, path_str, section_number
                        )

                        if figure_title:
                            block["normalized_text"] = (
                                f"[{figure_title}] \n{block['normalized_text']}"
                            )

                        blocks.append(block)

        return blocks

    def _extract_inline_text(self, elem: _Element) -> str:
        """Recursively extracts and formats text from an XML element and its children.

        Args:
            elem (_Element): The XML element node from which to extract text.

        Returns:
            str: The raw, concatenated string containing all text fragments and
                formatted reference tokens, preserving internal spacing.
        """
        parts: list[str] = []

        if elem.text:
            parts.append(elem.text)

        for child in elem:
            child_tag = get_local_name(child)
            child_content = self._extract_inline_text(child)

            if child_tag in ("xref", "eref"):
                default_target = "UNKNOWN" if child_tag == "xref" else "LINK"
                target = child.get("target", default_target)

                if child_content.strip():
                    parts.append(f"{child_content} [{target}]")
                else:
                    parts.append(f"[{target}]")
            else:
                parts.append(child_content)

            if child.tail:
                parts.append(child.tail)

        return "".join(parts)

    def _normalize_inline_element(self, node: _Element) -> str:
        """Extracts plain text from an inline element while formatting cross-reference tokens.

        Args:
            node (_Element): The target XML element node containing mixed-content text.

        Returns:
            str: Sanitized plain-text representation of the element.
        """
        raw_text = self._extract_inline_text(node)
        return " ".join(raw_text.split())

    def _extract_bibliographic_metadata(
        self, node: _Element, category: ReferenceCategory
    ) -> ReferenceMetadataDict:
        """Extracts deep bibliographic structural metadata from a reference specification tag.

        Args:
            node (_Element): The XML element node matching <reference> or <referencegroup>.
            category (ReferenceCategory): Specifier denoting the citation context category.

        Returns:
            ReferenceMetadataDict: Structured citation data optimized for relational mapping.
        """
        anchor = node.get("anchor", "UNKNOWN")
        target_url = node.get("target")

        front_node = find_child_by_local_name(node, "front")
        title_node = (
            find_child_by_local_name(front_node, "title")
            if front_node is not None
            else None
        )
        title = (
            "".join(map(str, title_node.itertext())).strip()
            if title_node is not None
            else None
        )

        doi = None
        series_name = None
        series_value = None

        for info in find_children_by_local_name(node, "seriesInfo"):
            name = info.get("name", "")
            value = info.get("value")

            if name.upper() == "DOI":
                doi = value
            elif name.upper() in ("RFC", "INTERNET-DRAFT", "BCP", "STD", "FYI"):
                series_name = name
                series_value = value

        ref_data: ReferenceMetadataDict = {
            "anchor": anchor,
            "category": category,
            "title": title,
        }

        if target_url is not None:
            ref_data["target_url"] = target_url

        if doi is not None:
            ref_data["doi"] = doi

        if series_name is not None:
            ref_data["series_name"] = series_name

        if series_value is not None:
            ref_data["series_value"] = series_value

        return ref_data

    def _build_block(
        self,
        node: _Element,
        tag: str,
        hierarchy_path: str,
        section_number: str | None = None,
    ) -> CanonicalBlockDict:
        """Constructs a normalized intermediate block dictionary from an XML element.

        Args:
            node (_Element): The target XML node to extract data from.
            tag (str): Local XML tag name of the target node.
            hierarchy_path (str): Full hierarchy breadcrumb text path string.
            section_number (str | None): Section structural location index.

        Returns:
            CanonicalBlockDict: Structured intermediate block artifact matching schema boundaries.
        """
        source_fragment = etree.tostring(
            node, encoding="unicode", pretty_print=False
        ).strip()

        ref_meta = None

        path_lower = hierarchy_path.lower()

        if tag == "t":
            normalized_text = self._normalize_inline_element(node)

        elif tag in ["ul", "ol", "dl"]:
            items: list[str] = []

            for child in node:
                child_tag = get_local_name(child)
                if child_tag == "li":
                    item_text = self._normalize_inline_element(child)
                    if item_text:
                        items.append(f"* {item_text}")
                elif child_tag == "dt":
                    item_text = self._normalize_inline_element(child)
                    if item_text:
                        items.append(f"; {item_text}")
                elif child_tag == "dd":
                    item_text = self._normalize_inline_element(child)
                    if item_text:
                        items.append(f": {item_text}")

            normalized_text = "\n".join(items)

        elif tag == "reference":
            title_nodes: list[_Element] = cast(
                list[_Element], node.xpath(".//*[local-name()='title']")
            )
            title = "Untitled"

            if title_nodes:
                raw_title = "".join(str(t) for t in title_nodes[0].itertext())
                cleaned_title = raw_title.strip()
                if cleaned_title:
                    title = cleaned_title

            author_nodes: list[_Element] = cast(
                list[_Element], node.xpath(".//*[local-name()='author']")
            )
            authors: list[str] = [
                name for a in author_nodes if (name := a.get("fullname")) is not None
            ]
            authors_str = ", ".join(authors) if authors else "Unknown Authors"

            normalized_text = f"Citations Target [{node.get('anchor', 'REF')}]: {title} by {authors_str}."
            normalized_text = " ".join(normalized_text.split())

            if any(tok in path_lower for tok in self._INFORMATIVE_TOKENS):
                category = "Informative"
            elif any(tok in path_lower for tok in self._NORMATIVE_TOKENS):
                category = "Normative"
            else:
                category = "Informative"

            ref_meta = self._extract_bibliographic_metadata(node, category)

        else:
            raw_text = "".join(str(t) for t in node.itertext())
            if tag in ["sourcecode", "artwork"]:
                normalized_text = raw_text.strip("\n")
            else:
                normalized_text = " ".join(raw_text.split())

        raw_lang = node.get("type")
        sourcecode_type = None

        clean_lang = raw_lang.lower() if raw_lang else None

        if (
            tag == "sourcecode"
            and clean_lang
            and clean_lang in get_args(SourcecodeFormat)
        ):
            sourcecode_type = cast(SourcecodeFormat, clean_lang)

        if tag == "sourcecode" and clean_lang == "abnf":
            block_type: IntermediateBlockType = "abnf"
        else:
            block_type = XML_TAG_TO_INTERMEDIATE_TYPE_MAP.get(tag, "prose")

        block_type = refine_block_type(block_type, path_lower)

        element_id = node.get("anchor") or node.get("pn")

        metadata_data: BlockMetadataDict = {
            "element_id": element_id,
            "normative_keywords": [],
        }

        if section_number is not None:
            metadata_data["section_number"] = section_number

        if ref_meta is not None:
            metadata_data["reference_metadata"] = ref_meta

        block_data: CanonicalBlockDict = {
            "rfc_id": self.rfc_id,
            "hierarchy_path": hierarchy_path,
            "block_type": block_type,
            "source_type": "xml",
            "normalized_text": normalized_text,
            "source_fragment": source_fragment,
            "parsing_confidence": 1.0,
            "metadata": metadata_data,
        }

        if sourcecode_type is not None:
            block_data["sourcecode_type"] = sourcecode_type

        return block_data
