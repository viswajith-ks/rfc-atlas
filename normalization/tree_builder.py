"""Tree builder engine for assembling flat intermediate blocks into nested canonical RFC documents."""

import hashlib
import json
import logging
from pathlib import Path
from typing import get_args

from metadata.schema import RFCIndexEntryDict, RFCPublicationDate
from normalization.schema import (
    INTERMEDIATE_TO_FINAL_TYPE_MAP,
    Block,
    BlockType,
    CanonicalBlockDict,
    NormalizedRFC,
    NormativeKeyword,
    NormativeStatement,
    RFCMetadata,
    Section,
    SourceType,
)

logger = logging.getLogger(__name__)


class CanonicalTreeBuilder:
    """Assembles flat text or XML block dictionaries into structured Pydantic NormalizedRFC documents."""

    def __init__(self, metadata_lookup_path: Path) -> None:
        """Initializes the tree builder by loading global metadata index lookups.

        Args:
            metadata_lookup_path (Path): Path to the compiled metadata JSON file.

        Raises:
            FileNotFoundError: If the metadata lookup path does not exist on disk.
        """
        lookup_path = metadata_lookup_path
        if not lookup_path.exists():
            raise FileNotFoundError(f"Missing metadata index at {lookup_path}")

        with lookup_path.open(encoding="utf-8") as f:
            self.metadata_lookup: dict[str, RFCIndexEntryDict] = json.load(f)

        self.valid_keywords = frozenset(get_args(NormativeKeyword))

    @staticmethod
    def _parse_section_path(
        h_path: str, section_number: str | None, rfc_number: int
    ) -> tuple[str, str, str]:
        """Parses a hierarchical breadcrumb path string into distinct identification fields.

        Args:
            h_path (str): The raw section hierarchy breadcrumb text path string.
            section_number (str | None): Optional section identifier extracted by the parser.
            rfc_number (int): Numeric identifier of the target RFC, used to generate
                deterministic hashes for unnumbered or unknown sections.

        Returns:
            tuple[str, str, str]: A 3-tuple containing:
                - The extracted section identifier (e.g., "1.1" or "unknown").
                - A block-safe section token (e.g., "sec1.1" or "secunknown-<hash>").
                - The cleaned section title (e.g., "Introduction").
        """
        if h_path.lower() in ("document root", "preface"):
            return "preface", "preface", h_path

        path_parts = h_path.split(" > ")
        last_part = path_parts[-1]

        if section_number:
            sec_id = section_number.strip(".")
            return sec_id, f"sec{sec_id}", last_part.strip()

        id_split = last_part.split(" ", 1)
        is_numbered = len(id_split) > 1 and any(c.isdigit() for c in id_split[0])

        if is_numbered:
            section_id = id_split[0]
            sec_token = f"sec{id_split[0].strip('.')}"
            title = id_split[1]
            return section_id, sec_token, title.strip()

        stable_hash = hashlib.md5(f"{rfc_number}:{h_path}".encode()).hexdigest()[:8]
        return "unknown", f"secunknown-{stable_hash}", last_part.strip()

    def build_tree(
        self,
        rfc_number: int,
        flat_blocks: list[CanonicalBlockDict],
        source_type: SourceType,
    ) -> NormalizedRFC:
        """Aggregates a flat collection of intermediate blocks into a validated document DOM structure.

        Args:
            rfc_number (int): Numeric identifier of the target RFC.
            flat_blocks (list[CanonicalBlockDict]): Flat list of extracted intermediate blocks.
            source_type (SourceType): Structural parser format type specifier string.

        Returns:
            NormalizedRFC: Populated and validated root canonical Pydantic model artifact.

        Raises:
            ValueError: If the provided RFC numeric identifier is zero or negative.
        """
        if rfc_number <= 0:
            raise ValueError(
                f"Cannot build canonical tree. The provided rfc_number '{rfc_number}' "
                f"is invalid. Check file naming or parser extraction logic."
            )

        fallback_entry: RFCIndexEntryDict = {
            "rfc_number": rfc_number,
            "title": f"RFC {rfc_number}",
            "published_at": None,
            "status": "UNKNOWN",
            "stream": "IETF",
            "authors": [],
            "obsoletes": [],
            "updates": [],
            "updated_by": [],
            "protocol_family": None,
        }

        if str(rfc_number) not in self.metadata_lookup:
            logger.warning(
                f"No metadata entry discovered for RFC {rfc_number} inside the lookup ledger. "
                f"Generating placeholder structural fallbacks."
            )

        meta_dict = self.metadata_lookup.get(str(rfc_number), fallback_entry)
        pub_date_dict = meta_dict.get("published_at")
        parsed_date = None

        if pub_date_dict is not None:
            parsed_date = RFCPublicationDate(
                year=pub_date_dict["year"],
                month=pub_date_dict["month"],
            )

        metadata = RFCMetadata(
            rfc_number=rfc_number,
            source_type=source_type,
            title=meta_dict.get("title", f"RFC {rfc_number}"),
            published_at=parsed_date,
            status=meta_dict.get("status", "UNKNOWN"),
            stream=meta_dict.get("stream", "IETF"),
            authors=meta_dict.get("authors", []),
            obsoletes=meta_dict.get("obsoletes", []),
            updates=meta_dict.get("updates", []),
            updated_by=meta_dict.get("updated_by", []),
            protocol_family=meta_dict.get("protocol_family"),
        )

        preface_blocks: list[Block] = []
        sections_list: list[Section] = []
        h_path_to_section: dict[str, Section] = {}
        section_block_counters: dict[str, int] = {}

        for fb in flat_blocks:
            normative_stmts: list[NormativeStatement] = []

            extracted_statements = fb["metadata"].get("normative_statements", [])

            for stmt in extracted_statements:
                kw = stmt["keyword"]
                if kw in self.valid_keywords:
                    normative_stmts.append(
                        NormativeStatement(
                            keyword=kw,
                            statement_text=stmt["statement_text"],
                        )
                    )

            raw_type = fb["block_type"]
            strict_type: BlockType = INTERMEDIATE_TO_FINAL_TYPE_MAP.get(
                raw_type, "paragraph"
            )
            h_path = fb.get("hierarchy_path", "Document Root")
            sec_num = fb["metadata"].get("section_number")

            section_id, sec_token, title = self._parse_section_path(
                h_path, sec_num, rfc_number
            )
            section_block_counters[sec_token] = (
                section_block_counters.get(sec_token, 0) + 1
            )
            blk_index = section_block_counters[sec_token]

            block = Block(
                block_id=f"rfc{rfc_number}-{sec_token}-blk{blk_index}",
                block_type=strict_type,
                sourcecode_type=fb.get("sourcecode_type"),
                source_fragment=fb["source_fragment"],
                normalized_text=fb["normalized_text"],
                parsing_confidence=fb["parsing_confidence"],
                normative_statements=normative_stmts,
            )

            if h_path.lower() in ("document root", "preface"):
                preface_blocks.append(block)
            else:
                if h_path not in h_path_to_section:
                    path_parts = h_path.split(" > ")
                    new_section = Section(
                        section_id=section_id,
                        title=title,
                        hierarchy_path=path_parts,
                        section_depth=len(path_parts),
                        blocks=[],
                    )
                    sections_list.append(new_section)
                    h_path_to_section[h_path] = new_section

                h_path_to_section[h_path].blocks.append(block)

        return NormalizedRFC(
            rfc_id=int(rfc_number),
            metadata=metadata,
            sections=sections_list,
            preface_blocks=preface_blocks,
        )
