"""Canonical Tree Assembly Module

This module serves as the primary schema enforcer. It accepts flat lists of
heuristically or structurally parsed blocks and combines them with global
metadata to construct deeply nested, fully compliant Pydantic NormalizedRFC objects.
"""

import contextlib
import json
import os
from datetime import datetime
from typing import Any

from normalization.schema import (
    Block,
    NormalizedRFC,
    NormativeStatement,
    RFCMetadata,
    Section,
)


class CanonicalTreeBuilder:
    """Transforms a flat list of parsed blocks into the strict, nested
    NormalizedRFC Pydantic schema using the canonical metadata index.
    """

    def __init__(self, metadata_lookup_path: str):
        """Initializes the Canonical Tree Builder.

        Args:
            metadata_lookup_path (str): File path to the pre-compiled JSON metadata lookup table.

        Raises:
            FileNotFoundError: If the metadata lookup table has not been generated.
        """
        if not os.path.exists(metadata_lookup_path):
            raise FileNotFoundError(f"Missing metadata index at {metadata_lookup_path}")

        with open(metadata_lookup_path, encoding="utf-8") as f:
            self.metadata_lookup = json.load(f)

    def build_tree(
        self, rfc_id: str, flat_blocks: list[dict[str, Any]], source_type: str
    ) -> NormalizedRFC:
        """Assembles flat document blocks into a nested Pydantic DOM.

        Resolves string-based publication dates into Python datetime objects, groups
        sequential blocks under their respective hierarchy paths, and validates all
        types against the Canonical Normalized RFC Schema.

        Args:
            rfc_id (str): The numerical identifier of the document (e.g., "8446").
            flat_blocks (List[Dict[str, Any]]): The linear array of extracted text blocks.
            source_type (str): The origin format (either "xml" or "txt").

        Returns:
            NormalizedRFC: A fully populated, validated Pydantic object representing the document.
        """
        rfc_num = str(int(rfc_id))  # Normalizes "0001" to "1"
        meta_dict = self.metadata_lookup.get(rfc_num, {})

        # Convert text-based publication dates into strict datetime objects
        raw_date = meta_dict.get("published_at")
        parsed_date = None
        if raw_date:
            with contextlib.suppress(ValueError):
                parsed_date = datetime.strptime(raw_date, "%B %Y").date()

        # 1. Hydrate the Metadata Object
        metadata = RFCMetadata(
            rfc_number=int(rfc_num),
            source_type=source_type,
            title=meta_dict.get("title", f"RFC {rfc_num}"),
            published_at=parsed_date,
            status=meta_dict.get("status", "UNKNOWN"),
            stream=meta_dict.get("stream", "IETF"),
            authors=meta_dict.get("authors", []),
            obsoletes=meta_dict.get("obsoletes", []),
            updates=meta_dict.get("updates", []),
            updated_by=meta_dict.get("updated_by", []),
            protocol_family=meta_dict.get("protocol_family"),
        )

        preface_blocks = []
        sections_dict = {}  # Groups blocks by their hierarchy_path string

        # Map our flat parser block_types to the strict schema Literal types
        type_mapping = {
            "prose": "paragraph",
            "normative": "paragraph",  # Normative is a property of a paragraph
            "artwork": "artwork",
            "sourcecode": "sourcecode",
            "table": "table",
            "list": "list",
            "abnf": "abnf",
        }

        # 2. Process Blocks and Group by Section
        for i, fb in enumerate(flat_blocks):
            # Convert normative keywords into the Pydantic NormativeStatement schema
            normative_stmts = []
            keywords = fb.get("metadata", {}).get("normative_keywords", [])
            for kw in keywords:
                # Enforce strict schema constraints for BCP-14 keywords
                if kw in ["MUST", "SHOULD", "MAY", "MUST NOT", "SHOULD NOT"]:
                    normative_stmts.append(
                        NormativeStatement(
                            keyword=kw, statement_text=fb["normalized_text"]
                        )
                    )

            # Resolve strict block type
            raw_type = fb.get("block_type", "prose")
            strict_type = type_mapping.get(raw_type, "paragraph")

            # Hydrate the Block Object
            block = Block(
                block_id=f"rfc{rfc_num}-blk{i}",
                block_type=strict_type,
                source_fragment=fb.get("source_fragment", ""),
                normalized_text=fb.get("normalized_text", ""),
                parsing_confidence=fb.get("parsing_confidence", 0.5),
                normative_statements=normative_stmts,
            )

            h_path = fb.get("hierarchy_path", "Document Root")

            # Route to Preface vs. Sections
            if h_path == "Document Root" or h_path.lower() == "preface":
                preface_blocks.append(block)
            else:
                if h_path not in sections_dict:
                    # e.g., "3. Security Considerations > 3.1. Threats" -> ["3. Security Considerations", "3.1. Threats"]
                    path_parts = h_path.split(" > ")
                    last_part = path_parts[-1]

                    # Heuristic to separate section ID ("3.1") from Title ("Threats")
                    id_split = last_part.split(" ", 1)
                    section_id = (
                        id_split[0]
                        if len(id_split) > 1 and any(c.isdigit() for c in id_split[0])
                        else "unknown"
                    )
                    title = id_split[1] if len(id_split) > 1 else last_part

                    sections_dict[h_path] = Section(
                        section_id=section_id.strip("."),
                        title=title.strip(),
                        hierarchy_path=path_parts,
                        section_depth=len(path_parts),
                        blocks=[],
                    )
                sections_dict[h_path].blocks.append(block)

        # 3. Assemble and Validate the Final Canonical Tree
        canonical_rfc = NormalizedRFC(
            rfc_id=int(rfc_num),
            metadata=metadata,
            sections=list(sections_dict.values()),
            preface_blocks=preface_blocks,
        )

        return canonical_rfc
