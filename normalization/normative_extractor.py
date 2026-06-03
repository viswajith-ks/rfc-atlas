"""Normative Requirement Extraction Module

Provides text analysis utilities to identify BCP-14 (RFC 2119) normative directives
within canonical protocol documents. It applies strict hierarchical safeguards to
prevent false positives inside metadata sections or appendices.
"""

import re


class NormativeExtractor:
    """Scans canonical JSON blocks and enriches them by identifying RFC 2119 normative directives.
    Includes deterministic guardrails to prevent false positives in metadata sections.
    """

    def __init__(self):
        """Initializes the normative extractor with strict BCP-14 regex patterns
        and defines the hierarchical exempt list.
        """
        # Regex to match RFC 2119 keywords exactly.
        self.keyword_pattern = re.compile(
            r"\b(MUST(?:\s+NOT)?|REQUIRED|SHALL(?:\s+NOT)?|SHOULD(?:\s+NOT)?|RECOMMENDED|MAY|OPTIONAL)\b"
        )

        # Sections that contain meta-references to rules, not actual protocol rules.
        self.exempt_sections = {
            "abstract",
            "acknowledgments",
            "acknowledgements",
            "status of this memo",
            "table of contents",
            "author's address",
            "authors' addresses",
            "copyright notice",
        }

    def _is_exempt(self, hierarchy_path: str) -> bool:
        """Determines if a block belongs to a metadata section that should be ignored.
        Handles the exact depth nuance: "Abstract" is skipped, "3.1 Abstract Syntax" is scanned.

        Args:
            hierarchy_path (str): The breadcrumb path of the section.

        Returns:
            bool: True if the section is exempt from normative scanning, False otherwise.
        """
        # Get the actual current section name (the last part of the breadcrumb trail)
        parts = [p.strip().lower() for p in hierarchy_path.split(">")]
        current_section = parts[-1]

        # Strip legacy appendix prefixes to cleanly evaluate the base title
        clean_section = re.sub(
            r"^(appendix\s+[a-z0-9.]+\s*)", "", current_section
        ).strip()

        return clean_section in self.exempt_sections

    def process_blocks(self, blocks: list) -> list:
        """Iterates through a list of canonical blocks and applies normative enrichment.

        Args:
            blocks (list): A list of flat dictionaries representing extracted document blocks.

        Returns:
            list: The modified list of blocks with normative metadata appended.
        """
        enriched_blocks = []

        for block in blocks:
            # Scans standard prose excluding exempt metadata sections.
            if block["block_type"] == "prose" and not self._is_exempt(
                block["hierarchy_path"]
            ):
                # Find all unique RFC 2119 keywords in the text
                matches = self.keyword_pattern.findall(block["normalized_text"])

                if matches:
                    unique_keywords = list(set(matches))

                    # 1. Elevate the block type
                    block["block_type"] = "normative"

                    # 2. Inject the extracted keywords into the metadata schema
                    block["metadata"]["normative_keywords"] = unique_keywords

            enriched_blocks.append(block)

        return enriched_blocks
