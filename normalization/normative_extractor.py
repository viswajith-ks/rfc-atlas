"""BCP-14 normative requirement extraction engine for text block enrichment."""

import re
from typing import get_args

from normalization.schema import (
    CanonicalBlockDict,
    ExtractedStatementDict,
    NormativeKeyword,
)


class NormativeExtractor:
    """Scans intermediate text blocks to isolate and normalize BCP-14 requirement keywords."""

    _VALID_KEYWORDS = frozenset(get_args(NormativeKeyword))

    # Strict word-boundary match for exact BCP-14 compliance keywords to prevent partial matches
    # (e.g., catching "MUST" without catching "MUSTARD").
    _KEYWORD_PATTERN = re.compile(
        r"\b(MUST\s+NOT|MUST|REQUIRED|SHALL\s+NOT|SHALL|SHOULD\s+NOT|SHOULD|NOT\s+RECOMMENDED|RECOMMENDED|MAY|OPTIONAL)\b"
    )

    # NLP regex to safely split block text into individual sentences
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    _NORMALIZATION_MAP: dict[str, NormativeKeyword] = {
        "MUST": "MUST",
        "REQUIRED": "MUST",
        "SHALL": "MUST",
        "MUST NOT": "MUST NOT",
        "SHALL NOT": "MUST NOT",
        "SHOULD": "SHOULD",
        "RECOMMENDED": "SHOULD",
        "SHOULD NOT": "SHOULD NOT",
        "NOT RECOMMENDED": "SHOULD NOT",
        "MAY": "MAY",
        "OPTIONAL": "MAY",
    }

    _EXEMPT_SECTIONS: frozenset[str] = frozenset(
        {
            "abstract",
            "acknowledgments",
            "acknowledgements",
            "status of this memo",
            "table of contents",
            "author's address",
            "authors' addresses",
            "copyright notice",
            "references",
        }
    )

    def _is_exempt(self, hierarchy_path: str) -> bool:
        """Evaluates a section trajectory path to identify structural exclusions.

        Args:
            hierarchy_path (str): The full breadcrumb trajectory text path string.

        Returns:
            bool: True if the section context matches an exclusion constraint, else False.
        """
        parts = hierarchy_path.split(" > ")

        if any(p.lower() == "back" for p in parts):
            return True

        current_section = parts[-1]
        clean_section = (
            # Matches hierarchy paths starting with "appendix" followed by a space, alphanumeric characters/dots,
            # and trailing spaces. Used to cleanly strip appendix prefixes for section evaluations.
            re.sub(
                r"^(appendix\s+[a-z0-9.]+\s*)", "", current_section, flags=re.IGNORECASE
            )
            .strip()
            .lower()
        )

        return clean_section in self._EXEMPT_SECTIONS

    def process_blocks(
        self, blocks: list[CanonicalBlockDict]
    ) -> list[CanonicalBlockDict]:
        """Iterates across intermediate blocks to append extracted keyword metadata vectors.

        Args:
            blocks (list[CanonicalBlockDict]): A list of target intermediate block maps.

        Returns:
            list[CanonicalBlockDict]: Processed block collection records with added keywords.
        """
        enriched_blocks: list[CanonicalBlockDict] = []
        target_block_types = ("prose", "security", "list", "table")

        for block in blocks:
            if block["block_type"] in target_block_types and not self._is_exempt(
                block["hierarchy_path"]
            ):
                sentences = self._SENTENCE_SPLIT_RE.split(block["normalized_text"])
                extracted_statements: list[ExtractedStatementDict] = []

                for sentence in sentences:
                    for match in self._KEYWORD_PATTERN.finditer(sentence):
                        kw = " ".join(match.group(0).split())

                        normalized_kw = self._NORMALIZATION_MAP.get(kw)

                        if normalized_kw is not None:
                            statement: ExtractedStatementDict = {
                                "keyword": normalized_kw,
                                "statement_text": sentence.strip(),
                            }
                            extracted_statements.append(statement)

                if extracted_statements:
                    block["metadata"]["normative_statements"] = extracted_statements
            enriched_blocks.append(block)

        return enriched_blocks
