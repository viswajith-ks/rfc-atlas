"""BCP-14 normative requirement extraction engine for text block enrichment."""

import re
from typing import cast, get_args

from normalization.schema import CanonicalBlockDict, NormativeKeyword


class NormativeExtractor:
    """Scans intermediate text blocks to isolate and normalize BCP-14 requirement keywords."""

    _VALID_KEYWORDS = frozenset(get_args(NormativeKeyword))

    # Strict word-boundary match for exact BCP-14 compliance keywords to prevent partial matches
    # (e.g., catching "MUST" without catching "MUSTARD").
    _KEYWORD_PATTERN = re.compile(
        r"\b(MUST\s+NOT|MUST|REQUIRED|SHALL\s+NOT|SHALL|SHOULD\s+NOT|SHOULD|NOT\s+RECOMMENDED|RECOMMENDED|MAY|OPTIONAL)\b"
    )

    _NORMALIZATION_MAP: dict[str, NormativeKeyword] = {
        "REQUIRED": "MUST",
        "SHALL": "MUST",
        "SHALL NOT": "MUST NOT",
        "RECOMMENDED": "SHOULD",
        "NOT RECOMMENDED": "SHOULD NOT",
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
                matches: list[str] = self._KEYWORD_PATTERN.findall(
                    block["normalized_text"]
                )

                if matches:
                    normalized_keywords: list[NormativeKeyword] = []

                    for kw in matches:
                        clean_kw = " ".join(kw.split())
                        normalized_keywords.append(
                            self._NORMALIZATION_MAP.get(
                                clean_kw, cast(NormativeKeyword, clean_kw)
                            )
                        )

                    validated_keywords = cast(
                        list[NormativeKeyword],
                        [
                            kw
                            for kw in normalized_keywords
                            if kw in self._VALID_KEYWORDS
                        ],
                    )

                    if validated_keywords:
                        block = cast(
                            CanonicalBlockDict,
                            {
                                **block,
                                "metadata": {
                                    **block["metadata"],
                                    "normative_keywords": validated_keywords,
                                },
                            },
                        )

            enriched_blocks.append(block)

        return enriched_blocks
