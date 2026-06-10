"""Heuristic text parser for extracting structured blocks from legacy plaintext RFCs (RFC 1 - 8649)."""

import re
from collections import deque
from pathlib import Path

from normalization.schema import CanonicalBlockDict, IntermediateBlockType


class LegacyTextParser:
    """Heuristic structural parser for plaintext IETF RFC documents."""

    # Mathematically identical to the original wildcard logic, but strictly linear.
    # [^\S\r\n] safely matches ALL horizontal Unicode/ASCII whitespace.
    # (?:\r\n?|\n) universally handles Windows (\r\n), Unix (\n), and Classic Mac (\r).
    _PAGINATION_RE = re.compile(
        r"\n*[^\n]*\[Page\s+\d+\][^\n]*(?:\r\n?|\n)"  # 1. Footer line
        r"(?:[^\S\r\n]*(?:\r\n?|\n))*"  # 2. Blank padding lines (Classic Mac Safe)
        r"[^\S\r\n]*\x0c"  # 3. The literal form-feed character
        r"(?:[^\S\r\n]*(?:\r\n?|\n))*"  # 4. Blank lines before header
        r"[^\n]*(?:\r\n?|\n)"  # 5. Running header line
        r"(?:[^\S\r\n]*(?:\r\n?|\n))*",  # 6. Cleans all trailing layout padding
        re.IGNORECASE,
    )

    # A structurally flexible regular expression absorbing spelling and pluralization variations
    _UNNUMBERED_HEADERS_RE = re.compile(
        r"^(?:"
        r"abstract|"
        r"status\s+of\s+this\s+memo|"
        r"table\s+of\s+contents|"
        r"introduction|"
        r"security\s+considerations|"
        r"references|"
        r"authors?'?s?'?\s+addresses?|"
        r"acknowledgements?"
        r")$",
        re.IGNORECASE,
    )

    # Captures numeric hierarchy headers. Group 1 grabs the dot-separated numbers (e.g., "1.2.3"),
    # and Group 2 grabs the remaining title text.
    _NUMERIC_HEADER_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)*)(?:\.|\s+)(.*)$")

    # Captures Appendix-style headers. Group 1 grabs "Appendix A.1",
    # and Group 2 grabs the remaining title text after ignoring trailing punctuation.
    _APPENDIX_HEADER_RE = re.compile(
        r"^(Appendix\s+[A-Z0-9]+(?:\.[0-9]+)*)(?:\.|\s+|-*)(.*)$",
        re.IGNORECASE,
    )

    # Captures legacy Roman Numeral headers. Group 1 grabs the numeral (e.g., "IV"), Group 2 grabs the title.
    _ROMAN_HEADER_RE = re.compile(r"^([IVXLCDM]+)\.\s+(.*)$", re.IGNORECASE)

    _BACK_MATTER_TITLES = frozenset(
        {
            "references",
            "author's address",
            "authors' addresses",
            "acknowledgments",
            "acknowledgements",
        }
    )

    _CONFIDENCE_MODERN: float = 0.8
    _CONFIDENCE_EARLY: float = 0.6
    _EARLY_RFC_THRESHOLD: int = 1000

    def __init__(self, txt_filepath: Path) -> None:
        """Initializes the parser and loads the plaintext document into memory.

        Args:
            txt_filepath (Path): Path to the target RFC plaintext file.

        Raises:
            ValueError: If a valid numeric RFC ID cannot be extracted from the filename.
        """
        self.txt_filepath = txt_filepath

        with self.txt_filepath.open(encoding="utf-8", errors="ignore") as f:
            self.raw_text = f.read()

        filename = self.txt_filepath.name

        # Matches the literal string "rfc", captures one or more digits, and expects exactly ".txt".
        # Used to defensively extract the ID from the file path.
        match = re.search(r"rfc(\d+)\.txt", filename, re.IGNORECASE)

        if not match:
            raise ValueError(f"Cannot extract numeric RFC ID from filename: {filename}")

        self.rfc_id = int(match.group(1))

        self.base_confidence = (
            self._CONFIDENCE_MODERN
            if self.rfc_id >= self._EARLY_RFC_THRESHOLD
            else self._CONFIDENCE_EARLY
        )

    def _strip_pagination(self, text: str) -> str:
        """Removes form feeds, footers, and headers associated with page boundaries.

        Args:
            text (str): Raw multi-page text body of the RFC.

        Returns:
            str: Normalized single-stream text body stripped of pagination blocks.
        """
        return self._PAGINATION_RE.sub("\n", text)

    def _split_squashed_prose(self, text: str) -> tuple[str, str | None]:
        """Separates an inline section header from trailing body text paragraph boundaries.

        Args:
            text (str): Raw line segment text found after structural tokens.

        Returns:
            tuple[str, str | None]: Cleaned section header title and trailing prose segment if present.
        """
        if not text:
            return "", None

        # Splits text where a section header has been improperly squashed against body prose.
        # Splits on either 2+ spaces followed by a capital letter,
        # OR a period after a lowercase letter/digit followed by a space and a capital letter.
        squash_split = re.split(
            r"\s{2,}(?=[A-Z])|(?<=[a-z0-9]\.)\s+(?=[A-Z])", text, maxsplit=1
        )

        if len(squash_split) == 2 and len(squash_split[0].split()) <= 6:
            return squash_split[0].strip(), squash_split[1].strip()

        return text, None

    def _evaluate_header(self, block: str) -> tuple[bool, int, str, str | None]:
        """Evaluates a raw block structure to determine if it functions as a section header.

        Args:
            block (str): Target text block pulled from processing stream.

        Returns:
            tuple[bool, int, str, str | None]: Classification indicator flag, structural section depth,
                isolated header title string, and remaining trailing text paragraphs.
        """
        lines = block.split("\n")
        if not lines or not lines[0].strip():
            return False, 0, "", None

        first_line = lines[0].strip()
        has_underline = False

        # Matches a line consisting of exactly 3 or more hyphens, equal signs,
        # underscores, or spaces. Used to detect underline-style section headers.
        if len(lines) >= 2 and re.match(r"^[-=_\s]{3,}$", lines[1].strip()):
            has_underline = True

        score = 0
        depth = 1
        title = first_line

        prose_remainder = (
            "\n".join(lines[1:]) if not has_underline else "\n".join(lines[2:])
        )
        if prose_remainder and not prose_remainder.strip():
            prose_remainder = None

        numeric_match = self._NUMERIC_HEADER_RE.match(first_line)
        appendix_match = self._APPENDIX_HEADER_RE.match(first_line)
        roman_match = self._ROMAN_HEADER_RE.match(first_line)

        normalized_line = first_line.lower().rstrip(".:- ")
        is_unnum = bool(self._UNNUMBERED_HEADERS_RE.match(normalized_line))

        if is_unnum:
            score += 10
            title = first_line
            depth = 1

        elif numeric_match:
            score += 8
            num_part = numeric_match.group(1)
            rest = numeric_match.group(2).strip()
            depth = len(num_part.split("."))
            title, extra_prose = self._split_squashed_prose(rest)

            if extra_prose:
                prose_remainder = (
                    extra_prose + "\n" + (prose_remainder if prose_remainder else "")
                )

        elif appendix_match:
            score += 9
            app_part = appendix_match.group(1)
            rest = appendix_match.group(2).strip()

            # Matches the exact prefix "Appendix " at the start of a string.
            # Used to strip the label for depth calculations.
            cleaned_app = re.sub(r"^Appendix\s+", "", app_part, flags=re.IGNORECASE)
            depth = len(cleaned_app.split("."))
            title, extra_prose = self._split_squashed_prose(rest)

            if extra_prose:
                prose_remainder = (
                    extra_prose + "\n" + (prose_remainder if prose_remainder else "")
                )

        elif roman_match:
            score += 7
            rest = roman_match.group(2).strip()
            depth = 1
            title, extra_prose = self._split_squashed_prose(rest)

            if extra_prose:
                prose_remainder = (
                    extra_prose + "\n" + (prose_remainder if prose_remainder else "")
                )

        if has_underline:
            score += 10

        if first_line.isupper() and len(first_line) < 60:
            score += 4

        if len(lines) == 1 or (
            has_underline and len([line for line in lines if line.strip()]) == 2
        ):
            score += 4

        if first_line.endswith((",", ";")) or re.search(
            # Detects 2+ spaces immediately followed by common conjunctions/prepositions.
            # Used to negatively score lines, identifying them as lists or tables rather than section headers.
            r"\s{2,}(?:and|or|the|of)\s+",
            first_line.lower(),
        ):
            score -= 6

        if len(first_line) > 120:
            score -= 12

        threshold = 10 if self.rfc_id >= self._EARLY_RFC_THRESHOLD else 8

        if score >= threshold:
            if not title.strip():
                title = first_line

            return (
                True,
                depth,
                title,
                prose_remainder.strip() if prose_remainder else None,
            )

        return False, 0, "", None

    @staticmethod
    def is_indented_block(block_lines: list[str]) -> bool:
        """Evaluates if standard layout margin offsets match formatting definitions.

        Args:
            block_lines (list[str]): Fragment row elements separated out by row breaks.

        Returns:
            bool: True if alignment satisfies criteria, else False.
        """
        return all(line.startswith("    ") for line in block_lines if line.strip())

    def _extract_blocks(self, clean_text: str) -> list[CanonicalBlockDict]:
        """Segments layout rows into contextual text chunks based on formatting profiles.

        Args:
            clean_text (str): Cleaned unified plain text document stream.

        Returns:
            list[CanonicalBlockDict]: Flat list of extracted, structured block records.
        """
        normalized_text = clean_text.replace("\r\n", "\n").replace("\r", "\n")

        # Matches two newline characters separated only by optional whitespace.
        # Used to chunk the raw text stream into isolated paragraphs.
        raw_blocks = re.split(r"\n\s*\n", normalized_text)

        merged_blocks: list[str] = []
        was_previous_indented = False

        for block in raw_blocks:
            block = block.strip("\n")
            if not block.strip():
                continue

            lines = block.split("\n")
            is_indented = self.is_indented_block(lines)

            if is_indented and was_previous_indented and merged_blocks:
                merged_blocks[-1] += "\n\n" + block
            else:
                merged_blocks.append(block)

            was_previous_indented = is_indented

        canonical_blocks: list[CanonicalBlockDict] = []
        current_hierarchy = ["Document Root"]
        block_queue = deque(merged_blocks)

        prefix_to_title: dict[str, str] = {}
        last_known_at_depth: dict[int, str] = {0: "Document Root"}

        in_back_matter = False

        while block_queue:
            block = block_queue.popleft().strip("\n")
            if not block.strip():
                continue

            is_header, depth, title, prose_remainder = self._evaluate_header(block)

            if is_header:
                clean_title = " ".join(title.split()).strip(".:- ")
                if not clean_title:
                    clean_title = "Untitled Section"

                lines = block.split("\n")
                first_line = lines[0].strip()

                numeric_match = self._NUMERIC_HEADER_RE.match(first_line)
                appendix_match = self._APPENDIX_HEADER_RE.match(first_line)

                reconstructed_hierarchy: list[str] = []

                if numeric_match:
                    num_part = numeric_match.group(1)
                    prefix_to_title[num_part] = clean_title

                    parts = num_part.split(".")
                    for i in range(1, len(parts) + 1):
                        sub_prefix = ".".join(parts[:i])
                        if sub_prefix in prefix_to_title:
                            reconstructed_hierarchy.append(prefix_to_title[sub_prefix])
                        else:
                            reconstructed_hierarchy.append(
                                last_known_at_depth.get(i, f"Untitled Level {i}")
                            )

                elif appendix_match:
                    app_part = appendix_match.group(1)
                    prefix_to_title[app_part.lower()] = clean_title

                    # Matches the exact prefix "Appendix " at the start of a string.
                    # Used to strip the label for depth calculations.
                    cleaned_app = re.sub(
                        r"^Appendix\s+", "", app_part, flags=re.IGNORECASE
                    )
                    parts = cleaned_app.split(".")
                    for i in range(1, len(parts) + 1):
                        sub_parts = ".".join(parts[:i])
                        sub_prefix = f"appendix {sub_parts}".lower()
                        if sub_prefix in prefix_to_title:
                            reconstructed_hierarchy.append(prefix_to_title[sub_prefix])
                        else:
                            reconstructed_hierarchy.append(
                                last_known_at_depth.get(i, f"Untitled Level {i}")
                            )

                else:
                    for d in range(1, depth):
                        reconstructed_hierarchy.append(
                            last_known_at_depth.get(d, f"Untitled Level {d}")
                        )
                    reconstructed_hierarchy.append(clean_title)

                normalized_line = first_line.lower().rstrip(".:- ")
                if appendix_match or normalized_line in self._BACK_MATTER_TITLES:
                    in_back_matter = True

                for idx, level_title in enumerate(reconstructed_hierarchy, start=1):
                    last_known_at_depth[idx] = level_title

                last_known_at_depth = {
                    d: t
                    for d, t in last_known_at_depth.items()
                    if d <= len(reconstructed_hierarchy)
                }

                if in_back_matter:
                    current_hierarchy = ["Back"] + reconstructed_hierarchy
                else:
                    current_hierarchy = reconstructed_hierarchy

                if prose_remainder and prose_remainder.strip():
                    block_queue.appendleft(prose_remainder)

                continue

            lines = block.split("\n")
            is_indented = self.is_indented_block(lines)
            block_type: IntermediateBlockType = "prose"

            if is_indented:
                abnf_score = 0

                if "::=" in block:
                    abnf_score += 8
                if " =/ " in block:
                    abnf_score += 8

                # Matches ABNF hex/decimal/binary definitions (e.g., %x0D, %b10000000).
                if re.search(r"%[xdb][0-9a-fA-F.-]+", block):
                    abnf_score += 6

                # Matches ABNF repetition sequence operators (e.g., 1*DIGIT, *CHAR).
                if re.search(r"\b\d*\*+\d*[a-zA-Z]|\b\d+[a-zA-Z]", block):
                    abnf_score += 4

                # Exact word boundary matches for standard IETF ABNF core rules.
                core_rules = r"\b(CRLF|DIGIT|ALPHA|HEXDIG|DQUOTE|SP|WSP|OCTET|VCHAR|HTAB|BIT|CHAR|CTL)\b"
                if re.search(core_rules, block):
                    abnf_score += 3

                # Matches ABNF variable assignment syntax (e.g., rule-name = ) at the beginning of a line.
                rule_assignments = re.findall(
                    r"^\s*[a-zA-Z0-9-]+\s*=\s*", block, re.MULTILINE
                )
                abnf_score += len(rule_assignments) * 2

                if "[" in block and "]" in block and len(rule_assignments) < 2:
                    abnf_score -= 5

                # Matches common shell-scripting keywords at the start of a line to deliberately penalize a
                # block's ABNF score (preventing shell code from being misclassified as ABNF).
                if re.search(
                    r"^\s*(?:if|export|set|chown|chmod)\b", block, re.MULTILINE
                ):
                    abnf_score -= 8

                block_type = "abnf" if abnf_score >= 8 else "artwork"
                normalized_text = block

            else:
                normalized_text = " ".join(block.split())

            path_lower = " > ".join(current_hierarchy).lower()

            if "security considerations" in path_lower:
                if block_type == "prose":
                    block_type = "security"
            elif "references" in path_lower and block_type == "prose":
                block_type = "references"

            canonical_blocks.append(
                {
                    "rfc_id": self.rfc_id,
                    "hierarchy_path": " > ".join(current_hierarchy),
                    "block_type": block_type,
                    "source_type": "txt",
                    "normalized_text": normalized_text,
                    "source_fragment": block,
                    "parsing_confidence": self.base_confidence,
                    "metadata": {
                        "element_id": None,
                        "normative_keywords": [],
                    },
                }
            )

        return canonical_blocks

    def parse_document(self) -> list[CanonicalBlockDict]:
        """Triggers the step-by-step extraction workflow across loaded file strings.

        Returns:
            list[CanonicalBlockDict]: Flat list of unified, parsed block dictionary entities.
        """
        clean_text = self._strip_pagination(self.raw_text)
        return self._extract_blocks(clean_text)
