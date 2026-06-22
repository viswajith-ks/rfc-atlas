"""Heuristic text parser for extracting structured blocks from legacy plaintext RFCs (RFC 1 - 8649)."""

import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from normalization.schema import CanonicalBlockDict, IntermediateBlockType
from parsers.base import refine_block_type
from utils.exceptions import MalformedFilenameError


@dataclass
class HierarchyState:
    """Mutable state tracker for hierarchical breadcrumb resolution."""

    prefix_to_title: dict[str, str] = field(default_factory=dict[str, str])
    last_known_at_depth: dict[int, str] = field(
        default_factory=lambda: {0: "Document Root"}
    )
    in_back_matter: bool = False


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

    # Strict Roman Numeral validation (prevents matching English words like "VALID." or "CIVIL.")
    # Uses lookahead to ensure at least one valid char, followed by strict subtractive notation rules
    _ROMAN_HEADER_RE = re.compile(
        r"^(?=[MDCLXVI])(M*(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\.\s+(.*)$",
        re.IGNORECASE,
    )

    _BACK_MATTER_TITLES = frozenset({
        "references",
        "author's address",
        "authors' addresses",
        "acknowledgments",
        "acknowledgements",
    })

    _CONFIDENCE_MODERN: float = 0.8
    _CONFIDENCE_EARLY: float = 0.6
    _EARLY_RFC_THRESHOLD: int = 1000

    def __init__(self, txt_filepath: Path) -> None:
        """Initializes the parser and loads the plaintext document into memory.

        Args:
            txt_filepath (Path): Path to the target RFC plaintext file.

        Raises:
            MalformedFilenameError: If a valid numeric RFC ID cannot be extracted from the filename.
        """
        self.txt_filepath = txt_filepath

        try:
            with self.txt_filepath.open(encoding="utf-8") as f:
                self.raw_text = f.read()
        except UnicodeDecodeError:
            with self.txt_filepath.open(encoding="latin-1") as f:
                self.raw_text = f.read()

        filename = self.txt_filepath.name

        # Matches the literal string "rfc", captures one or more digits, and expects exactly ".txt".
        # Used to defensively extract the ID from the file path.
        match = re.search(r"rfc(\d+)\.txt", filename, re.IGNORECASE)

        if not match:
            raise MalformedFilenameError(filename)

        self.rfc_id = int(match.group(1))

        self.base_confidence = (
            self._CONFIDENCE_MODERN
            if self.rfc_id >= self._EARLY_RFC_THRESHOLD
            else self._CONFIDENCE_EARLY
        )

    def _strip_pagination(self, text: str) -> str:
        """Removes pagination boundaries and contextually stitches the document.

        This method removes the standard IETF pagination footprint (Footers,
        Form-Feeds, and Running Headers). It then applies a structural heuristic
        to the lines immediately preceding and following the page break to
        determine if they are part of the same continuous block (e.g., a sentence
        spanning two pages) or separate blocks (e.g., a table ending, followed
        by a new section header).

        Args:
            text (str): Raw multi-page text body of the RFC.

        Returns:
            str: Normalized single-stream text body where cross-page sentences
                are seamlessly stitched, but structural boundaries are preserved.
        """
        # Replace the matched pagination block with a highly unique temporary token
        tokenized = self._PAGINATION_RE.sub("\n__RFC_ATLAS_PAGE_BREAK__\n", text)

        lines = tokenized.split("\n")
        stitched_lines: list[str] = []

        for i, line in enumerate(lines):
            if line.strip() == "__RFC_ATLAS_PAGE_BREAK__":
                prev_line = ""
                for j in range(len(stitched_lines) - 1, -1, -1):
                    if stitched_lines[j].strip():
                        prev_line = stitched_lines[j]
                        break

                next_line = ""
                for j in range(i + 1, len(lines)):
                    if lines[j].strip():
                        next_line = lines[j]
                        break

                first_next = next_line.strip()
                is_header = False

                if first_next:
                    normalized_line = first_next.lower().rstrip(".:- ")
                    is_header = (
                        bool(self._NUMERIC_HEADER_RE.match(first_next))
                        or bool(self._APPENDIX_HEADER_RE.match(first_next))
                        or bool(self._UNNUMBERED_HEADERS_RE.match(normalized_line))
                    )

                prev_indent = len(prev_line) - len(prev_line.lstrip(" \t"))
                next_indent = len(next_line) - len(next_line.lstrip(" \t"))

                if is_header or prev_indent != next_indent:
                    stitched_lines.append("")
            else:
                stitched_lines.append(line)

        return "\n".join(stitched_lines)

    @staticmethod
    def _split_squashed_prose(text: str) -> tuple[str, str | None]:
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

    def _parse_header_pattern(
        self, first_line: str
    ) -> tuple[int, int, str, str | None]:
        """Matches a header against known structural patterns to extract base traits.

        Args:
            first_line (str): The isolated first line of the text block.

        Returns:
            tuple[int, int, str, str | None]: A tuple containing the base score,
                structural depth, isolated title, and any squashed prose remainder.
        """
        normalized_line = first_line.lower().rstrip(".:- ")

        if self._UNNUMBERED_HEADERS_RE.match(normalized_line):
            return 10, 1, first_line, None

        if match := self._NUMERIC_HEADER_RE.match(first_line):
            num_part, rest = match.group(1), match.group(2).strip()
            title, extra_prose = self._split_squashed_prose(rest)
            return 8, len(num_part.split(".")), title, extra_prose

        if match := self._APPENDIX_HEADER_RE.match(first_line):
            app_part, rest = match.group(1), match.group(2).strip()
            cleaned_app = re.sub(r"^Appendix\s+", "", app_part, flags=re.IGNORECASE)
            title, extra_prose = self._split_squashed_prose(rest)
            return 9, len(cleaned_app.split(".")), title, extra_prose

        if match := self._ROMAN_HEADER_RE.match(first_line):
            rest = match.group(2).strip()
            title, extra_prose = self._split_squashed_prose(rest)
            return 7, 1, title, extra_prose

        return 0, 1, first_line, None

    @staticmethod
    def _calculate_header_modifiers(
        first_line: str, active_lines: int, *, has_underline: bool
    ) -> int:
        """Calculates positive and negative scoring modifiers for a potential header.

        Args:
            first_line (str): The isolated first line of the text block.
            active_lines (int): Count of non-empty lines within the block.
            has_underline (bool): Flag indicating if the block uses underline formatting.

        Returns:
            int: The calculated scoring modifier to apply to the base pattern score.
        """
        modifier = 0

        if has_underline:
            modifier += 10

        if first_line.isupper() and len(first_line) < 60:
            modifier += 4

        if active_lines == 1 or (has_underline and active_lines == 2):
            modifier += 4

        if first_line.endswith((",", ";")) or re.search(
            r"\s{2,}(?:and|or|the|of)\s+", first_line.lower()
        ):
            modifier -= 6

        if len(first_line) > 120:
            modifier -= 12

        return modifier

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
        has_underline = len(lines) >= 2 and bool(
            re.match(r"^[-=_\s]{3,}$", lines[1].strip())
        )

        prose_start_idx = 2 if has_underline else 1
        prose_remainder = "\n".join(lines[prose_start_idx:]).strip() or None

        base_score, depth, title, extra_prose = self._parse_header_pattern(first_line)

        active_lines = len([line for line in lines if line.strip()])
        modifier_score = self._calculate_header_modifiers(
            first_line, active_lines, has_underline=has_underline
        )

        total_score = base_score + modifier_score

        if extra_prose:
            prose_remainder = (
                f"{extra_prose}\n{prose_remainder}" if prose_remainder else extra_prose
            )

        threshold = 10 if self.rfc_id >= self._EARLY_RFC_THRESHOLD else 8

        if total_score >= threshold:
            clean_title = title.strip() or first_line
            return True, depth, clean_title, prose_remainder

        return False, 0, "", None

    @staticmethod
    def is_indented_block(block_lines: list[str]) -> bool:
        """Evaluates if standard layout margin offsets match formatting definitions.

        Args:
            block_lines (list[str]): Fragment row elements separated out by row breaks.

        Returns:
            bool: True if alignment satisfies criteria, else False.
        """
        for line in block_lines:
            if not line.strip():
                continue

            if not line.startswith("    "):
                return False
        return True

    def _merge_indented_blocks(self, clean_text: str) -> list[str]:
        """Splits raw text into paragraph blocks and merges contiguous indented blocks.

        Args:
            clean_text (str): Cleaned unified plain text document stream.

        Returns:
            list[str]: Collection of grouped layout strings.
        """
        normalized_text = clean_text.replace("\r\n", "\n").replace("\r", "\n")
        raw_blocks = re.split(r"\n\s*\n", normalized_text)

        merged_blocks: list[str] = []
        was_previous_indented = False

        for raw_block in raw_blocks:
            stripped_block = raw_block.strip("\n")
            if not stripped_block.strip():
                continue

            lines = stripped_block.split("\n")
            is_indented = self.is_indented_block(lines)

            if is_indented and was_previous_indented and merged_blocks:
                merged_blocks[-1] += "\n\n" + stripped_block
            else:
                merged_blocks.append(stripped_block)

            was_previous_indented = is_indented

        return merged_blocks

    @staticmethod
    def _calculate_abnf_score(block: str) -> int:
        """Calculates the likelihood that a block contains ABNF syntax heuristics.

        Args:
            block (str): The isolated block text.

        Returns:
            int: The calculated confidence score for ABNF classification.
        """
        score = 0
        if "::=" in block:
            score += 8
        if " =/ " in block:
            score += 8
        if re.search(r"%[xdb][0-9a-fA-F.-]+", block):
            score += 6
        if re.search(r"\b\d*\*+\d*[a-zA-Z]|\b\d+[a-zA-Z]", block):
            score += 4

        core_rules = (
            r"\b(CRLF|DIGIT|ALPHA|HEXDIG|DQUOTE|SP|WSP|OCTET|VCHAR|HTAB|BIT|CHAR|CTL)\b"
        )
        if re.search(core_rules, block):
            score += 3

        rule_assignments = re.findall(r"^\s*[a-zA-Z0-9-]+\s*=\s*", block, re.MULTILINE)
        score += len(rule_assignments) * 2

        if "[" in block and "]" in block and len(rule_assignments) < 2:
            score -= 5
        if re.search(r"^\s*(?:if|export|set|chown|chmod)\b", block, re.MULTILINE):
            score -= 8

        return score

    def _build_path_array(
        self,
        first_line: str,
        clean_title: str,
        depth: int,
        state: HierarchyState,
    ) -> list[str]:
        """Constructs the breadcrumb array based on numeric or appendix patterns.

        Args:
            first_line (str): First line of the evaluated header block.
            clean_title (str): The stripped section title.
            depth (int): The structural depth calculated by the header parser.
            state (HierarchyState): Mutable state tracker containing current prefix maps and depths.

        Returns:
            list[str]: The reconstructed list of hierarchical breadcrumbs.
        """
        reconstructed: list[str] = []

        if numeric_match := self._NUMERIC_HEADER_RE.match(first_line):
            num_part = numeric_match.group(1)
            state.prefix_to_title[num_part] = clean_title
            parts = num_part.split(".")
            for i in range(1, len(parts) + 1):
                sub_prefix = ".".join(parts[:i])
                fallback = state.last_known_at_depth.get(i, f"Untitled Level {i}")
                reconstructed.append(state.prefix_to_title.get(sub_prefix, fallback))
            return reconstructed

        if appendix_match := self._APPENDIX_HEADER_RE.match(first_line):
            app_part = appendix_match.group(1)
            state.prefix_to_title[app_part.lower()] = clean_title
            cleaned_app = re.sub(r"^Appendix\s+", "", app_part, flags=re.IGNORECASE)
            parts = cleaned_app.split(".")
            for i in range(1, len(parts) + 1):
                sub_parts = ".".join(parts[:i])
                sub_prefix = f"appendix {sub_parts}".lower()
                fallback = state.last_known_at_depth.get(i, f"Untitled Level {i}")
                reconstructed.append(state.prefix_to_title.get(sub_prefix, fallback))
            return reconstructed

        reconstructed.extend(
            state.last_known_at_depth.get(d, f"Untitled Level {d}")
            for d in range(1, depth)
        )
        reconstructed.append(clean_title)

        return reconstructed

    def _resolve_hierarchy_path(
        self,
        first_line: str,
        clean_title: str,
        depth: int,
        state: HierarchyState,
    ) -> list[str]:
        """Resolves the document hierarchy path and mutates tracking state in place.

        Args:
            first_line (str): First line of the evaluated header block.
            clean_title (str): The stripped section title.
            depth (int): The structural depth calculated by the header parser.
            state (HierarchyState): Mutable state tracker for hierarchical breadcrumb resolution.

        Returns:
            list[str]: The newly reconstructed path array.
        """
        reconstructed = self._build_path_array(first_line, clean_title, depth, state)

        normalized_line = first_line.lower().rstrip(".:- ")
        appendix_match = bool(self._APPENDIX_HEADER_RE.match(first_line))

        if appendix_match or normalized_line in self._BACK_MATTER_TITLES:
            state.in_back_matter = True
        elif state.in_back_matter and re.match(r"^\d+\.", normalized_line):
            state.in_back_matter = False

        for idx, level_title in enumerate(reconstructed, start=1):
            state.last_known_at_depth[idx] = level_title

        stale_keys = [k for k in state.last_known_at_depth if k > len(reconstructed)]
        for k in stale_keys:
            del state.last_known_at_depth[k]

        if state.in_back_matter:
            return ["Back", *reconstructed]
        return reconstructed

    def _build_content_block(
        self, block: str, current_hierarchy: list[str]
    ) -> CanonicalBlockDict:
        """Constructs a normalized content block from raw text and hierarchy context.

        Args:
            block (str): Target prose or artwork text block.
            current_hierarchy (list[str]): The live structural breadcrumb path.

        Returns:
            CanonicalBlockDict: Structured artifact ready for canonical JSON schema.
        """
        lines = block.split("\n")
        is_indented = self.is_indented_block(lines)
        block_type: IntermediateBlockType = "prose"

        if is_indented:
            abnf_score = self._calculate_abnf_score(block)
            block_type = "abnf" if abnf_score >= 8 else "artwork"
            normalized_text = block
        else:
            normalized_text = " ".join(block.split())

        path_string = " > ".join(current_hierarchy)
        block_type = refine_block_type(block_type, path_string.lower())

        return {
            "rfc_id": self.rfc_id,
            "hierarchy_path": path_string,
            "block_type": block_type,
            "source_type": "txt",
            "normalized_text": normalized_text,
            "source_fragment": block,
            "parsing_confidence": self.base_confidence,
            "metadata": {"element_id": None},
        }

    def _extract_blocks(self, clean_text: str) -> list[CanonicalBlockDict]:
        """Segments layout rows into contextual text chunks based on formatting profiles.

        Args:
            clean_text (str): Cleaned unified plain text document stream.

        Returns:
            list[CanonicalBlockDict]: Flat list of extracted, structured block records.
        """
        merged_blocks = self._merge_indented_blocks(clean_text)

        canonical_blocks: list[CanonicalBlockDict] = []
        current_hierarchy = ["Document Root"]
        block_queue = deque(merged_blocks)

        state = HierarchyState()

        while block_queue:
            block = block_queue.popleft().strip("\n")
            if not block.strip():
                continue

            is_header, depth, title, prose_remainder = self._evaluate_header(block)

            if is_header:
                clean_title = (
                    " ".join(title.split()).strip(".:- ") or "Untitled Section"
                )
                first_line = block.split("\n")[0].strip()

                current_hierarchy = self._resolve_hierarchy_path(
                    first_line,
                    clean_title,
                    depth,
                    state,
                )

                if prose_remainder and prose_remainder.strip():
                    block_queue.appendleft(prose_remainder)
                continue

            canonical_blocks.append(self._build_content_block(block, current_hierarchy))

        return canonical_blocks

    def parse_document(self) -> list[CanonicalBlockDict]:
        """Triggers the step-by-step extraction workflow across loaded file strings.

        Returns:
            list[CanonicalBlockDict]: Flat list of unified, parsed block dictionary entities.
        """
        clean_text = self._strip_pagination(self.raw_text)
        return self._extract_blocks(clean_text)
