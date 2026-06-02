import re
import os
import json


class LegacyTextParser:
    """
    A heuristic structural parser for legacy plaintext IETF RFC documents (RFC 1 - 8649).

    Before the adoption of xml2rfc v3, RFCs were published as pure plaintext files
    formatted by the 'nroff' typesetting engine. This parser reverse-engineers those
    formatting rules (such as pagination limits and strict indentation rules) using
    regular expressions to reconstruct semantic blocks into a canonical JSON schema.

    Attributes:
        txt_filepath (str): The local file path to the raw text RFC.
        raw_text (str): The unparsed string content of the document.
        rfc_number (str): The official RFC number extracted from the filename.
    """

    def __init__(self, txt_filepath):
        """
        Initializes the parser and loads the legacy text document into memory.

        Args:
            txt_filepath (str): Path to the target RFC .txt file.
        """
        self.txt_filepath = txt_filepath

        # Uses errors="ignore" to safely handle ancient 1980s character encodings
        with open(txt_filepath, "r", encoding="utf-8", errors="ignore") as f:
            self.raw_text = f.read()

        # Extract the top-level RFC number for data provenance
        filename = os.path.basename(txt_filepath)
        match = re.search(r"rfc(\d+)\.txt", filename, re.IGNORECASE)
        self.rfc_number = match.group(1) if match else "UNKNOWN"

    def _strip_pagination(self, text):
        """
        Removes the nroff typesetting page breaks to prevent semantic chunks
        from being sliced in half.

        Strictly removes the footer line, the form feed character (\\x0c), AND
        the next page's header line, along with all surrounding blank lines,
        replacing the entire void with a single newline to knit paragraphs back together.

        Args:
            text (str): The raw text of the RFC.

        Returns:
            str: The text with all pagination artifacts removed.
        """
        pagination_pattern = re.compile(
            r"\n*[^\n]*\[Page\s+\d+\][^\n]*\n\s*\x0c\s*[^\n]*\n\n*", re.IGNORECASE
        )
        return re.sub(pagination_pattern, "\n", text)

    def _extract_blocks(self, clean_text):
        """
        Splits the cleaned text into blocks, detects section headers to build
        lineage paths, and applies indentation heuristics to determine block types.

        Args:
            clean_text (str): The RFC text stripped of pagination.

        Returns:
            list[dict]: A list of canonical JSON artifacts representing each block.
        """
        # In nroff formatting, standard blocks are separated by double newlines
        raw_blocks = re.split(r"\n\s*\n", clean_text)
        canonical_blocks = []

        # Tracks the current section path (e.g., "1 Introduction > 1.4 Scope")
        current_hierarchy = ["Document Root"]

        # Matches numbered section headers (e.g., "1. Introduction" or "2.1.1. Spec")
        section_pattern = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+([A-Z].+)$")

        # Standard unnumbered top-level sections common in older RFCs
        unnumbered_headers = [
            "Abstract",
            "Status of this Memo",
            "Table of Contents",
            "Introduction",
            "Security Considerations",
            "References",
            "Author's Address",
        ]

        for block in raw_blocks:
            block = block.strip("\n")
            if not block.strip():
                continue

            # --- 1. Hierarchy & Section Detection ---
            first_line = block.split("\n")[0].strip()
            sec_match = section_pattern.match(first_line)
            is_unnumbered = (
                first_line in unnumbered_headers and len(block.split("\n")) == 1
            )

            # If the block is a header, update the hierarchy path and skip extraction
            if sec_match or is_unnumbered:
                title = first_line
                if sec_match:
                    # Calculate depth (e.g., "2.1." -> depth 2) to maintain accurate breadcrumbs
                    depth = len(sec_match.group(1).strip(".").split("."))
                    current_hierarchy = current_hierarchy[: depth - 1] + [title]
                else:
                    current_hierarchy = [title]
                continue

            # --- 2. Indentation Heuristics ---
            lines = block.split("\n")

            # Check if all lines are indented by 4 OR MORE spaces
            # (Standard nroff prose is indented by 3 spaces. 4+ indicates artwork/code)
            is_indented = all(
                len(line) - len(line.lstrip()) >= 4 for line in lines if line.strip()
            )

            if is_indented:
                # Detect ABNF grammar by looking for common assignment operators
                if "::=" in block or re.search(
                    r"^\s*[a-zA-Z0-9-]+\s*=\s*", block, re.MULTILINE
                ):
                    block_type = "abnf"
                else:
                    block_type = "artwork"

                # Preserve exact structural whitespace for diagrams and code
                normalized_text = block
            else:
                block_type = "prose"
                # Compress excessive newlines and spaces for standard readable text
                normalized_text = " ".join(block.split())

            # --- 3. Contextual Data Routing ---
            # Isolate specialized sections into dedicated database tables
            path_lower = " > ".join(current_hierarchy).lower()
            if "security considerations" in path_lower:
                block_type = "security"
            elif "references" in path_lower:
                block_type = "references"

            # --- 4. Canonical Artifact Construction ---
            canonical_blocks.append(
                {
                    "rfc_id": self.rfc_number,
                    "hierarchy_path": " > ".join(current_hierarchy),
                    "block_type": block_type,
                    "source_type": "txt",
                    "normalized_text": normalized_text,
                    "source_fragment": block,
                    "parsing_confidence": 0.8,  # Reflects the fuzzy nature of text heuristics
                    "metadata": {"element_id": None},  # XML anchors do not exist in txt
                }
            )

        return canonical_blocks

    def parse_document(self):
        """
        Executes the full legacy parsing pipeline.

        Returns:
            list[dict]: A list of canonical JSON artifacts representing each block.
        """
        clean_text = self._strip_pagination(self.raw_text)
        return self._extract_blocks(clean_text)


if __name__ == "__main__":
    txt_file = "data/raw/rfcs_txt/rfc2616.txt"
    output_file = "data/normalized/rfc2616_normalized.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Parsing {txt_file}...")
    parser = LegacyTextParser(txt_file)
    canonical_blocks = parser.parse_document()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(canonical_blocks, f, indent=2)
    print(f"Success! {len(canonical_blocks)} blocks written.")
