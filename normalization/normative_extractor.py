import json
import re
import os


class NormativeExtractor:
    """
    Scans canonical JSON blocks and enriches them by identifying RFC 2119 normative directives.
    Includes deterministic guardrails to prevent false positives in metadata sections.
    """

    def __init__(self):
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

    def _is_exempt(self, hierarchy_path):
        """
        Determines if a block belongs to a metadata section that should be ignored.
        Handles the exact depth nuance: "Abstract" is skipped, "3.1 Abstract Syntax" is scanned.
        """
        # Get the actual current section name (the last part of the breadcrumb trail)
        parts = [p.strip().lower() for p in hierarchy_path.split(">")]
        current_section = parts[-1]

        # Since numbered sections have numbers in them (e.g., "3.1 abstract syntax"),
        # an exact match against our exempt list naturally protects deep normative content!
        # We also strip out legacy appendix prefixes just in case (e.g., "Appendix A. Acknowledgments")
        clean_section = re.sub(
            r"^(appendix\s+[a-z0-9.]+\s*)", "", current_section
        ).strip()

        return clean_section in self.exempt_sections

    def process_blocks(self, blocks):
        """
        Iterates through a list of canonical blocks and applies normative enrichment.
        """
        enriched_blocks = []

        for block in blocks:
            # We only scan standard prose that is NOT in an exempt metadata section.
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


# --- Execution Example ---
if __name__ == "__main__":
    input_file = "data/normalized/rfc2616_normalized.json"
    output_file = "data/normalized/rfc2616_enriched.json"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        exit(1)

    print(f"Loading blocks from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        canonical_blocks = json.load(f)

    extractor = NormativeExtractor()
    enriched_blocks = extractor.process_blocks(canonical_blocks)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enriched_blocks, f, indent=2)

    total_blocks = len(enriched_blocks)
    normative_count = sum(1 for b in enriched_blocks if b["block_type"] == "normative")

    print(f"Success! Enriched {output_file}")
    print(f"Total Blocks: {total_blocks}")
    print(f"True Normative Requirements Found: {normative_count}")
