"""Base parser interface protocols for RFC document processing."""

from typing import Protocol

from normalization.schema import CanonicalBlockDict, IntermediateBlockType


class RFCParser(Protocol):
    """Protocol defining the structural interface for era-specific RFC parsers."""

    rfc_id: int

    def parse_document(self) -> list[CanonicalBlockDict]:
        """Parses an RFC document into a flat list of canonical intermediate blocks.

        Returns:
            list[CanonicalBlockDict]: A flat list of canonical intermediate block dictionaries
                ready for normative extraction and tree assembly.
        """
        ...


def refine_block_type(
    block_type: IntermediateBlockType, hierarchy_path: str
) -> IntermediateBlockType:
    """Promotes generic prose blocks to specialized types based on section hierarchy.

    Examines the hierarchical breadcrumbs of a document block to determine if
    a generic text block (like 'prose' or 'paragraph') should be semantically
    rerouted to a highly specialized vector table, such as 'security' or 'references'.

    Args:
        block_type (IntermediateBlockType): The initial structural classification of the block (e.g., 'prose').
        hierarchy_path (str): The pre-joined, lowercase hierarchical section path (e.g., '1. introduction > 1.1 background').

    Returns:
        IntermediateBlockType: The refined semantic block type. Returns the original block_type
                               if no hierarchical refinement is applicable.
    """
    if not hierarchy_path or block_type not in ("prose", "paragraph"):
        return block_type

    if "security considerations" in hierarchy_path:
        return "security"
    if "references" in hierarchy_path:
        return "references"

    return block_type
