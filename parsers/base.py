"""Base parser interface protocols for RFC document processing."""

from typing import Protocol

from normalization.schema import CanonicalBlockDict


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
