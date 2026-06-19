"""Metadata Schemas for RFC Atlas.

ARCHITECTURAL NOTE (TypedDict vs Pydantic):
We intentionally maintain dual structures for these concepts.
1. `TypedDict` models are used for the global in-memory lookup table. This guarantees
   O(1) access speed and near-zero memory overhead across multiprocessing workers.
2. `Pydantic` models are used exclusively at the final normalization boundary to
   enforce strict serialization and validation of the Canonical Artifacts.
"""

from typing import TypedDict

from pydantic import BaseModel


class RFCPublicationDateDict(TypedDict):
    """Dictionary schema for intermediate storage of publication dates."""

    year: int
    month: int | None


class RFCPublicationDate(BaseModel):
    """Pydantic model representing RFC publication dates with variable precision.

    Preserves historical index dates that offer only year-level or month-year
    precision without generating inaccurate calendar days.
    """

    year: int
    month: int | None


class RFCIndexEntryDict(TypedDict):
    """Dictionary schema for structured entries in the compiled metadata lookup table."""

    rfc_number: int
    title: str
    published_at: RFCPublicationDateDict | None
    status: str
    stream: str
    authors: list[str]
    obsoletes: list[int]
    updates: list[int]
    updated_by: list[int]
    # Placeholder for future pipeline enrichment (e.g., IANA registry mapping).
    # Not populated by the baseline IETF index parser.
    protocol_family: str | None
