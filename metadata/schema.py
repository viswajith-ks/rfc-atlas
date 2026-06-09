"""Data schemas for RFC index metadata validation and intermediate storage."""

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
    protocol_family: str | None
