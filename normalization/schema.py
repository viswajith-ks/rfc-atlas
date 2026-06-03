"""Canonical Normalized RFC Schema

This module defines the foundational data contracts for the RFC Intelligence System.
All upstream parsing pipelines (both modern XML and legacy TXT) MUST conform
to these Pydantic models before data is embedded or stored.

These models guarantee structural preservation, citation integrity, and deterministic
parsing confidence across all three eras of RFC documents.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class RFCMetadata(BaseModel):
    """Canonical metadata extracted from the RFC Index XML.
    Provides the ground-truth temporal and relational context for the document.
    """

    rfc_number: int
    source_type: Literal["xml", "txt"]
    title: str
    published_at: date | None
    status: (
        str | None
    )  # e.g., "INTERNET_STANDARD", "INFORMATIONAL", "PROPOSED_STANDARD"
    stream: str | None  # e.g., "IETF", "IAB", "IRTF"
    authors: list[str] = Field(default_factory=list)

    # Graph Database Readiness: Mapped directly to Neo4j edges later
    obsoletes: list[int] = Field(default_factory=list)
    updates: list[int] = Field(default_factory=list)
    updated_by: list[int] = Field(default_factory=list)
    protocol_family: str | None


class NormativeStatement(BaseModel):
    """Extracted BCP-14 Normative Requirement.
    Identifies strict protocol requirements for downstream LLM synthesis.
    """

    keyword: Literal["MUST", "SHOULD", "MAY", "MUST NOT", "SHOULD NOT"]
    statement_text: str
    actor: str | None = None  # Who the requirement applies to (if extractable)
    referenced_rfcs: list[int] = Field(default_factory=list)


class Block(BaseModel):
    """The atomic unit of the intelligence system.

    This preserves both the original text (for UI citation integrity) and
    the normalized text (for semantic embedding search).
    """

    block_id: str  # Deterministic ID, e.g., "rfc8446-sec3.1-blk4"
    block_type: Literal["paragraph", "sourcecode", "artwork", "table", "list", "abnf"]

    source_fragment: str  # The raw, untouched string from the source file (For UI)
    normalized_text: str  # Cleaned text used for LLM context and Vector Search

    # Trust metric: 1.0 for native XML tags, ~0.8 for regex nroff, ~0.4 for typewriter guesses
    parsing_confidence: float = Field(ge=0.0, le=1.0)

    normative_statements: list[NormativeStatement] = Field(default_factory=list)


class Section(BaseModel):
    """Represents a structural hierarchy node within an RFC.
    Holds the contiguous blocks that make up that section.
    """

    section_id: str  # e.g., "3.1"
    title: str
    hierarchy_path: list[str]  # e.g., ["3", "3.1"]
    section_depth: int
    blocks: list[Block]


class NormalizedRFC(BaseModel):
    """The root canonical artifact for a single parsed RFC.
    This is the final state of the document before chunking and embedding.
    """

    rfc_id: int
    metadata: RFCMetadata
    sections: list[Section]

    # Content that appears before the first formal section (e.g., Abstract, Status)
    preface_blocks: list[Block] = Field(default_factory=list)
