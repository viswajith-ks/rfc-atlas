"""Canonical schemas and type definitions for normalized RFC document structures."""

from pathlib import Path
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from metadata.schema import RFCPublicationDate

ReferenceCategory = Literal["Normative", "Informative"]
SourceType: TypeAlias = Literal["xml", "txt"]

NormativeKeyword: TypeAlias = Literal[
    "MUST",
    "SHOULD",
    "MAY",
    "MUST NOT",
    "SHOULD NOT",
]

BlockType: TypeAlias = Literal[
    "paragraph",
    "sourcecode",
    "artwork",
    "table",
    "list",
    "abnf",
    "security",
    "references",
]

SourcecodeFormat: TypeAlias = Literal[
    "yang",
    "asn.1",
    "mib",
    "json",
    "pseudocode",
    "c",
    "go",
]

IntermediateBlockType: TypeAlias = Literal[
    "prose",
    "artwork",
    "table",
    "list",
    "abnf",
    "security",
    "references",
    "sourcecode",
]

INTERMEDIATE_TO_FINAL_TYPE_MAP: dict[IntermediateBlockType, BlockType] = {
    "prose": "paragraph",
    "artwork": "artwork",
    "sourcecode": "sourcecode",
    "table": "table",
    "list": "list",
    "abnf": "abnf",
    "security": "security",
    "references": "references",
}
"""Maps parser-specific intermediate categories into strict canonical BlockTypes."""

XML_TAG_TO_INTERMEDIATE_TYPE_MAP: dict[str, IntermediateBlockType] = {
    "t": "prose",
    "ul": "list",
    "ol": "list",
    "dl": "list",
    "reference": "references",
    "sourcecode": "sourcecode",
    "artwork": "artwork",
    "table": "table",
}
"""Translates valid modern xml2rfc v3 structural elements directly into parser tokens."""


class ReferenceMetadataDict(TypedDict):
    """Structured bibliographic metadata extracted from an RFC reference block."""

    anchor: str
    category: ReferenceCategory
    title: str | None
    target_url: NotRequired[str]
    doi: NotRequired[str]
    series_name: NotRequired[str]
    series_value: NotRequired[str]


class BlockMetadataDict(TypedDict):
    """Metadata constraints attached to an individual content block."""

    element_id: str | None
    normative_keywords: list[NormativeKeyword]
    section_number: NotRequired[str]
    reference_metadata: NotRequired[ReferenceMetadataDict]


class CanonicalBlockDict(TypedDict):
    """Flat dictionary representation of an extracted block before tree assembly."""

    rfc_id: int
    hierarchy_path: str
    block_type: IntermediateBlockType
    sourcecode_type: NotRequired[SourcecodeFormat]
    source_type: SourceType
    normalized_text: str
    source_fragment: str
    parsing_confidence: float
    metadata: BlockMetadataDict


class RFCMetadata(BaseModel):
    """Relational and temporal metadata defining the ground-truth context of an RFC."""

    rfc_number: int
    source_type: SourceType
    title: str
    published_at: RFCPublicationDate | None = None

    status: str | None = Field(
        default=None,
        description="Document status (e.g., INTERNET STANDARD, PROPOSED STANDARD).",
    )
    stream: str | None = Field(
        default=None,
        description="Publishing stream context (e.g., IETF, IAB, IRTF).",
    )

    authors: list[str] = Field(default_factory=list)
    obsoletes: list[int] = []
    updates: list[int] = []
    updated_by: list[int] = []
    protocol_family: str | None = Field(
        default=None,
        description=(
            "Placeholder for future semantic enrichment (e.g., Transport, Routing, Cryptography). "
            "Currently unpopulated as it is not natively provided by the baseline IETF index."
        ),
    )


class NormativeStatement(BaseModel):
    """Extracted BCP-14 normative requirement clause."""

    keyword: NormativeKeyword
    statement_text: str

    actor: str | None = Field(
        default=None,
        description="Target protocol actor or system subject responsible for execution.",
    )
    referenced_rfcs: list[int] = Field(
        default=[],
        description="Isolated cross-referenced protocol identifiers found within this requirement.",
    )

    @model_serializer(mode="wrap")
    def serialize_model(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Intercepts model serialization to strip out the 'actor' field if it is uninitialized.

        This optimizes downstream database storage sizing while ensuring that fields like
        'protocol_family' retain explicit null markers at the root level.

        Args:
            handler (SerializerFunctionWrapHandler): The default Pydantic core serialization logic.

        Returns:
            dict[str, Any]: The cleaned model state dictionary missing the null 'actor' property.
        """
        data = handler(self)
        if data.get("actor") is None:
            data.pop("actor", None)
        return data


class Block(BaseModel):
    """Atomic content unit preserving raw source data and normalized text layouts."""

    block_id: str
    block_type: BlockType

    sourcecode_type: SourcecodeFormat | None = Field(
        default=None,
        description="Language format if explicitly declared by the source file.",
    )

    source_fragment: str
    normalized_text: str
    parsing_confidence: float = Field(ge=0.0, le=1.0)
    normative_statements: list[NormativeStatement]

    @model_serializer(mode="wrap")
    def serialize_model(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Intercepts model serialization to strip out 'sourcecode_type' if it is uninitialized.

        Prevents sparse structural block types from padding output payloads with redundant
        null variables across massive ingestion sets.

        Args:
            handler (SerializerFunctionWrapHandler): The default Pydantic core serialization logic.

        Returns:
            dict[str, Any]: The cleaned model state dictionary missing the null 'sourcecode_type' property.
        """
        data = handler(self)
        if data.get("sourcecode_type") is None:
            data.pop("sourcecode_type", None)
        return data


class Section(BaseModel):
    """Structural node within an RFC hierarchy containing contiguous child blocks."""

    section_id: str
    title: str
    hierarchy_path: list[str]
    section_depth: int
    blocks: list[Block]


class NormalizedRFC(BaseModel):
    """Root canonical artifact representing a fully parsed and structured RFC document."""

    rfc_id: int
    metadata: RFCMetadata
    sections: list[Section]
    preface_blocks: list[Block] = []

    def save_to_disk(self, filepath: Path) -> None:
        """Serializes the canonical RFC artifact to a JSON file on disk with strict field routing.

        Omit sparse structural block fields (`sourcecode_type`, `actor`) while preserving
        explicit null markers for relational metadata parameters (`protocol_family`).
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.model_dump_json(indent=2), encoding="utf-8")
