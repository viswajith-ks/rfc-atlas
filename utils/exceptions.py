"""Domain exception hierarchy for the RFC Atlas ingestion and processing engine."""

from pathlib import Path


class RFCAtlasError(Exception):
    """Base exception for all domain-specific failures inside the RFC Atlas."""


# --- 1. Orchestration & Environment Errors ---


class SingletonViolationError(RFCAtlasError):
    """Raised when an attempt is made to instantiate a strict Singleton twice."""

    def __init__(self, class_name: str) -> None:
        super().__init__(
            f"Illegal secondary instantiation of singleton class: {class_name}"
        )


class UnsupportedHostOSError(RFCAtlasError):
    """Raised when the underlying OS kernel does not support required I/O primitives."""

    def __init__(self, platform_name: str) -> None:
        super().__init__(
            f"Operating system '{platform_name}' is unsupported. Linux is strictly required."
        )


class MetadataIndexCompilationError(RFCAtlasError):
    """Raised when the metadata lookup ledger fails to compile from raw sources."""

    def __init__(self, reason: Exception) -> None:
        super().__init__(f"Fatal error compiling metadata lookup index: {reason}")


# --- 2. Corpus & Dependency Errors ---


class CorpusDependencyError(RFCAtlasError):
    """Raised when a foundational protocol ledger or manifest cannot be located on disk."""

    def __init__(self, expected_path: Path | str, dependency_name: str) -> None:
        super().__init__(
            f"Required corpus dependency '{dependency_name}' missing at: {expected_path}"
        )


# --- 3. Parsing & Normalization Errors ---


class InvalidRFCNumberError(RFCAtlasError):
    """Raised when a protocol document claims an impossible or non-positive identifier."""

    def __init__(self, rfc_number: int) -> None:
        super().__init__(
            f"Invalid RFC identifier '{rfc_number}'. Identifier must be a positive integer."
        )


class MalformedFilenameError(RFCAtlasError):
    """Raised when an input document violates the strict 'rfcXXXX.ext' naming convention."""

    def __init__(self, filename: str) -> None:
        super().__init__(
            f"Filename '{filename}' violates canonical extraction pattern 'rfc[0-9]+.ext'."
        )


class MalformedXMLRootError(RFCAtlasError):
    """Raised when an input XML document lacks a valid numeric ID attribute."""

    def __init__(self, filepath: Path | str) -> None:
        super().__init__(
            f"Root element lacks a valid numeric RFC ID in XML document: {filepath}"
        )


class MalformedIndexXMLError(RFCAtlasError):
    """Raised when the global RFC Index XML file is structurally corrupt or unparsable."""

    def __init__(self, filepath: Path | str, parse_error: Exception) -> None:
        super().__init__(
            f"RFC index XML is structurally malformed at '{filepath}': {parse_error}"
        )


class MissingTelemetryLogError(RFCAtlasError, FileNotFoundError):
    """Raised when telemetry analysis is attempted before telemetry logs exist."""

    def __init__(self, log_path: Path | str) -> None:
        super().__init__(
            f"Error: Target log file '{log_path}' could not be discovered. "
            f"Verify that you successfully ran the ingestion pipeline framework first."
        )
