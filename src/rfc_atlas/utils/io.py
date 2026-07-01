"""I/O utilities for the RFC Atlas pipeline."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


@contextmanager
def atomic_write(
    filepath: Path, mode: str = "w", encoding: str | None = "utf-8"
) -> Generator[IO[Any], None, None]:
    """Context manager for crash-safe atomic file writes.

    Yields:
        A temporary file handle. On successful exit, the temporary file
        is atomically renamed to the target filepath. On exception, the
        temporary file is discarded.
    """
    tmp_path = filepath.with_name(f"{filepath.name}.tmp")

    if "b" in mode:
        encoding = None

    try:
        with tmp_path.open(mode, encoding=encoding) as f:
            yield f
        Path(tmp_path).replace(filepath)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
