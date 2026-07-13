import math

import numpy as np
import pyarrow as pa
import pytest

from rfc_atlas.vector_store.schema import (
    VECTOR_DIMENSIONS,
    _safe_optional_int,  # pyright: ignore[reportPrivateUsage]
    _safe_required_int,  # pyright: ignore[reportPrivateUsage]
    build_lance_table,
)


def test_safe_required_int() -> None:
    assert _safe_required_int(404) == 404
    assert _safe_required_int("404") == 404
    assert _safe_required_int(math.pi) == 3

    with pytest.raises(ValueError, match="Cannot cast 'bad_string' to int"):
        _safe_required_int("bad_string")
    with pytest.raises(ValueError, match=r"Required integer field is missing \(None\)"):
        _safe_required_int(None)
    with pytest.raises(
        ValueError, match="Invalid type for required integer: <class 'list'>"
    ):
        _safe_required_int([])


def test_safe_optional_int() -> None:
    assert _safe_optional_int(404) == 404
    assert _safe_optional_int("404") == 404
    assert _safe_optional_int(math.pi) == 3

    # Garbage / Missing traps
    assert _safe_optional_int("bad_string") is None
    assert _safe_optional_int(None) is None
    assert _safe_optional_int({}) is None


def test_build_lance_table() -> None:
    records = [
        {
            "chunk_id": "rfc9999-sec1",
            "rfc_number": "9999",
            "rfc_title": "Test Protocol",
            "status": "PROPOSED STANDARD",
            "stream": "IETF",
            "obsoletes": [],
            "updated_by": [10000],
            "block_type": "paragraph",
            "table_route": "prose",
            "hierarchy_path": "Introduction",
            "text_payload": "This is a test chunk.",
            "parsing_confidence": 0.95,
            "normative_statements": [{"keyword": "MUST", "statement_text": "Test"}],
        }
    ]

    fake_vector = np.random.rand(VECTOR_DIMENSIONS).astype(np.float32)
    vector_arrow_array = pa.FixedSizeListArray.from_arrays(
        pa.array(fake_vector, type=pa.float32()), VECTOR_DIMENSIONS
    )

    table = build_lance_table(records, vector_arrow_array)

    assert isinstance(table, pa.Table)
    assert table.num_rows == 1
    assert table["rfc_number"][0].as_py() == 9999

    # Verify array mapping definitions and values (#31)
    assert table["obsoletes"][0].as_py() == []
    assert table["updated_by"][0].as_py() == [10000]
    assert pa.types.is_list(table["updated_by"].type)
    assert table["updated_by"].type.value_field.name == "element"

    # Verify optional fields gracefully fell back to null/None (#33)
    assert table["rfc_year"][0].as_py() is None
    assert table["rfc_month"][0].as_py() is None
    assert table["sourcecode_type"][0].as_py() is None
