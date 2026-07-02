import math

import numpy as np
import pyarrow as pa

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

    # Garbage / Missing traps
    assert _safe_required_int("bad_string") == -1
    assert _safe_required_int(None) == -1
    assert _safe_required_int([]) == -1


def test_safe_optional_int() -> None:
    assert _safe_optional_int(404) == 404
    assert _safe_optional_int("404") == 404
    assert _safe_optional_int(math.pi) == 3

    # Garbage / Missing traps
    assert _safe_optional_int("bad_string") is None
    assert _safe_optional_int(None) is None
    assert _safe_optional_int({}) is None


def test_build_lance_table() -> None:
    # 1. Create a dummy record with missing optional fields
    records = [
        {
            "chunk_id": "rfc9999-sec1",
            "rfc_number": "9999",  # String that should be coerced
            "rfc_title": "Test Protocol",
            "status": "PROPOSED STANDARD",
            # rfc_year missing to test optional
            # rfc_month missing to test optional
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

    # 2. Mock a 256-dimensional PyArrow vector array
    fake_vector = np.random.rand(VECTOR_DIMENSIONS).astype(np.float32)
    vector_arrow_array = pa.FixedSizeListArray.from_arrays(
        pa.array(fake_vector, type=pa.float32()), VECTOR_DIMENSIONS
    )

    # 3. Build the table
    table = build_lance_table(records, vector_arrow_array)

    # 4. Assertions
    assert isinstance(table, pa.Table)
    assert table.num_rows == 1

    # Verify coercion logic triggered
    assert table["rfc_number"][0].as_py() == 9999

    # Verify JSON serialization of normative statements
    statements_json = table["normative_statements_json"][0].as_py()
    assert "MUST" in statements_json

    # Verify optional fields gracefully fell back to null/None
    assert table["rfc_year"][0].as_py() is None
    assert table["sourcecode_type"][0].as_py() is None
