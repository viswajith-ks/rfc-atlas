"""Strict PyArrow schema definitions and adapters for LanceDB columnar storage."""

import json
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa

VECTOR_DIMENSIONS: int = 256
EPSILON: float = 1e-12

LANCE_CHUNK_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string(), nullable=False),
    pa.field("rfc_number", pa.int32(), nullable=False),
    pa.field("rfc_title", pa.string(), nullable=True),
    pa.field("status", pa.string(), nullable=True),
    pa.field("rfc_year", pa.int16(), nullable=True),
    pa.field("rfc_month", pa.int8(), nullable=True),
    pa.field("stream", pa.string(), nullable=True),
    pa.field("obsoletes", pa.list_(pa.field("element", pa.int32())), nullable=False),
    pa.field("updated_by", pa.list_(pa.field("element", pa.int32())), nullable=False),
    pa.field("block_type", pa.string(), nullable=False),
    pa.field("table_route", pa.string(), nullable=False),
    pa.field("hierarchy_path", pa.string(), nullable=False),
    pa.field("text_payload", pa.string(), nullable=False),
    pa.field("sourcecode_type", pa.string(), nullable=True),
    pa.field("parsing_confidence", pa.float32(), nullable=False),
    pa.field("normative_statements_json", pa.string(), nullable=False),
    pa.field(
        "vector",
        pa.list_(pa.field("element", pa.float32()), VECTOR_DIMENSIONS),
        nullable=False,
    ),
])


def _safe_required_int(val: object) -> int:
    """Defensive casting gatekeeper for non-nullable integer columns.

    Args:
        val (object): The raw parsed value from the JSON record.

    Returns:
        int: The safely cast integer.

    Raises:
        ValueError: If the value is missing, None, or fails to cast.
    """
    if val is None:
        e = "Required integer field is missing (None)"
        raise ValueError(e)
    if isinstance(val, (int, float, str)):
        try:
            return int(val)
        except ValueError as e:
            err = f"Cannot cast '{val}' to int"
            raise ValueError(err) from e
    e = f"Invalid type for required integer: {type(val)}"
    raise ValueError(e)


def _safe_optional_int(val: object) -> int | None:
    """Defensive casting gatekeeper for nullable integer columns.

    Args:
        val (object): The raw parsed value from the JSON record.

    Returns:
        int | None: The safely cast integer, or None if the value is missing or invalid.
    """
    if val is None:
        return None
    if isinstance(val, (int, float, str)):
        try:
            return int(val)
        except ValueError:
            return None
    return None


def normalize_and_convert_vectors(raw_embeddings: npt.NDArray[np.float32]) -> pa.Array:
    """Slices, L2-normalizes, and converts raw embeddings to PyArrow format.

    Args:
        raw_embeddings (npt.NDArray[np.float32]): The raw embeddings given by the model.

    Returns:
        pa.Array: A PyArrow FixedSizeListArray of the normalized vectors.
    """
    sliced = raw_embeddings[:, :VECTOR_DIMENSIONS]
    norms: npt.NDArray[np.float32] = np.sqrt(
        np.sum(sliced * sliced, axis=1, keepdims=True, dtype=np.float32)
    )
    norms[norms < EPSILON] = EPSILON
    normalized_vectors = sliced / norms

    flat_vector_data = normalized_vectors.ravel()
    return pa.FixedSizeListArray.from_arrays(
        pa.array(flat_vector_data, type=pa.float32()), VECTOR_DIMENSIONS
    )


def build_lance_table(
    records: list[dict[str, Any]],
    vector_arrow_array: pa.Array,
    schema: pa.Schema | None = None,
) -> pa.Table:
    """Transforms a batch of raw JSON dicts and pre-computed vectors to a PyArrow Table.

    Operates via zero-copy SIMD columnar array assembly.

    Args:
        records (list[dict[str, Any]]): The list of raw chunk dictionaries to pack.
        vector_arrow_array (pa.Array): The pre-computed PyArrow array containing
            dense vectors.
        schema (pa.Schema | None): Optional PyArrow schema override. Defaults to
            LANCE_CHUNK_SCHEMA.

    Returns:
        pa.Table: A constructed PyArrow Table ready for LanceDB persistence.
    """
    cols = [
        pa.array([r["chunk_id"] for r in records], type=pa.string()),
        pa.array(
            [_safe_required_int(r["rfc_number"]) for r in records], type=pa.int32()
        ),
        pa.array([r.get("rfc_title") for r in records], type=pa.string()),
        pa.array([r.get("status") for r in records], type=pa.string()),
        pa.array(
            [_safe_optional_int(r.get("rfc_year")) for r in records], type=pa.int16()
        ),
        pa.array(
            [_safe_optional_int(r.get("rfc_month")) for r in records], type=pa.int8()
        ),
        pa.array([r.get("stream") for r in records], type=pa.string()),
        pa.array(
            [r.get("obsoletes") or [] for r in records],
            type=pa.list_(pa.field("element", pa.int32())),
        ),
        pa.array(
            [r.get("updated_by") or [] for r in records],
            type=pa.list_(pa.field("element", pa.int32())),
        ),
        pa.array([r["block_type"] for r in records], type=pa.string()),
        pa.array([r["table_route"] for r in records], type=pa.string()),
        pa.array([r["hierarchy_path"] for r in records], type=pa.string()),
        pa.array([r["text_payload"] for r in records], type=pa.string()),
        pa.array([r.get("sourcecode_type") for r in records], type=pa.string()),
        pa.array([r["parsing_confidence"] for r in records], type=pa.float32()),
        pa.array(
            [json.dumps(r.get("normative_statements", [])) for r in records],
            type=pa.string(),
        ),
        vector_arrow_array,
    ]

    return pa.Table.from_arrays(cols, schema=schema or LANCE_CHUNK_SCHEMA)
