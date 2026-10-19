"""Strict, privacy-preserving input parsing for the shape gate."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

MAX_COMPONENT = Decimal("1000000000")
MAX_VECTOR_LENGTH = 100000


class InputError(Exception):
    """An invalid input, with a safe message suitable only for internal use."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputError("duplicate key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise InputError("non-finite number")


def _loads_json(text: str, *, decimals: bool) -> Any:
    try:
        return json.loads(
            text,
            parse_int=Decimal if decimals else int,
            parse_float=Decimal,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, InvalidOperation, ValueError, OverflowError) as exc:
        raise InputError("invalid JSON") from exc


def _read_utf8(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise InputError("unreadable UTF-8 input") from exc


def parse_policy(path: Path) -> int:
    text = _read_utf8(path)
    value = _loads_json(text, decimals=False)
    if not isinstance(value, dict) or set(value) != {"dimension"}:
        raise InputError("invalid policy schema")
    dimension = value["dimension"]
    if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= MAX_VECTOR_LENGTH:
        raise InputError("invalid dimension")
    return dimension


def parse_rows(path: Path, dimension: int) -> tuple[int, list[int], list[int]]:
    text = _read_utf8(path)
    if text == "":
        return 0, [], []
    # Only CR and LF delimit records. Unicode separators remain ordinary data.
    parts = re.split(r"\r\n|\r|\n", text)
    if text.endswith(("\r", "\n")):
        parts.pop()  # A single final line terminator is permitted.
    wrong: list[int] = []
    zero: list[int] = []
    count = 0
    seen_ids: set[str] = set()
    for line_number, line in enumerate(parts, 1):
        if not line.strip(" \t"):
            raise InputError("blank row")
        row = _loads_json(line, decimals=True)
        if not isinstance(row, dict) or set(row) != {"id", "vector"}:
            raise InputError("invalid row schema")
        identifier, vector = row["id"], row["vector"]
        if not isinstance(identifier, str) or identifier == "":
            raise InputError("invalid identifier")
        # Keep exact strings: canonically equivalent Unicode IDs remain distinct.
        if identifier in seen_ids:
            raise InputError("duplicate identifier")
        seen_ids.add(identifier)
        if not isinstance(vector, list) or len(vector) > MAX_VECTOR_LENGTH:
            raise InputError("invalid vector")
        for component in vector:
            if isinstance(component, bool) or not isinstance(component, Decimal):
                raise InputError("invalid component")
            if not component.is_finite() or component.copy_abs() > MAX_COMPONENT:
                raise InputError("component outside domain")
        count += 1
        if len(vector) != dimension:
            wrong.append(line_number)
        if vector and all(component.is_zero() for component in vector):
            zero.append(line_number)
    return count, wrong, zero



def inspect_inputs(rows_path: Path, policy_path: Path) -> tuple[dict[str, Any], bool]:
    """Parse both files even when one fails; return safe findings and validity."""
    try:
        dimension = parse_policy(policy_path)
        policy_error = False
    except InputError:
        dimension = 1  # Safe placeholder to let the rows parser run.
        policy_error = True
    try:
        count, wrong, zero = parse_rows(rows_path, dimension)
        rows_error = False
    except InputError:
        count, wrong, zero = 0, [], []
        rows_error = True
    if policy_error or rows_error:
        return {"status": "invalid_input"}, False
    issues = bool(wrong or zero)
    return {
        "status": "shape_issues" if issues else "valid",
        "record_count": count,
        "wrong_dimension_lines": wrong,
        "zero_vector_lines": zero,
    }, issues
