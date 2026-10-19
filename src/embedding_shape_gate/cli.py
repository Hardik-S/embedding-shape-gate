"""Console entry point with deliberately fixed, non-sensitive output."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .validator import inspect_inputs

HELP = """usage: embedding-shape-gate ROWS.jsonl POLICY.json

Check synthetic embedding fixture dimensions and all-zero vectors offline.
Input errors produce a fixed redacted result. No vectors or identifiers are emitted.
"""
INVALID = {"status": "invalid_input"}


def main() -> int:
    args = sys.argv[1:]
    if args in (["-h"], ["--help"]):
        sys.stdout.write(HELP)
        return 0
    if len(args) != 2 or any(arg.startswith("-") for arg in args):
        _write(INVALID)
        return 2
    try:
        result, issues = inspect_inputs(Path(args[0]), Path(args[1]))
    except Exception:
        # Never expose exception text, paths, IDs, vectors or source values.
        result, issues = INVALID, False
    _write(result)
    if result is INVALID or result == INVALID:
        return 2
    return 1 if issues else 0


def _write(value: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
