# Launch and verification

From the repository root, install as a non-editable package and exercise the installed command:

```text
python -m pip install .
embedding-shape-gate --help
embedding-shape-gate examples/rows.jsonl examples/policy.json
python -m pytest -q
```

To verify exit behavior, use valid, issue-containing, and malformed private input files. The command emits no input values, IDs, paths, dimensions, or exception text. Malformed inputs and command-line usage errors return the fixed redacted JSON result.
