# Embedding Shape Gate

An offline, standard-library CLI that checks **synthetic** embedding fixture rows for expected vector length and nonempty all-zero vectors before an indexing step. It does not create embeddings or assess vector meaning, similarity, norms, security, or novelty.

Requires Python 3.10 or newer. Install into the active environment with `python -m pip install .`, then run:

```text
embedding-shape-gate examples/rows.jsonl examples/policy.json
```

The example deliberately contains a zero vector and a wrong-dimension vector, so its result is `shape_issues` with physical line numbers. Input errors produce only `{"status":"invalid_input"}` on stdout and exit 2; shape findings exit 1; clean inputs exit 0. Help is available with `embedding-shape-gate --help`.

See [CONTRACT.md](CONTRACT.md) for the frozen input, numeric, output, and privacy contract. Memory use grows with input size; there are no runtime dependencies.
