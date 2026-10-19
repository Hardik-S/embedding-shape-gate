from pathlib import Path

import pytest

from embedding_shape_gate.validator import InputError, inspect_inputs, parse_policy, parse_rows


def write(tmp_path: Path, rows: bytes, policy: bytes = b'{"dimension":2}'):
    r = tmp_path / "rows.jsonl"
    p = tmp_path / "policy.json"
    r.write_bytes(rows)
    p.write_bytes(policy)
    return r, p


def test_valid_and_independent_findings(tmp_path):
    r, p = write(tmp_path, b'{"id":"a","vector":[0,-0e99]}\n{"id":"b","vector":[0]}')
    result, issues = inspect_inputs(r, p)
    assert issues
    assert result == {"status": "shape_issues", "record_count": 2,
                      "wrong_dimension_lines": [2], "zero_vector_lines": [1, 2]}


def test_numeric_domain_exact_decimal_bound_and_exponents(tmp_path):
    r, p = write(tmp_path, b'{"id":"x","vector":[1e9,-1e9,1e-9999]}', b'{"dimension":3}')
    assert parse_rows(r, 4) == (1, [1], [])
    r.write_text('{"id":"x","vector":[1000000000.0000000000000000001]}', encoding="utf-8")
    with pytest.raises(InputError):
        parse_rows(r, 1)
    r.write_text('{"id":"x","vector":[1e999999999999999999999]}', encoding="utf-8")
    with pytest.raises(InputError):
        parse_rows(r, 1)


def test_empty_rows_and_empty_vector(tmp_path):
    r, p = write(tmp_path, b"")
    assert inspect_inputs(r, p) == ({"status": "valid", "record_count": 0,
                                     "wrong_dimension_lines": [], "zero_vector_lines": []}, False)
    r.write_bytes(b'{"id":"e","vector":[]}')
    assert inspect_inputs(r, p)[0]["wrong_dimension_lines"] == [1]
    assert inspect_inputs(r, p)[0]["zero_vector_lines"] == []


@pytest.mark.parametrize("ending", [b"\n", b"\r", b"\r\n"])
def test_supported_line_endings(tmp_path, ending):
    r, p = write(tmp_path, b'{"id":"a","vector":[1,2]}' + ending + b'{"id":"b","vector":[2,3]}' + ending)
    assert parse_rows(r, 2)[0] == 2


def test_duplicate_and_unicode_identifiers_are_exact(tmp_path):
    r, p = write(tmp_path, '{"id":"é","vector":[1]}\n{"id":"é","vector":[1]}'.encode(), b'{"dimension":1}')
    assert parse_rows(r, 1)[0] == 2
    r.write_text('{"id":"same","vector":[]}\n{"id":"same","vector":[]}', encoding="utf-8")
    with pytest.raises(InputError):
        parse_rows(r, 1)


@pytest.mark.parametrize("bad", [b'{"id":"x","vector":[true]}', b'{"id":"x","vector":[NaN]}',
                                   b'{"id":"x","vector":[Infinity]}', b'{"id":"x","vector":[[1]]}',
                                   b'{"id":"x","vector":[1],"extra":0}',
                                   b'{"id":"x","vector":[1],"vector":[1]}',
                                   b'{"id":"x","vector":[1]}\n\n'])
def test_malformed_rows_rejected(tmp_path, bad):
    r, p = write(tmp_path, bad, b'{"dimension":1}')
    with pytest.raises(InputError):
        parse_rows(r, 1)


@pytest.mark.parametrize("policy", [b'{"dimension":true}', b'{"dimension":1.0}', b'{"dimension":0}',
                                     b'{"dimension":100001}', b'{"dimension":1,"x":2}',
                                     b'{"dimension":1,"dimension":1}', b'\xff'])
def test_policy_strictness(tmp_path, policy):
    r, p = write(tmp_path, b"", policy)
    with pytest.raises(InputError):
        parse_policy(p)


def test_utf8_and_unicode_separator_not_record_delimiter(tmp_path):
    r, p = write(tmp_path, b'{"id":"x","vector":[1]}\xe2\x80\xa8{"id":"y","vector":[1]}', b'{"dimension":1}')
    with pytest.raises(InputError):
        parse_rows(r, 1)
    r.write_bytes(b'{"id":"x","vector":[1]}\xff')
    with pytest.raises(InputError):
        parse_rows(r, 1)


def test_both_parsers_are_attempted_on_invalid_inputs(tmp_path, monkeypatch):
    import embedding_shape_gate.validator as validator
    r, p = write(tmp_path, b"bad", b"bad")
    called = []
    original_policy, original_rows = validator.parse_policy, validator.parse_rows
    def policy(path):
        called.append("policy")
        return original_policy(path)
    def rows(path, dimension):
        called.append("rows")
        return original_rows(path, dimension)
    monkeypatch.setattr(validator, "parse_policy", policy)
    monkeypatch.setattr(validator, "parse_rows", rows)
    assert inspect_inputs(r, p) == ({"status": "invalid_input"}, False)
    assert called == ["policy", "rows"]


def test_bool_and_nonfinite_numeric_values(tmp_path):
    r, _ = write(tmp_path, b'{"id":"x","vector":[-0.0]}')
    assert parse_rows(r, 1)[2] == [1]
    for value in (b'true', b'NaN', b'Infinity'):
        r.write_bytes(b'{"id":"x","vector":[' + value + b']}')
        with pytest.raises(InputError):
            parse_rows(r, 1)

import json
import os
import subprocess
import sysconfig
from pathlib import Path


def test_installed_console_contract(tmp_path):
    scripts = Path(sysconfig.get_path("scripts"))
    executable = scripts / ("embedding-shape-gate.exe" if os.name == "nt" else "embedding-shape-gate")
    assert executable.is_file(), f"install the project before running this test: {executable}"
    rows = tmp_path / "private-rows.jsonl"
    policy = tmp_path / "private-policy.json"
    policy.write_text('{"dimension":2}', encoding="utf-8")

    rows.write_text('{"id":"private-id","vector":[1,2]}', encoding="utf-8")
    valid = subprocess.run([str(executable), str(rows), str(policy)], capture_output=True, text=True)
    assert (valid.returncode, valid.stdout, valid.stderr) == (0,
        '{"status":"valid","record_count":1,"wrong_dimension_lines":[],"zero_vector_lines":[]}\n', "")

    rows.write_text('{"id":"private-id","vector":[0]}', encoding="utf-8")
    issue = subprocess.run([str(executable), str(rows), str(policy)], capture_output=True, text=True)
    assert (issue.returncode, issue.stdout, issue.stderr) == (1,
        '{"status":"shape_issues","record_count":1,"wrong_dimension_lines":[1],"zero_vector_lines":[1]}\n', "")

    rows.write_text('{"id":"PRIVATE-ID","vector":[1],"secret":"PRIVATE-VALUE"}', encoding="utf-8")
    invalid = subprocess.run([str(executable), str(rows), str(policy)], capture_output=True, text=True)
    assert (invalid.returncode, invalid.stdout, invalid.stderr) == (2, '{"status":"invalid_input"}\n', "")
    assert "PRIVATE" not in invalid.stdout + invalid.stderr

    help_result = subprocess.run([str(executable), "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0 and "usage: embedding-shape-gate" in help_result.stdout
    assert help_result.stderr == ""
