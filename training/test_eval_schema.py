"""Tests for training/eval_schema.py"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from eval_schema import (
    VALID_BOUNDARY_PAIRS,
    VALID_CATEGORIES,
    validate_file,
    validate_sample,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_valid_categories(self):
        assert VALID_CATEGORIES == {"prose", "code", "structured", "artifact"}

    def test_valid_boundary_pairs(self):
        assert VALID_BOUNDARY_PAIRS == {
            "prose_code",
            "prose_structured",
            "prose_artifact",
            "code_structured",
            "code_artifact",
            "structured_artifact",
        }


# ---------------------------------------------------------------------------
# validate_sample
# ---------------------------------------------------------------------------


class TestValidateSample:
    def test_valid_sample_no_boundary(self):
        sample = {
            "text": "hello world",
            "expected_category": "prose",
            "boundary_pair": None,
        }
        assert validate_sample(sample) is True

    def test_valid_sample_with_boundary(self):
        sample = {
            "text": "def foo(): pass",
            "expected_category": "code",
            "boundary_pair": "prose_code",
        }
        assert validate_sample(sample) is True

    def test_valid_sample_boundary_second_category(self):
        """expected_category can be either side of the boundary pair."""
        sample = {
            "text": "some prose text here",
            "expected_category": "prose",
            "boundary_pair": "prose_code",
        }
        assert validate_sample(sample) is True

    def test_missing_text_field(self):
        sample = {"expected_category": "prose", "boundary_pair": None}
        assert validate_sample(sample) is False

    def test_empty_text_field(self):
        sample = {
            "text": "",
            "expected_category": "prose",
            "boundary_pair": None,
        }
        assert validate_sample(sample) is False

    def test_text_not_string(self):
        sample = {
            "text": 123,
            "expected_category": "prose",
            "boundary_pair": None,
        }
        assert validate_sample(sample) is False

    def test_invalid_category(self):
        sample = {
            "text": "hello",
            "expected_category": "unknown",
            "boundary_pair": None,
        }
        assert validate_sample(sample) is False

    def test_missing_category(self):
        sample = {"text": "hello", "boundary_pair": None}
        assert validate_sample(sample) is False

    def test_invalid_boundary_pair(self):
        sample = {
            "text": "hello",
            "expected_category": "prose",
            "boundary_pair": "invalid_pair",
        }
        assert validate_sample(sample) is False

    def test_boundary_pair_category_mismatch(self):
        """Category must be one of the two in the boundary pair."""
        sample = {
            "text": "hello",
            "expected_category": "artifact",
            "boundary_pair": "prose_code",
        }
        assert validate_sample(sample) is False

    def test_boundary_pair_missing_key_treated_as_none(self):
        """If boundary_pair key is absent, treat as None (valid)."""
        sample = {"text": "hello", "expected_category": "prose"}
        assert validate_sample(sample) is True

    def test_all_categories_accepted(self):
        for cat in VALID_CATEGORIES:
            sample = {
                "text": "some text",
                "expected_category": cat,
                "boundary_pair": None,
            }
            assert validate_sample(sample) is True, f"Category {cat} should be valid"

    def test_all_boundary_pairs_accepted(self):
        for pair in VALID_BOUNDARY_PAIRS:
            cat = pair.split("_")[0]
            sample = {
                "text": "some text",
                "expected_category": cat,
                "boundary_pair": pair,
            }
            assert validate_sample(sample) is True, f"Pair {pair} with {cat} should be valid"


# ---------------------------------------------------------------------------
# validate_file
# ---------------------------------------------------------------------------


class TestValidateFile:
    def _write_jsonl(self, lines: list[dict], path: str):
        with open(path, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

    def test_valid_file(self, tmp_path):
        fpath = str(tmp_path / "valid.jsonl")
        self._write_jsonl(
            [
                {"text": "hello", "expected_category": "prose", "boundary_pair": None},
                {"text": "def x(): pass", "expected_category": "code", "boundary_pair": "prose_code"},
            ],
            fpath,
        )
        valid_count, errors = validate_file(fpath)
        assert valid_count == 2
        assert errors == []

    def test_file_with_invalid_lines(self, tmp_path):
        fpath = str(tmp_path / "mixed.jsonl")
        self._write_jsonl(
            [
                {"text": "hello", "expected_category": "prose", "boundary_pair": None},
                {"text": "", "expected_category": "prose", "boundary_pair": None},
                {"expected_category": "prose", "boundary_pair": None},
            ],
            fpath,
        )
        valid_count, errors = validate_file(fpath)
        assert valid_count == 1
        assert len(errors) == 2

    def test_file_with_malformed_json(self, tmp_path):
        fpath = str(tmp_path / "bad.jsonl")
        with open(fpath, "w") as f:
            f.write('{"text": "ok", "expected_category": "prose", "boundary_pair": null}\n')
            f.write("not json at all\n")
        valid_count, errors = validate_file(fpath)
        assert valid_count == 1
        assert len(errors) == 1
        assert "line 2" in errors[0].lower() or "2" in errors[0]

    def test_empty_file(self, tmp_path):
        fpath = str(tmp_path / "empty.jsonl")
        with open(fpath, "w") as f:
            pass
        valid_count, errors = validate_file(fpath)
        assert valid_count == 0
        assert errors == []


# ---------------------------------------------------------------------------
# __main__ block
# ---------------------------------------------------------------------------


class TestMainBlock:
    def test_main_with_valid_file(self, tmp_path):
        fpath = str(tmp_path / "test.jsonl")
        with open(fpath, "w") as f:
            f.write(json.dumps({"text": "hi", "expected_category": "prose", "boundary_pair": None}) + "\n")

        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "eval_schema", fpath],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        assert result.returncode == 0
        assert "1" in result.stdout  # should mention valid count

    def test_main_with_invalid_file(self, tmp_path):
        fpath = str(tmp_path / "bad.jsonl")
        with open(fpath, "w") as f:
            f.write(json.dumps({"text": "", "expected_category": "prose"}) + "\n")

        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "eval_schema", fpath],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        assert result.returncode == 1
