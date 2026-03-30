"""Tests for training/eval_schema.py"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from eval_schema import (
    PROVENANCE_FIELDS,
    VALID_BOUNDARY_PAIRS,
    VALID_CATEGORIES,
    diversity_report,
    validate_file,
    validate_provenance,
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

    def test_accepts_extra_provenance_fields(self):
        """validate_sample should accept samples with extra provenance fields."""
        sample = {
            "text": "hello world",
            "expected_category": "prose",
            "boundary_pair": None,
            "model": "gpt-4",
            "temperature": 0.7,
            "prompt_template": "default",
            "content_domain": "news",
            "length_bucket": "short",
            "reasoning_mode": False,
            "sub_type": "article",
        }
        assert validate_sample(sample) is True


# ---------------------------------------------------------------------------
# PROVENANCE_FIELDS constant
# ---------------------------------------------------------------------------


class TestProvenanceFields:
    def test_provenance_fields_constant(self):
        assert PROVENANCE_FIELDS == {
            "model",
            "temperature",
            "prompt_template",
            "content_domain",
            "length_bucket",
            "reasoning_mode",
            "sub_type",
        }


# ---------------------------------------------------------------------------
# validate_provenance
# ---------------------------------------------------------------------------


class TestValidateProvenance:
    def _valid_provenance(self) -> dict:
        return {
            "model": "gpt-4",
            "temperature": 0.7,
            "prompt_template": "default",
            "content_domain": "news",
            "length_bucket": "short",
            "reasoning_mode": False,
            "sub_type": "article",
        }

    def test_valid_provenance(self):
        assert validate_provenance(self._valid_provenance()) is True

    def test_valid_provenance_int_temperature(self):
        sample = self._valid_provenance()
        sample["temperature"] = 1
        assert validate_provenance(sample) is True

    def test_valid_provenance_temperature_zero(self):
        sample = self._valid_provenance()
        sample["temperature"] = 0.0
        assert validate_provenance(sample) is True

    def test_valid_provenance_temperature_two(self):
        sample = self._valid_provenance()
        sample["temperature"] = 2.0
        assert validate_provenance(sample) is True

    def test_missing_model(self):
        sample = self._valid_provenance()
        del sample["model"]
        assert validate_provenance(sample) is False

    def test_missing_temperature(self):
        sample = self._valid_provenance()
        del sample["temperature"]
        assert validate_provenance(sample) is False

    def test_missing_sub_type(self):
        sample = self._valid_provenance()
        del sample["sub_type"]
        assert validate_provenance(sample) is False

    def test_empty_model(self):
        sample = self._valid_provenance()
        sample["model"] = ""
        assert validate_provenance(sample) is False

    def test_model_not_string(self):
        sample = self._valid_provenance()
        sample["model"] = 123
        assert validate_provenance(sample) is False

    def test_temperature_too_low(self):
        sample = self._valid_provenance()
        sample["temperature"] = -0.1
        assert validate_provenance(sample) is False

    def test_temperature_too_high(self):
        sample = self._valid_provenance()
        sample["temperature"] = 2.1
        assert validate_provenance(sample) is False

    def test_temperature_not_numeric(self):
        sample = self._valid_provenance()
        sample["temperature"] = "hot"
        assert validate_provenance(sample) is False

    def test_invalid_length_bucket(self):
        sample = self._valid_provenance()
        sample["length_bucket"] = "tiny"
        assert validate_provenance(sample) is False

    def test_all_length_buckets_accepted(self):
        for bucket in ("short", "medium", "long"):
            sample = self._valid_provenance()
            sample["length_bucket"] = bucket
            assert validate_provenance(sample) is True, f"Bucket {bucket} should be valid"

    def test_reasoning_mode_not_bool(self):
        sample = self._valid_provenance()
        sample["reasoning_mode"] = "yes"
        assert validate_provenance(sample) is False

    def test_empty_prompt_template(self):
        sample = self._valid_provenance()
        sample["prompt_template"] = ""
        assert validate_provenance(sample) is False

    def test_empty_content_domain(self):
        sample = self._valid_provenance()
        sample["content_domain"] = ""
        assert validate_provenance(sample) is False

    def test_empty_sub_type(self):
        sample = self._valid_provenance()
        sample["sub_type"] = ""
        assert validate_provenance(sample) is False


# ---------------------------------------------------------------------------
# diversity_report
# ---------------------------------------------------------------------------


class TestDiversityReport:
    def _make_sample(self, **overrides) -> dict:
        base = {
            "text": "hello world",
            "expected_category": "prose",
            "boundary_pair": None,
            "model": "gpt-4",
            "temperature": 0.7,
            "prompt_template": "default",
            "content_domain": "news",
            "length_bucket": "short",
            "reasoning_mode": False,
            "sub_type": "article",
        }
        base.update(overrides)
        return base

    def _write_jsonl(self, samples: list[dict], path: str):
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    def test_basic_diversity_report(self, tmp_path):
        fpath = str(tmp_path / "data.jsonl")
        self._write_jsonl(
            [
                self._make_sample(sub_type="article", model="gpt-4", temperature=0.7, content_domain="news", length_bucket="short"),
                self._make_sample(sub_type="article", model="claude-3", temperature=1.0, content_domain="science", length_bucket="medium"),
                self._make_sample(sub_type="essay", model="gpt-4", temperature=0.7, content_domain="news", length_bucket="long"),
            ],
            fpath,
        )
        report = diversity_report(fpath)
        assert report["total_samples"] == 3

        # article sub_type
        article = report["per_sub_type"]["article"]
        assert article["model_distribution"] == {"gpt-4": 1, "claude-3": 1}
        assert article["temperature_values"] == {0.7, 1.0}
        assert article["template_count"] == 1
        assert article["domain_count"] == 2
        assert article["length_buckets"] == {"short", "medium"}

        # essay sub_type
        essay = report["per_sub_type"]["essay"]
        assert essay["model_distribution"] == {"gpt-4": 1}
        assert essay["temperature_values"] == {0.7}
        assert essay["template_count"] == 1
        assert essay["domain_count"] == 1
        assert essay["length_buckets"] == {"long"}

    def test_empty_file(self, tmp_path):
        fpath = str(tmp_path / "empty.jsonl")
        with open(fpath, "w") as f:
            pass
        report = diversity_report(fpath)
        assert report["total_samples"] == 0
        assert report["per_sub_type"] == {}


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
