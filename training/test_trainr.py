"""Tests for trainr package scaffolding: CLI, shared/io, shared/api, core/schema."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


class TestCLI:
    def test_trainr_help_shows_subgroups(self):
        result = subprocess.run(
            [sys.executable, "-m", "trainr.cli"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent)},
        )
        # --help isn't needed; calling the group with no args shows help
        # Actually, click groups with no subcommand show help by default
        # Let's use --help explicitly
        result = subprocess.run(
            ["uv", "run", "trainr", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        assert result.returncode == 0
        assert "data" in result.stdout
        assert "pipeline" in result.stdout
        assert "eval" in result.stdout
        assert "audit" in result.stdout

    def test_trainr_data_help(self):
        result = subprocess.run(
            ["uv", "run", "trainr", "data", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        assert result.returncode == 0
        assert "Data sourcing" in result.stdout

    def test_trainr_pipeline_help(self):
        result = subprocess.run(
            ["uv", "run", "trainr", "pipeline", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        assert result.returncode == 0
        assert "Core training pipeline" in result.stdout

    def test_trainr_eval_help(self):
        result = subprocess.run(
            ["uv", "run", "trainr", "eval", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        assert result.returncode == 0
        assert "Model evaluation" in result.stdout

    def test_trainr_audit_help(self):
        result = subprocess.run(
            ["uv", "run", "trainr", "audit", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        assert result.returncode == 0
        assert "Label auditing" in result.stdout


# ---------------------------------------------------------------------------
# shared/io
# ---------------------------------------------------------------------------


class TestSharedIO:
    def test_read_write_jsonl_roundtrip(self, tmp_path):
        from trainr.shared.io import read_jsonl, write_jsonl

        records = [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]
        fpath = tmp_path / "test.jsonl"
        write_jsonl(records, fpath)
        result = read_jsonl(fpath)
        assert result == records

    def test_read_jsonl_skips_blank_lines(self, tmp_path):
        from trainr.shared.io import read_jsonl

        fpath = tmp_path / "test.jsonl"
        fpath.write_text('{"a": 1}\n\n{"a": 2}\n  \n')
        result = read_jsonl(fpath)
        assert len(result) == 2

    def test_write_jsonl_creates_file(self, tmp_path):
        from trainr.shared.io import write_jsonl

        fpath = tmp_path / "new.jsonl"
        write_jsonl([{"x": 1}], fpath)
        assert fpath.exists()
        lines = fpath.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_read_write_parquet_roundtrip(self, tmp_path):
        import polars as pl

        from trainr.shared.io import read_parquet, write_parquet

        df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        fpath = tmp_path / "test.parquet"
        write_parquet(df, fpath)
        result = read_parquet(fpath)
        assert result.shape == (3, 2)
        assert result["col1"].to_list() == [1, 2, 3]

    def test_read_jsonl_empty_file(self, tmp_path):
        from trainr.shared.io import read_jsonl

        fpath = tmp_path / "empty.jsonl"
        fpath.write_text("")
        result = read_jsonl(fpath)
        assert result == []


# ---------------------------------------------------------------------------
# shared/api
# ---------------------------------------------------------------------------


class TestSharedAPI:
    def test_get_anthropic_api_key_success(self):
        from trainr.shared.api import get_anthropic_api_key

        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-123"}):
            assert get_anthropic_api_key() == "test-key-123"

    def test_get_anthropic_api_key_missing(self):
        from trainr.shared.api import get_anthropic_api_key

        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                get_anthropic_api_key()

    def test_get_openrouter_api_key_success(self):
        from trainr.shared.api import get_openrouter_api_key

        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-key-456"}):
            assert get_openrouter_api_key() == "or-key-456"

    def test_get_openrouter_api_key_missing(self):
        from trainr.shared.api import get_openrouter_api_key

        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                get_openrouter_api_key()

    def test_get_openai_api_key_success(self):
        from trainr.shared.api import get_openai_api_key

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "oai-key-789"}):
            assert get_openai_api_key() == "oai-key-789"

    def test_get_openai_api_key_missing(self):
        from trainr.shared.api import get_openai_api_key

        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
                get_openai_api_key()


# ---------------------------------------------------------------------------
# core/schema (import path migration)
# ---------------------------------------------------------------------------


class TestCoreSchemaImport:
    def test_can_import_from_trainr_core_schema(self):
        from trainr.core.schema import (
            PROVENANCE_FIELDS,
            VALID_BOUNDARY_PAIRS,
            VALID_CATEGORIES,
            VALID_LENGTH_BUCKETS,
            diversity_report,
            validate_file,
            validate_provenance,
            validate_sample,
        )

        assert "prose" in VALID_CATEGORIES
        assert "prose_code" in VALID_BOUNDARY_PAIRS
        assert "model" in PROVENANCE_FIELDS
        assert "short" in VALID_LENGTH_BUCKETS
