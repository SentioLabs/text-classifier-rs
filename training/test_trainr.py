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

    def test_trainr_pipeline_run_help(self):
        result = subprocess.run(
            ["uv", "run", "trainr", "pipeline", "run", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        assert result.returncode == 0
        assert "--input" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--eval" in result.stdout
        assert "Run full pipeline" in result.stdout

    def test_trainr_pipeline_run_requires_input_and_output_dir(self):
        result = subprocess.run(
            ["uv", "run", "trainr", "pipeline", "run"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        assert result.returncode != 0
        assert "Missing" in result.stderr or "required" in result.stderr.lower()

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


# ---------------------------------------------------------------------------
# pipeline run command
# ---------------------------------------------------------------------------


class TestPipelineRunCommand:
    def test_run_invokes_featurize_dedup_train(self, tmp_path):
        """pipeline run chains featurize -> dedup -> train via ctx.invoke."""
        from click.testing import CliRunner

        from trainr.commands.pipeline import pipeline, featurize, dedup, train

        calls = []

        def mock_featurize_main(argv):
            calls.append(("featurize", argv))

        def mock_dedup_main():
            calls.append(("dedup",))

        def mock_train_main(argv):
            calls.append(("train", argv))

        with mock.patch("trainr.core.featurize.main", mock_featurize_main), \
             mock.patch("trainr.core.dedup.main", mock_dedup_main), \
             mock.patch("trainr.core.train.main", mock_train_main):
            runner = CliRunner()
            result = runner.invoke(pipeline, [
                "run",
                "--input", "/tmp/raw.parquet",
                "--output-dir", str(tmp_path),
            ])
            assert result.exit_code == 0, result.output + (result.exception and str(result.exception) or "")
            assert len(calls) == 3
            assert calls[0][0] == "featurize"
            assert calls[1][0] == "dedup"
            assert calls[2][0] == "train"

    def test_run_invokes_eval_and_analyze_when_eval_paths_given(self, tmp_path):
        """pipeline run chains eval and analyze when --eval is provided."""
        from click.testing import CliRunner

        from trainr.commands.pipeline import pipeline

        calls = []

        def mock_featurize_main(argv):
            calls.append(("featurize",))

        def mock_dedup_main():
            calls.append(("dedup",))

        def mock_train_main(argv):
            calls.append(("train",))

        def mock_eval_main(argv):
            calls.append(("eval", argv))

        def mock_analyze_main(argv):
            calls.append(("analyze", argv))

        with mock.patch("trainr.core.featurize.main", mock_featurize_main), \
             mock.patch("trainr.core.dedup.main", mock_dedup_main), \
             mock.patch("trainr.core.train.main", mock_train_main), \
             mock.patch("trainr.core.eval_onnx.main", mock_eval_main), \
             mock.patch("trainr.core.analyze_eval.main", mock_analyze_main):
            runner = CliRunner()
            result = runner.invoke(pipeline, [
                "run",
                "--input", "/tmp/raw.parquet",
                "--output-dir", str(tmp_path),
                "--eval", "/tmp/eval_clear.jsonl",
                "--eval", "/tmp/eval_boundary.jsonl",
            ])
            assert result.exit_code == 0, result.output + (result.exception and str(result.exception) or "")
            # featurize + dedup + train + eval + 2x analyze = 6 calls
            assert len(calls) == 6
            assert calls[3][0] == "eval"
            assert calls[4][0] == "analyze"
            assert calls[5][0] == "analyze"

    def test_run_skips_eval_when_no_eval_paths(self, tmp_path):
        """pipeline run does not invoke eval/analyze when --eval is not provided."""
        from click.testing import CliRunner

        from trainr.commands.pipeline import pipeline

        calls = []

        def mock_featurize_main(argv):
            calls.append(("featurize",))

        def mock_dedup_main():
            calls.append(("dedup",))

        def mock_train_main(argv):
            calls.append(("train",))

        with mock.patch("trainr.core.featurize.main", mock_featurize_main), \
             mock.patch("trainr.core.dedup.main", mock_dedup_main), \
             mock.patch("trainr.core.train.main", mock_train_main):
            runner = CliRunner()
            result = runner.invoke(pipeline, [
                "run",
                "--input", "/tmp/raw.parquet",
                "--output-dir", str(tmp_path),
            ])
            assert result.exit_code == 0
            assert len(calls) == 3  # Only featurize, dedup, train
